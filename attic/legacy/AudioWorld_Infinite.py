#!/usr/bin/env python3
# Infinite-scroll Alien DJ – MVC with procedural tiles & glowing “generated” borders.

import os, sys, math, time, uuid, tempfile, wave, struct, threading, queue, hashlib, secrets, random
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import numpy as np
import pygame

# ============ Engine/window ============
SCREEN_W, SCREEN_H = 1200, 700
FPS = 60

# Tile size (world is an infinite grid of these)
TILE_W, TILE_H = 300, 350

# Pygame audio device rate (provider will resample to this)
ENGINE_SR = 44100

# ============ Startup/config ============
START_ZONE_COORD = (0, 0)  # (zx, zy)
START_ZONE_NAME  = "Scrapyard Funk"
START_THEME_WAV  = r"C:\Games\CosmicUnderground\inventory\rec_1756545018_Scrapyard Funk_d5ae11.wav"

AUTO_PREFETCH_NEIGHBORS = True           # N/E/S/W
PREFETCH_RADIUS = 1                      # ring radius for prefetch (1 = N/E/S/W only)
MAX_ZONE_CACHE = 48                      # soft cap on # of tiles we keep in memory

SESSION_SEED = secrets.randbits(32)      # affects names/params; fresh every run

# ============ Musical defaults ============
DEFAULT_BARS = 8
DEFAULT_TIMESIG = (4, 4)

# ============ Data types ============
@dataclass
class ZoneSpec:
    name: str
    bpm: int
    key_mode: str
    scene: str
    mood: str
    bars: int = DEFAULT_BARS
    timesig: Tuple[int, int] = DEFAULT_TIMESIG
    prompt_override: Optional[str] = None

@dataclass
class GeneratedLoop:
    wav_path: str
    duration_sec: float
    bpm: float
    key_mode: str
    prompt: str
    provider_meta: Dict = field(default_factory=dict)

@dataclass
class ZoneRuntime:
    coord: Tuple[int, int]              # (zx, zy)
    spec: ZoneSpec
    loop: Optional[GeneratedLoop] = None
    generating: bool = False
    error: Optional[str] = None
    last_seed: Optional[int] = None     # for debugging

@dataclass
class Player:
    x: float
    y: float
    speed: float = 6.0

# ============ Audio provider base ============
class AudioProvider:
    def generate(self, zone: ZoneSpec) -> GeneratedLoop: raise NotImplementedError
    @staticmethod
    def duration_for(zone: ZoneSpec) -> float:
        beats = zone.bars * zone.timesig[0]
        return beats * (60.0 / zone.bpm)

# ============ Stable Audio provider ============
# Use the LocalStableAudioProvider you already have from previous step (with the squeeze(0) fix).
# Paste it here if it lives in a different file, or import it. Minimal inline version:

class LocalStableAudioProvider(AudioProvider):
    def __init__(self):
        try:
            import torch
            from stable_audio_tools import get_pretrained_model
            self._torch = torch
            self._get_pretrained_model = get_pretrained_model
        except Exception as e:
            raise RuntimeError(
                "Stable Audio backend unavailable. Install inside your venv:\n"
                "  pip install --no-deps stable-audio-tools==0.0.19\n"
                "  pip install numpy einops soundfile huggingface_hub pygame\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
            ) from e
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        self._torch.set_num_threads(max(1, os.cpu_count() // 2))
        self.device = (
            "cuda" if self._torch.cuda.is_available() else
            ("mps" if getattr(self._torch.backends, "mps", None)
                     and self._torch.backends.mps.is_available() else "cpu")
        )
        print(f"[StableLocal] Loading model… (device={self.device})", flush=True)
        try:
            self.model, self.cfg = self._get_pretrained_model("stabilityai/stable-audio-open-small")
        except Exception as e:
            raise RuntimeError("Failed to load Stable Audio model.") from e
        self.model = self.model.to(self.device).eval()
        self.sr_model = int(self.cfg["sample_rate"])
        print(f"[StableLocal] Ready. sr={self.sr_model}", flush=True)
        self.tmpdir = tempfile.mkdtemp(prefix="alien_dj_local_")

    def _prompt(self, z: ZoneSpec) -> str:
        if z.prompt_override: return z.prompt_override
        return (f"{z.mood} alien scene: {z.scene}. "
                f"Groove-forward, clean downbeats, loopable {z.bars} bars. "
                f"{z.bpm} BPM, {z.key_mode}. Minimal silence at start/end.")

    def _ensure_length(self, wav_np: np.ndarray, sr: int, seconds_total: float) -> np.ndarray:
        target = int(round(seconds_total * sr))
        T = wav_np.shape[0]
        if T < target:
            reps = int(math.ceil(target / T))
            wav_np = np.tile(wav_np, (reps, 1))[:target]
        elif T > target:
            wav_np = wav_np[:target]
        fade = max(2, int(sr * 0.003))
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wav_np[:fade] *= ramp[:, None]
        wav_np[-fade:] *= ramp[::-1][:, None]
        return wav_np

    def _resample_linear(self, x: np.ndarray, sr_in: int, sr_out: int) -> Tuple[np.ndarray, int]:
        if sr_in == sr_out: return x, sr_in
        dur = x.shape[0] / sr_in
        tgt = int(round(dur * sr_out))
        t_in  = np.linspace(0, dur, x.shape[0], endpoint=False, dtype=np.float64)
        t_out = np.linspace(0, dur, tgt,       endpoint=False, dtype=np.float64)
        y = np.empty((tgt, x.shape[1]), dtype=np.float32)
        for ch in range(x.shape[1]):
            y[:, ch] = np.interp(t_out, t_in, x[:, ch]).astype(np.float32)
        return y, sr_out

    def generate(self, zone: ZoneSpec) -> GeneratedLoop:
        import soundfile as sf
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        seconds_total = AudioProvider.duration_for(zone)
        sample_size   = int(round(seconds_total * self.sr_model))
        seed = secrets.randbits(31)  # random every time (fresh run + regen)
        prompt = self._prompt(zone)
        print(f"[StableLocal] seed={seed} for zone '{zone.name}'", flush=True)

        t0 = time.time()
        with self._torch.inference_mode():
            audio = generate_diffusion_cond(
                self.model,
                steps=8, cfg_scale=1.0, sampler_type="pingpong",
                conditioning=[{"prompt": prompt, "seconds_total": seconds_total}],
                sample_size=sample_size, device=self.device, seed=seed
            )
        # (B, D, T) -> (T, D)
        audio = audio.squeeze(0).to(self._torch.float32).cpu()   # (2, T)
        audio = audio.transpose(0, 1).numpy().copy()             # (T, 2)
        peak = float(np.max(np.abs(audio))) or 1.0
        audio = (audio / peak) * 0.98
        audio = self._ensure_length(audio, self.sr_model, seconds_total)
        audio, out_sr = self._resample_linear(audio, self.sr_model, ENGINE_SR)

        out_path = os.path.join(self.tmpdir, f"{uuid.uuid4().hex}.wav")
        sf.write(out_path, audio, out_sr, subtype="PCM_16")
        print(f"[StableLocal] gen {zone.name} in {time.time()-t0:.2f}s → {os.path.basename(out_path)}", flush=True)
        return GeneratedLoop(out_path, seconds_total, zone.bpm, zone.key_mode, prompt,
                             {"backend":"stable_local","seed":seed,"sr":out_sr})

# ============ Audio playback & recording ============
class AudioPlayer:
    def __init__(self):
        pygame.mixer.init(frequency=ENGINE_SR, size=-16, channels=2, buffer=1024)
        self.boundary_event = pygame.USEREVENT + 1
        self.current: Optional[str] = None
        self.loop_ms = 0

    def play_loop(self, wav_path: str, duration_sec: float, fade_ms: int = 200):
        self.loop_ms = max(1, int(duration_sec * 1000))
        pygame.time.set_timer(self.boundary_event, self.loop_ms)
        pygame.mixer.music.load(wav_path)
        pygame.mixer.music.play(loops=-1, fade_ms=fade_ms)
        self.current = wav_path

    def stop(self, fade_ms: int = 200):
        pygame.mixer.music.fadeout(fade_ms)
        pygame.time.set_timer(self.boundary_event, 0)
        self.current = None
        self.loop_ms = 0

class Recorder:
    def __init__(self):
        self.dir = os.path.abspath("./inventory")
        os.makedirs(self.dir, exist_ok=True)
        self.thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.active_path: Optional[str] = None

    def is_recording(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, loop_path: str, max_seconds: float, meta: Dict):
        if self.is_recording(): return
        self.stop_flag.clear()
        fname = f"rec_{int(time.time())}_{meta.get('zone','zone')}_{uuid.uuid4().hex[:6]}.wav"
        out_path = os.path.join(self.dir, fname)
        self.active_path = out_path

        def _run():
            try:
                with wave.open(loop_path, "rb") as src, wave.open(out_path, "wb") as dst:
                    dst.setnchannels(src.getnchannels())
                    dst.setsampwidth(src.getsampwidth())
                    dst.setframerate(src.getframerate())
                    frames_total = int(max_seconds * src.getframerate())
                    chunk = 2048; written = 0
                    while written < frames_total and not self.stop_flag.is_set():
                        to_read = min(chunk, frames_total - written)
                        data = src.readframes(to_read)
                        if not data: break
                        dst.writeframes(data)
                        written += to_read
                print(f"[REC] saved {out_path}")
            except Exception as e:
                print(f"[REC] error: {e}")

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        if self.thread: self.thread.join(timeout=2.0)
        self.thread = None
        return self.active_path

# ============ World Model (infinite grid) ============
class WorldModel:
    def __init__(self):
        self.cols, self.rows = None, None    # unused in infinite mode
        self.player = Player(0, 0, speed=6.0)
        # cache of zones by coord
        self.zones: Dict[Tuple[int,int], ZoneRuntime] = {}
        self.current_coord = (0, 0)
        self._listeners = []

        # Place player at start tile center
        self.set_start_coord(START_ZONE_COORD)

    # ---- procedural spec generation ----
    def _spec_for(self, coord: Tuple[int,int]) -> ZoneSpec:
        zx, zy = coord
        rng = random.Random((SESSION_SEED << 1) ^ (zx * 73856093) ^ (zy * 19349663))

        # Names (procedural). Force (0,0) to Scrapyard Funk.
        if coord == START_ZONE_COORD:
            name = START_ZONE_NAME
        else:
            ADJ = ["Neon","Moss","Crystal","Chrome","Fungal","Slime","Luminous","Rusty","Electric","Gleaming","Velvet","Shatter"]
            LOC = ["Bazaar","Docks","Canyon","Arcade","Grove","Foundry","Plaza","Sprawl","Station","Labyrinth","Galleries","Causeway"]
            TWIST = ["Funk","Pulse","Reverb","Flux","Jitter","Drift","Riff","Chromatics","Echo","Parade","Breaks","Mallets"]
            name = f"{rng.choice(ADJ)} {rng.choice(LOC)}"
            # 1/3 chance to postfix a twist
            if rng.random() < 0.33: name = f"{name} {rng.choice(TWIST)}"

        # BPM, key/mode, mood, scene (stable per coord)
        bpm = rng.randrange(88, 141, 2)
        modes = ["major","minor","Dorian","Phrygian","Mixolydian","Lydian","Aeolian"]
        tonics = ["A","B","C","D","E","F","G","A♭","B♭","C♯","D♯","F♯","G♯"]
        key_mode = f"{rng.choice(tonics)} {rng.choice(modes)}"
        moods = ["energetic","calm","angry","triumphant","melancholy","playful","brooding","gritty","glittery"]
        mood = rng.choice(moods)
        scenes = [
            "glittering synth arps, clacky percussion, funky market",
            "dubby bass, foghorn pads, cavernous reverb",
            "glassy mallets, airy choirs, shimmering echoes",
            "breakbeat kit, gnarly clav riffs, grimey amp",
            "hand percussion, kalimba, wooden clicks",
            "trip-hop haze, vinyl crackle, moody Rhodes",
            "UKG shuffle, vocal chops, subby bass",
            "parade brass, clav and wah guitar",
            "holo-arcade bleeps, punchy drums, neon hum"
        ]
        scene = rng.choice(scenes)

        return ZoneSpec(name=name, bpm=bpm, key_mode=key_mode, scene=scene, mood=mood)

    def get_zone(self, coord: Tuple[int,int]) -> ZoneRuntime:
        if coord not in self.zones:
            spec = self._spec_for(coord)
            self.zones[coord] = ZoneRuntime(coord=coord, spec=spec)
        return self.zones[coord]

    def set_start_coord(self, coord: Tuple[int,int]):
        self.current_coord = coord
        self.player.x = coord[0]*TILE_W + TILE_W/2
        self.player.y = coord[1]*TILE_H + TILE_H/2

    def add_zone_changed_listener(self, fn): self._listeners.append(fn)

    # ---- movement & zone change ----
    def move_player(self, dx: float, dy: float):
        self.player.x += dx
        self.player.y += dy
        zx = math.floor(self.player.x / TILE_W)
        zy = math.floor(self.player.y / TILE_H)
        newc = (zx, zy)
        if newc != self.current_coord:
            old = self.current_coord
            self.current_coord = newc
            for fn in self._listeners: fn(old, newc)

# ============ View (render only) ============
class GameView:
    def __init__(self, model: WorldModel):
        self.m = model
        self.font = pygame.font.SysFont("consolas", 18)

    def _draw_glow(self, surf: pygame.Surface, rect: pygame.Rect, pulse_t: float):
        # purple glow: pulse brightness using sine; draw a few inflated outlines to fake glow
        glow_color = (170 + int(60* (0.5+0.5*math.sin(pulse_t*2.0))), 60, 220)  # pulsing purple
        # use a separate surface with per-pixel alpha
        glow = pygame.Surface((rect.w+16, rect.h+16), pygame.SRCALPHA)
        base = pygame.Rect(8, 8, rect.w, rect.h)
        for i, alpha in enumerate((90, 60, 30)):
            pygame.draw.rect(glow, (*glow_color, alpha), base.inflate(8+i*6, 8+i*6), width=3, border_radius=14)
        surf.blit(glow, (rect.x-8, rect.y-8))

    def draw(self, screen: pygame.Surface, record_armed: bool, recorder_active: bool):
        screen.fill((14,10,18))
        pulse_t = pygame.time.get_ticks()/1000.0

        # Camera: center on player
        cam_x = self.m.player.x - SCREEN_W/2
        cam_y = self.m.player.y - SCREEN_H/2

        # Which tiles visible?
        min_zx = math.floor(cam_x / TILE_W) - 1
        max_zx = math.floor((cam_x + SCREEN_W) / TILE_W) + 1
        min_zy = math.floor(cam_y / TILE_H) - 1
        max_zy = math.floor((cam_y + SCREEN_H) / TILE_H) + 1

        for zy in range(min_zy, max_zy+1):
            for zx in range(min_zx, max_zx+1):
                coord = (zx, zy)
                z = self.m.get_zone(coord)
                # screen rect for this tile
                sx = int(zx*TILE_W - cam_x); sy = int(zy*TILE_H - cam_y)
                rect = pygame.Rect(sx+2, sy+2, TILE_W-4, TILE_H-4)
                active = (coord == self.m.current_coord)
                base_col = (40, 42, 70) if not active else (70, 88, 140)
                pygame.draw.rect(screen, base_col, rect, border_radius=12)

                # Glow if audio generated
                if z.loop is not None and z.error is None:
                    self._draw_glow(screen, rect, pulse_t)

                # NPC dot
                pygame.draw.circle(screen, (220,200,140), (rect.centerx, rect.centery), 14)

                # labels
                name = self.font.render(z.spec.name, True, (240,240,240))
                sub  = self.font.render(f"{z.spec.mood} | {z.spec.bpm} BPM | {z.spec.key_mode}", True, (200,220,255))
                screen.blit(name, (rect.x+10, rect.y+10))
                screen.blit(sub,  (rect.x+10, rect.y+34))

                # status/footer
                if z.generating: st = "generating…"
                elif z.error:    st = f"error: {z.error[:26]}…"
                elif not z.loop: st = "not generated"
                else:            st = os.path.basename(z.loop.wav_path)
                screen.blit(self.font.render(st, True, (180,180,180)), (rect.x+10, rect.bottom - 28))

        # Player
        pygame.draw.circle(screen, (255,100,120), (int(self.m.player.x - cam_x), int(self.m.player.y - cam_y)), 10)

        # HUD
        hud = [
            "Move: WASD/Arrows | G regenerate current | M cycle mood | E edit mood | R arm/stop record | I inventory | Esc quit"
        ]
        if record_armed:    hud.append("REC ARMED: will start at next loop boundary.")
        if recorder_active: hud.append("RECORDING… R to stop manually (or auto-stop at loop end).")
        y = SCREEN_H - 22*len(hud) - 8
        for line in hud:
            screen.blit(self.font.render(line, True, (240,240,240)), (10, y))
            y += 22

# ============ Audio Service ============
class AudioService:
    def __init__(self, model: WorldModel):
        self.m = model
        self.provider = LocalStableAudioProvider()  # no fallback
        self.player = AudioPlayer()
        self.recorder = Recorder()
        self.auto_stop_at_end = True
        self.record_armed = False
        self.gen_queue: "queue.Queue[Tuple[int,int]]" = queue.Queue()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

        self.m.add_zone_changed_listener(self.on_zone_changed)

        # Initial tile: preload theme if present, else generate
        if not self._maybe_preload(self.m.current_coord):
            self.request_generate(self.m.current_coord)
        if AUTO_PREFETCH_NEIGHBORS:
            self.prefetch_ring(self.m.current_coord, PREFETCH_RADIUS)

    # ---------- generation thread ----------
    def _worker(self):
        while True:
            coord = self.gen_queue.get()
            if coord is None: return
            z = self.m.get_zone(coord)
            if z.generating: continue
            z.generating = True; z.error = None
            try:
                loop = self.provider.generate(z.spec)
                z.loop = loop
                if coord == self.m.current_coord:
                    self.player.play_loop(loop.wav_path, loop.duration_sec, fade_ms=200)
            except Exception as e:
                z.error = str(e)
                print(f"[FATAL][GEN] {z.spec.name} @ {coord}: {e}", flush=True)
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                return
            finally:
                z.generating = False
            # prune cache occasionally
            self._prune_cache()

    def request_generate(self, coord: Tuple[int,int]):
        self.gen_queue.put(coord)

    # ---------- preload start theme ----------
    def _maybe_preload(self, coord: Tuple[int,int]) -> bool:
        if coord != START_ZONE_COORD: return False
        path = START_THEME_WAV
        if not os.path.isfile(path): return False
        # duration
        try:
            import soundfile as sf
            with sf.SoundFile(path) as f:
                duration = len(f) / float(f.samplerate)
        except Exception:
            return False
        z = self.m.get_zone(coord)
        z.loop = GeneratedLoop(path, duration, z.spec.bpm, z.spec.key_mode, f"Preloaded theme: {z.spec.name}", {"backend":"preloaded"})
        if coord == self.m.current_coord:
            self.player.play_loop(z.loop.wav_path, z.loop.duration_sec, fade_ms=140)
        print(f"[Preload] {z.spec.name} @ {coord} → {os.path.basename(path)} ({duration:.2f}s)")
        return True

    # ---------- zone change ----------
    def on_zone_changed(self, oldc: Tuple[int,int], newc: Tuple[int,int]):
        self.player.stop(fade_ms=160)
        self.record_armed = False
        if self.recorder.is_recording(): self.recorder.stop()

        z = self.m.get_zone(newc)
        # try preload (only hits for start coord), else play/generate
        if z.loop:
            self.player.play_loop(z.loop.wav_path, z.loop.duration_sec, fade_ms=180)
        else:
            if not self._maybe_preload(newc):
                self.request_generate(newc)
        if AUTO_PREFETCH_NEIGHBORS:
            self.prefetch_ring(newc, PREFETCH_RADIUS)

    # ---------- neighbor prefetch + cache prune ----------
    def prefetch_ring(self, center: Tuple[int,int], radius: int):
        cx, cy = center
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                if abs(dx)+abs(dy) != 1:   # N/E/S/W only (Manhattan dist 1)
                    continue
                c = (cx+dx, cy+dy)
                z = self.m.get_zone(c)
                if (z.loop is None) and (not z.generating):
                    if not self._maybe_preload(c):
                        self.request_generate(c)

    def _prune_cache(self):
        # Limit number of stored zones to avoid unbounded growth
        if len(self.m.zones) <= MAX_ZONE_CACHE: return
        cx, cy = self.m.current_coord
        # rank by distance (farther first)
        items = sorted(self.m.zones.items(), key=lambda kv: abs(kv[0][0]-cx)+abs(kv[0][1]-cy), reverse=True)
        to_remove = len(self.m.zones) - MAX_ZONE_CACHE
        removed = 0
        for (coord, z) in items:
            if coord == self.m.current_coord: continue
            if z.generating: continue
            # keep neighbors
            if abs(coord[0]-cx)+abs(coord[1]-cy) <= 1: continue
            # drop loop to free memory; keep spec so it can be regenerated on return
            z.loop = None
            removed += 1
            if removed >= to_remove: break

    # ---------- loop boundary callback ----------
    def on_boundary_tick(self):
        if self.record_armed and not self.recorder.is_recording():
            z = self.m.get_zone(self.m.current_coord)
            if z.loop:
                self.recorder.start(z.loop.wav_path, z.loop.duration_sec,
                                    {"zone": z.spec.name, "bpm": z.spec.bpm, "key": z.spec.key_mode, "mood": z.spec.mood})
                self.record_armed = False
        elif self.recorder.is_recording() and self.auto_stop_at_end:
            self.recorder.stop()

# ============ Controller ============
class GameController:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Alien DJ – Infinite Prototype")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()

        self.model = WorldModel()
        self.view  = GameView(self.model)
        self.audio = AudioService(self.model)
        pygame.display.set_caption(f"Alien DJ – Backend: {type(self.audio.provider).__name__}")

        pygame.key.set_repeat(250, 30)

    def cycle_mood(self):
        order = ["calm", "energetic", "angry", "triumphant","melancholy","playful","brooding","gritty","glittery"]
        z = self.model.get_zone(self.model.current_coord)
        cur = z.spec.mood.lower()
        # find next in list if present, else start at calm
        nxt = order[(order.index(cur)+1) % len(order)] if cur in order else "calm"
        z.spec.mood = nxt

    def edit_mood_text(self):
        pygame.key.set_repeat(0)
        font = pygame.font.SysFont("consolas", 20)
        entered = ""
        done = False
        while not done:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: pygame.quit(); sys.exit(0)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_RETURN: done = True
                    elif e.key == pygame.K_ESCAPE: done = True
                    elif e.key == pygame.K_BACKSPACE: entered = entered[:-1]
                    else:
                        if e.unicode and 32 <= ord(e.unicode) < 127:
                            entered += e.unicode
            self.screen.fill((10,10,15))
            self.screen.blit(font.render("Enter mood/descriptor (Enter=OK, Esc=Cancel):", True, (240,240,240)), (40, SCREEN_H//2 - 30))
            self.screen.blit(font.render(entered, True, (180,255,180)), (40, SCREEN_H//2 + 10))
            pygame.display.flip(); self.clock.tick(30)
        if entered.strip():
            self.model.get_zone(self.model.current_coord).spec.mood = entered.strip()
        pygame.key.set_repeat(250, 30)

    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: running = False
                elif e.type == self.audio.player.boundary_event:
                    self.audio.on_boundary_tick()
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q): running = False
                    elif e.key == pygame.K_g:
                        # regenerate current tile fresh
                        c = self.model.current_coord
                        z = self.model.get_zone(c)
                        z.loop = None
                        self.audio.request_generate(c)
                    elif e.key == pygame.K_m:
                        self.cycle_mood()
                        self.audio.request_generate(self.model.current_coord)
                    elif e.key == pygame.K_e:
                        self.edit_mood_text()
                        self.audio.request_generate(self.model.current_coord)
                    elif e.key == pygame.K_r:
                        if self.audio.recorder.is_recording():
                            self.audio.recorder.stop(); self.audio.record_armed = False
                            print("[REC] stopped.")
                        else:
                            self.audio.record_armed = True
                            print("[REC] armed: start at next loop boundary.")
                    elif e.key == pygame.K_i:
                        print_inventory("./inventory")

            # movement (infinite; no clamps)
            keys = pygame.key.get_pressed()
            dx = dy = 0.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]: dx -= self.model.player.speed
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: dx += self.model.player.speed
            if keys[pygame.K_UP] or keys[pygame.K_w]: dy -= self.model.player.speed
            if keys[pygame.K_DOWN] or keys[pygame.K_s]: dy += self.model.player.speed
            if dx or dy:
                self.model.move_player(dx, dy)

            # render
            self.view.draw(self.screen, self.audio.record_armed, self.audio.recorder.is_recording())
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()

# ============ Utils ============
def print_inventory(inv_dir: str):
    try:
        files = sorted(f for f in os.listdir(inv_dir) if f.lower().endswith(".wav"))
    except FileNotFoundError:
        files = []
    if not files:
        print("[INV] (empty)"); return
    print("[INV] Recorded clips:")
    for f in files: print("  -", f)

# ============ Entrypoint ============
if __name__ == "__main__":
    try:
        GameController().run()
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
