#!/usr/bin/env python3
"""
Alien DJ Bootlegger – MVC Prototype
- Model (data): player, zones, NPCs, zone transitions
- View (render): draws world from model only
- Controller/Services: input, audio generation & playback, recording

Backends:
  AUDIO_BACKEND=stable_local  → Local Stable Audio Small (stable_audio_tools)
  AUDIO_BACKEND=dummy         → synthetic sine/noise loops (no deps)
"""

import os, sys, math, time, json, uuid, tempfile, wave, struct, threading, queue, hashlib
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import numpy as np
import pygame
import secrets


# --------------- Engine constants --------------
SCREEN_W, SCREEN_H = 1200, 700
GRID_COLS, GRID_ROWS = 4, 2
FPS = 60

DEFAULT_BARS = 8
DEFAULT_TIMESIG = (4, 4)
ENGINE_SR = 44100  # pygame mixer sample rate

# --------------- Zone presets ------------------
DEFAULT_ZONES = [
    ("Neon Bazaar",      120, "G Mixolydian",  "glittering synth arps, clacky percussion, funky market", "energetic"),
    ("Moss Docks",        96, "F Phrygian",    "dubby bass, foghorn pads, cavernous reverb",             "brooding"),
    ("Crystal Canyon",   140, "C Lydian",      "glassy mallets, airy choirs, shimmering echoes",         "triumphant"),
    ("Scrapyard Funk",   108, "E minor",       "breakbeat kit, clavinet riffs, grimey amp",              "gritty"),
    ("Fungal Grove",     100, "D major",       "hand percussion, kalimba, wooden clicks",                "playful"),
    ("Neon Rain",         92, "F# Aeolian",    "trip-hop, vinyl crackle, moody Rhodes",                  "melancholy"),
    ("Hover Bazaar",     132, "B Dorian",      "UK garage shuffle, vocal chops, subby bass",             "energetic"),
    ("Slime Parade",     120, "D minor",       "big-band brass, clav, wah guitar, parade snare",         "happy"),
]


# --- Startup prefs ---
START_ZONE_NAME = "Scrapyard Funk"
PRELOAD_WAVS = {
    # Use a raw string for the Windows path (note the r"...")
    "Scrapyard Funk": r"C:\Games\CosmicUnderground\inventory\rec_1756545018_Scrapyard Funk_d5ae11.wav"
}
AUTO_PREFETCH_NEIGHBORS = True

# --------------- Data types --------------------
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
    spec: ZoneSpec
    rect: pygame.Rect
    loop: Optional[GeneratedLoop] = None
    generating: bool = False
    error: Optional[str] = None

@dataclass
class NPC:
    zone_index: int
    pos: Tuple[int, int]

@dataclass
class Player:
    x: int
    y: int
    speed: int = 6

# --------------- Audio provider base ----------
class AudioProvider:
    def generate(self, zone: ZoneSpec) -> GeneratedLoop:
        raise NotImplementedError

    @staticmethod
    def duration_for(zone: ZoneSpec) -> float:
        beats = zone.bars * zone.timesig[0]
        return beats * (60.0 / zone.bpm)

# --------------- Dummy provider ---------------
class DummyProvider(AudioProvider):
    def __init__(self):
        self.tmpdir = tempfile.mkdtemp(prefix="alien_dj_dummy_")

    def generate(self, zone: ZoneSpec) -> GeneratedLoop:
        dur = self.duration_for(zone)
        path = os.path.join(self.tmpdir, f"{uuid.uuid4().hex}.wav")
        fL = 160.0 + (hash(zone.name) % 9) * 20 + (hash(zone.mood) % 11)
        fR = 230.0 + (hash(zone.scene) % 7) * 23
        amp = 0.22
        with wave.open(path, "wb") as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(ENGINE_SR)
            total = int(dur * ENGINE_SR)
            for n in range(total):
                t = n / ENGINE_SR
                beat = t * zone.bpm / 60.0
                gate = 0.3 + 0.7*(1.0 if (beat % 1.0) < 0.18 else 0.6)
                sL = amp * math.sin(2*math.pi*fL*t) * gate
                sR = amp * math.sin(2*math.pi*fR*t + 0.1) * gate
                wf.writeframes(struct.pack("<hh", int(max(-1,min(1,sL))*32767),
                                                  int(max(-1,min(1,sR))*32767)))
        prompt = zone.prompt_override or f"{zone.mood} {zone.scene}, loopable {zone.bars} bars, {zone.bpm} BPM, {zone.key_mode}"
        return GeneratedLoop(path, dur, zone.bpm, zone.key_mode, prompt, {"backend":"dummy"})

# --------------- Local Stable Audio provider --
class LocalStableAudioProvider(AudioProvider):
    """
    Local Stable Audio Small/Open using stable_audio_tools.
    Hard-fails if dependencies or model are unavailable.
    """
    def __init__(self):
        try:
            import torch  # local alias kept for device checks later
            from stable_audio_tools import get_pretrained_model
            # Import here so import errors surface early with a clear message
            self._torch = torch
            self._get_pretrained_model = get_pretrained_model
        except Exception as e:
            raise RuntimeError(
                "Stable Audio backend unavailable: could not import 'stable_audio_tools' or 'torch'.\n"
                "Fix inside your venv, e.g.:\n"
                "  pip install --no-deps stable-audio-tools==0.0.19\n"
                "  pip install numpy einops soundfile huggingface_hub pygame\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
            ) from e

        os.environ.setdefault("OMP_NUM_THREADS", "2")
        self._torch.set_num_threads(max(1, os.cpu_count() // 2))

        # Pick device
        self.device = (
            "cuda" if self._torch.cuda.is_available() else
            ("mps" if getattr(self._torch.backends, "mps", None)
                     and self._torch.backends.mps.is_available() else "cpu")
        )
        print(f"[StableLocal] Loading model… (device={self.device})", flush=True)

        try:
            self.model, self.cfg = self._get_pretrained_model("stabilityai/stable-audio-open-small")
        except Exception as e:
            raise RuntimeError(
                "Failed to load Stable Audio model 'stabilityai/stable-audio-open-small'. "
                "Check your internet, Hugging Face auth (if required), and package versions."
            ) from e

        self.model = self.model.to(self.device).eval()
        self.sr_model = int(self.cfg["sample_rate"])
        print(f"[StableLocal] Ready. sr={self.sr_model}", flush=True)

        self.tmpdir = tempfile.mkdtemp(prefix="alien_dj_local_")

    def _prompt(self, z: ZoneSpec) -> str:
        if z.prompt_override: return z.prompt_override
        return (f"{z.mood} alien scene: {z.scene}. "
                f"Groove-forward, clean downbeats, loopable {z.bars} bars. "
                f"{z.bpm} BPM, {z.key_mode}. Minimal silence at start/end.")

    def _seed(self, z: ZoneSpec) -> int:
        #h = hashlib.sha256(f"{z.name}|{z.mood}|{z.bpm}|{z.key_mode}|{z.scene}".encode()).hexdigest()
        
        # New: always generate a fresh seed so results differ across sessions and regenerations
        return secrets.randbits(31)
        #return int(h[:8], 16) & ((1<<31)-1)

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
        from einops import rearrange
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        seconds_total = AudioProvider.duration_for(zone)
        sample_size   = int(round(seconds_total * self.sr_model))
        seed = self._seed(zone)
        prompt = self._prompt(zone)

        t0 = time.time()
        try:
            with self._torch.inference_mode():
                audio = generate_diffusion_cond(
                    self.model,
                    steps=8, cfg_scale=1.0, sampler_type="pingpong",
                    conditioning=[{"prompt": prompt, "seconds_total": seconds_total}],
                    sample_size=sample_size, device=self.device, seed=seed
                )
        except Exception as e:
            raise RuntimeError(f"Stable Audio generation failed for zone '{zone.name}': {e}") from e

        audio = audio.squeeze(0).to(self._torch.float32).cpu()    # (2, T)
        audio = audio.transpose(0, 1).numpy().copy()              # (T, 2)
        peak = float(np.max(np.abs(audio))) or 1.0
        audio = (audio / peak) * 0.98
        audio = self._ensure_length(audio, self.sr_model, seconds_total)
        audio, out_sr = self._resample_linear(audio, self.sr_model, ENGINE_SR)

        out_path = os.path.join(self.tmpdir, f"{uuid.uuid4().hex}.wav")
        sf.write(out_path, audio, out_sr, subtype="PCM_16")
        print(f"[StableLocal] gen {zone.name} in {time.time()-t0:.2f}s → {os.path.basename(out_path)}", flush=True)
        return GeneratedLoop(out_path, seconds_total, zone.bpm, zone.key_mode, prompt,
                             {"backend":"stable_local","seed":seed,"sr":out_sr})

# --------------- Provider factory ------------
def make_provider() -> "AudioProvider":
    # No fallback. If Stable Audio can't load, the program should fail.
    return LocalStableAudioProvider()

# --------------- Audio playback --------------
class AudioPlayer:
    """Owns pygame.mixer.music, handles loop boundaries and fade in/out."""
    def __init__(self):
        pygame.mixer.init(frequency=ENGINE_SR, size=-16, channels=2, buffer=1024)
        self.boundary_event = pygame.USEREVENT + 1
        self.current: Optional[str] = None
        self.loop_ms = 0

    def play_loop(self, wav_path: str, duration_sec: float, fade_ms: int = 200):
        # schedule boundary tick
        self.loop_ms = max(1, int(duration_sec * 1000))
        pygame.time.set_timer(self.boundary_event, self.loop_ms)
        # play
        pygame.mixer.music.load(wav_path)
        pygame.mixer.music.play(loops=-1, fade_ms=fade_ms)
        self.current = wav_path

    def stop(self, fade_ms: int = 200):
        pygame.mixer.music.fadeout(fade_ms)
        pygame.time.set_timer(self.boundary_event, 0)
        self.current = None
        self.loop_ms = 0

# --------------- Recorder --------------------
class Recorder:
    """Copies loop file 'live' from boundary to inventory; manual or auto-stop at end-of-loop."""
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

# --------------- WORLD MODEL (data) ----------
class WorldModel:
    """Authoritative game state. Emits zone_changed events on transitions."""
    def set_start_zone_by_name(self, name: str) -> bool:
        for idx, z in enumerate(self.zones):
            if z.spec.name == name:
                self.current_zone = idx
                self.player.x = z.rect.centerx
                self.player.y = z.rect.centery
                return True
        return False
    def __init__(self, cols: int, rows: int, screen_w: int, screen_h: int):
        self.cols, self.rows = cols, rows
        self.sw, self.sh = screen_w, screen_h
        self.col_w = self.sw // self.cols
        self.row_h = self.sh // self.rows

        self.player = Player(self.sw // 2, self.sh // 2)
        self.zones: List[ZoneRuntime] = []
        for i, (name, bpm, key_mode, scene, mood) in enumerate(DEFAULT_ZONES):
            c = i % cols; r = i // cols
            rect = pygame.Rect(c*self.col_w, r*self.row_h, self.col_w, self.row_h)
            self.zones.append(ZoneRuntime(ZoneSpec(name, bpm, key_mode, scene, mood), rect))

        # NPCs at zone centers
        self.npcs: List[NPC] = []
        for idx, z in enumerate(self.zones):
            cx = z.rect.x + z.rect.w // 2
            cy = z.rect.y + z.rect.h // 2
            self.npcs.append(NPC(zone_index=idx, pos=(cx, cy)))

        self.current_zone = self.zone_index_at(self.player.x, self.player.y)
        self._zone_changed_listeners = []

    def zone_index_at(self, x: int, y: int) -> int:
        c = min(self.cols-1, max(0, x // self.col_w))
        r = min(self.rows-1, max(0, y // self.row_h))
        return int(r * self.cols + c)

    def add_zone_changed_listener(self, fn):
        self._zone_changed_listeners.append(fn)

    def move_player(self, dx: int, dy: int):
        px = max(0, min(self.sw-1, self.player.x + dx))
        py = max(0, min(self.sh-1, self.player.y + dy))
        if px == self.player.x and py == self.player.y:
            return
        self.player.x, self.player.y = px, py
        new_zone = self.zone_index_at(px, py)
        if new_zone != self.current_zone:
            old_zone = self.current_zone
            self.current_zone = new_zone
            for fn in self._zone_changed_listeners:
                fn(old_zone, new_zone)

# --------------- VIEW (render only) ----------
class GameView:
    def __init__(self, model: WorldModel):
        self.m = model
        self.font = pygame.font.SysFont("consolas", 18)

    def draw(self, screen: pygame.Surface, record_armed: bool, recorder_active: bool):
        screen.fill((14,10,18))
        for idx, z in enumerate(self.m.zones):
            rect = z.rect.inflate(-4, -4)
            active = (idx == self.m.current_zone)
            col = (40, 42, 70) if not active else (70, 88, 140)
            pygame.draw.rect(screen, col, rect, border_radius=12)
            # NPC
            cx, cy = self.m.npcs[idx].pos
            pygame.draw.circle(screen, (220,200,140), (cx, cy), 16)
            # labels
            screen.blit(self.font.render(z.spec.name, True, (240,240,240)), (rect.x+10, rect.y+10))
            sub = f"{z.spec.mood} | {z.spec.bpm} BPM | {z.spec.key_mode}"
            screen.blit(self.font.render(sub, True, (200,220,255)), (rect.x+10, rect.y+34))
            # status
            if z.generating: st = "generating…"
            elif z.error:    st = f"error: {z.error[:26]}…"
            elif not z.loop: st = "needs generate (G)"
            else:            st = os.path.basename(z.loop.wav_path)
            screen.blit(self.font.render(st, True, (180,180,180)), (rect.x+10, rect.y + rect.h - 28))
        # Player
        pygame.draw.circle(screen, (255,100,120), (self.m.player.x, self.m.player.y), 10)
        # HUD
        hud = [
            "WASD/Arrows move | G generate | M cycle mood | E edit mood | R arm/stop record | I inventory | Esc quit"
        ]
        if record_armed:   hud.append("REC ARMED: will start at next loop boundary.")
        if recorder_active:hud.append("RECORDING… R to stop manually (or auto-stop at loop end).")
        y = SCREEN_H - 22*len(hud) - 8
        for line in hud:
            screen.blit(self.font.render(line, True, (240,240,240)), (10, y))
            y += 22

# --------------- AUDIO SERVICE ---------------
class AudioService:
    """Bridges world events to audio: generation queue + loop playback + boundary for recording."""
    def _neighbor_indices(self, idx: int) -> list[int]:
        c = idx % self.m.cols
        r = idx // self.m.cols
        out = []
        if r > 0: out.append(idx - self.m.cols)                 # North
        if r < self.m.rows - 1: out.append(idx + self.m.cols)   # South
        if c > 0: out.append(idx - 1)                           # West
        if c < self.m.cols - 1: out.append(idx + 1)             # East
        return out
    
    def _prefetch_neighbors(self, idx: int):
        for j in self._neighbor_indices(idx):
            zj = self.m.zones[j]
            if (zj.loop is None) and (not zj.generating):
                # If you ever add more preloads per zone name, they’ll be picked up here
                if not self._maybe_preload_zone(j):
                    self.request_generate(j)
    
    def _maybe_preload_zone(self, idx: int) -> bool:
        """If this zone has a preloaded WAV path, attach it as the loop and play if active."""
        import soundfile as sf
        zr = self.m.zones[idx]
        path = PRELOAD_WAVS.get(zr.spec.name)
        if not path or not os.path.isfile(path):
            return False
        # compute duration from file
        with sf.SoundFile(path) as f:
            duration = len(f) / float(f.samplerate)
        zr.loop = GeneratedLoop(
            wav_path=path,
            duration_sec=duration,
            bpm=zr.spec.bpm,
            key_mode=zr.spec.key_mode,
            prompt=f"Preloaded theme: {zr.spec.name}",
            provider_meta={"backend": "preloaded"}
        )
        # If this zone is currently active, start it now
        if idx == self.m.current_zone:
            self.player.play_loop(zr.loop.wav_path, zr.loop.duration_sec, fade_ms=120)
        print(f"[Preload] {zr.spec.name} → {os.path.basename(path)} ({duration:.2f}s)")
        return True

    def __init__(self, model: WorldModel):
        self.m = model
        self.provider = make_provider()
        self.player = AudioPlayer()
        self.recorder = Recorder()
        self.auto_stop_at_end = True
        self.record_armed = False
        self.gen_queue: "queue.Queue[int]" = queue.Queue()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

        # hook zone change -> switch music immediately
        self.m.add_zone_changed_listener(self.on_zone_changed)

        # generate initial zone
        if not self._maybe_preload_zone(self.m.current_zone):
            self.request_generate(self.m.current_zone)
        
        # always prefetch neighbors on startup
        if AUTO_PREFETCH_NEIGHBORS:
            self._prefetch_neighbors(self.m.current_zone)

    def _worker(self):
        while True:
            idx = self.gen_queue.get()
            if idx is None: return
            zr = self.m.zones[idx]
            if zr.generating: continue
            zr.generating = True; zr.error = None
            try:
                loop = self.provider.generate(zr.spec)
                zr.loop = loop
                if idx == self.m.current_zone:
                    self.player.play_loop(loop.wav_path, loop.duration_sec, fade_ms=200)
            except Exception as e:
                zr.error = str(e)
                print(f"[FATAL][GEN] {zr.spec.name}: {e}", flush=True)
                # Hard-fail: stop the app (cleanly) so you don't hear any dummy sound.
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                return
            finally:
                zr.generating = False


    def request_generate(self, idx: int):
        self.gen_queue.put(idx)

    def on_zone_changed(self, old_idx: int, new_idx: int):
        """Immediate audible change on zone transition."""
        new_zone = self.m.zones[new_idx]
        # fade out current music immediately
        self.player.stop(fade_ms=160)
        # disarm/stop recording on zone switch (design choice)
        self.record_armed = False
        if self.recorder.is_recording():
            self.recorder.stop()
        # If we have a loop already, play it now; else request generation
        if new_zone.loop:
            self.player.play_loop(new_zone.loop.wav_path, new_zone.loop.duration_sec, fade_ms=180)
        else:
            self.request_generate(new_idx)  # will auto-play when ready
        if AUTO_PREFETCH_NEIGHBORS:
            self._prefetch_neighbors(new_idx)
        if new_zone.loop:
            self.player.play_loop(new_zone.loop.wav_path, new_zone.loop.duration_sec, fade_ms=180)
        else:
            if not self._maybe_preload_zone(new_idx):
                self.request_generate(new_idx)

    # ---- recording boundary handling (hook from main pygame loop) ----
    def on_boundary_tick(self):
        if self.record_armed and not self.recorder.is_recording():
            z = self.m.zones[self.m.current_zone]
            if z.loop:
                self.recorder.start(z.loop.wav_path, z.loop.duration_sec,
                                    {"zone": z.spec.name, "bpm": z.spec.bpm, "key": z.spec.key_mode, "mood": z.spec.mood})
                self.record_armed = False
        elif self.recorder.is_recording() and self.auto_stop_at_end:
            self.recorder.stop()

# --------------- CONTROLLER -------------------
class GameController:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Alien DJ Bootlegger – MVC Prototype")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()

        # MVC wiring
        self.model = WorldModel(GRID_COLS, GRID_ROWS, SCREEN_W, SCREEN_H)
        # Place the player in the requested start zone before audio spins up
        self.model.set_start_zone_by_name(START_ZONE_NAME)
        self.view  = GameView(self.model)
        self.audio = AudioService(self.model)
        
        pygame.display.set_caption(f"Alien DJ – Backend: {type(self.audio.provider).__name__}")

        # input repeat for text entry toggled temporarily in edit dialog
        pygame.key.set_repeat(250, 30)

    # ---- helpers for mood & prompts ----
    def cycle_mood(self):
        order = ["calm", "energetic", "angry", "triumphant"]
        z = self.model.zones[self.model.current_zone]
        cur = z.spec.mood.lower()
        z.spec.mood = order[(order.index(cur)+1) % len(order)] if cur in order else "calm"

    def edit_mood_text(self):
        pygame.key.set_repeat(0)
        font = pygame.font.SysFont("consolas", 22)
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
            self.screen.blit(font.render("Enter mood/descriptor (Enter=OK, Esc=Cancel):", True, (240,240,240)),
                             (40, SCREEN_H//2 - 30))
            self.screen.blit(font.render(entered, True, (180,255,180)), (40, SCREEN_H//2 + 10))
            pygame.display.flip(); self.clock.tick(30)
        if entered.strip():
            self.model.zones[self.model.current_zone].spec.mood = entered.strip()
        pygame.key.set_repeat(250, 30)

    # ---- run loop ----
    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: running = False
                elif e.type == self.audio.player.boundary_event:
                    self.audio.on_boundary_tick()
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q): running = False
                    elif e.key == pygame.K_g:  # regenerate current zone
                        self.audio.request_generate(self.model.current_zone)
                    elif e.key == pygame.K_m:
                        self.cycle_mood(); self.audio.request_generate(self.model.current_zone)
                    elif e.key == pygame.K_e:
                        self.edit_mood_text(); self.audio.request_generate(self.model.current_zone)
                    elif e.key == pygame.K_r:
                        if self.audio.recorder.is_recording():
                            self.audio.recorder.stop(); self.audio.record_armed = False
                            print("[REC] stopped.")
                        else:
                            self.audio.record_armed = True
                            print("[REC] armed: start at next loop boundary.")
                    elif e.key == pygame.K_i:
                        print_inventory("./inventory")

            # movement
            keys = pygame.key.get_pressed()
            dx = dy = 0
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

# --------------- Utils -----------------------
def print_inventory(inv_dir: str):
    try:
        files = sorted(f for f in os.listdir(inv_dir) if f.lower().endswith(".wav"))
    except FileNotFoundError:
        files = []
    if not files:
        print("[INV] (empty)"); return
    print("[INV] Recorded clips:")
    for f in files: print("  -", f)

# --------------- Entrypoint -------------------
if __name__ == "__main__":
    try:
        GameController().run()
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
