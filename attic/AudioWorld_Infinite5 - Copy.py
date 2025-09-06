#!/usr/bin/env python3
# Infinite-scroll Alien DJ – Priority generation + rich procedural world.

import os, sys, math, time, uuid, tempfile, wave, struct, threading, queue, hashlib, secrets, random, itertools, heapq
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import numpy as np
import pygame
import itertools, heapq, threading, time

global FULLSCREEN
global SCREEN_W, SCREEN_H
# --- Display config ---
DEFAULT_FULLSCREEN = True  # set default; can be toggled at runtime with F11 / Alt+Enter

# ================= Window/engine =================

SCREEN_W, SCREEN_H = 1200, 700
FPS = 60
TILE_W, TILE_H = 300, 350
ENGINE_SR = 44100

# ================= Startup/config =================
START_ZONE_COORD = (0, 0)
START_ZONE_NAME  = "Scrapyard Funk"
START_THEME_WAV  = r"C:\Games\CosmicUnderground\inventory\rec_1756545018_Scrapyard Funk_d5ae11.wav"

PREFETCH_RADIUS = 1             # N/E/S/W
MAX_ZONE_CACHE  = 64            # prune old zones’ audio
SESSION_SEED    = secrets.randbits(32)  # fresh world feel each run

DEFAULT_BARS    = 8
DEFAULT_TIMESIG = (4, 4)

# ================= Datatypes =================
@dataclass
class ZoneSpec:
    name: str
    bpm: int
    key_mode: str
    scene: str
    mood: str
    bars: int = DEFAULT_BARS
    timesig: Tuple[int,int] = DEFAULT_TIMESIG
    prompt_override: Optional[str] = None
    # Procedural extras:
    biome: str = "Unknown"
    species: str = "Unknown"
    descriptors: List[str] = field(default_factory=list)   # adjectives/texture words
    instruments: List[str] = field(default_factory=list)   # instrument hints
    tags: List[str] = field(default_factory=list)          # contrasting/production tags

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
    coord: Tuple[int,int]
    spec: ZoneSpec
    loop: Optional[GeneratedLoop] = None
    generating: bool = False
    error: Optional[str] = None
    last_seed: Optional[int] = None

@dataclass
class Player:
    x: float
    y: float
    speed: float = 6.0

# ================= Audio provider base =================
class AudioProvider:
    def generate(self, zone: ZoneSpec) -> GeneratedLoop: raise NotImplementedError
    @staticmethod
    def duration_for(zone: ZoneSpec) -> float:
        beats = zone.bars * zone.timesig[0]
        return beats * (60.0 / zone.bpm)


       
CONTRAST_PAIRS = [
    ("fast","slow"), ("funky","classic"), ("brassy","woodwinds"),
    ("loud","soft"), ("layered","simple"), ("retro","futuristic")
]

SPECIES_POOLS = {
    "Chillaxians": dict(
        bpm=(84, 102),
        instruments=["warm pads","ice chimes","soft bass","brush kit","chorus keys"],
        textures=["calm","retro","soft","crystalline","airy"]
    ),
    "Glorpals": dict(
        bpm=(96, 122),
        instruments=["squelch synth","gel bells","drip fx","rubber bass","wet claps"],
        textures=["wet","slurpy","layered","sproingy"]
    ),
    "Bzaris": dict(
        bpm=(120, 140),
        instruments=["bitcrush blips","granular stutters","noisy hats","FM lead","resonant zap"],
        textures=["fast","glitchy","futuristic","electric"]
    ),
    "Shagdeliacs": dict(
        bpm=(96, 120),
        instruments=["wah guitar","clavinet","horn section","rimshot kit","upright-ish synth bass"],
        textures=["funky","brassy","jazzy","retro","hairy","furry"]
    ),
    "Rockheads": dict(
        bpm=(100, 126),
        instruments=["metal gongs","anvil hits","deep toms","clang perc","industrial pad"],
        textures=["mechanical","metallic","loud","magnetic"]
    ),
}

EXTRA_INSTR = ["kalimba","woodblocks","808 subs","tape echo stabs","analog brass","vocoder chops","hand drums","dub chords","arpeggiator"]
EXTRA_TEXTS = ["crystalline","laser","electric","mechanical","nuclear","magnetic","chemical","fuzzy","hair-clad","hirsute"]

MODES  = ["major","minor","Dorian","Phrygian","Mixolydian","Lydian","Aeolian"]
TONICS = ["A","B","C","D","E","F","G","A♭","B♭","C♯","D♯","F♯","G♯"]

def _prompt(self, z: ZoneSpec) -> str:
    if z.prompt_override:
        return z.prompt_override
    from random import Random
    return build_prompt_from_spec(z, bars=z.bars, rng=Random(), intensity=0.55)

def build_prompt_from_spec(spec, bars=8, rng=None, intensity=0.5):
    """
    spec: ZoneSpec with .species, .biome, .mood, .bpm, .key_mode (optional), .descriptors/instruments/tags (optional)
    intensity: 0..1 controls density and aggressiveness of textures
    """
    rng = rng or random.Random()

    # BPM & key
    bpm = getattr(spec, "bpm", None)
    species_opts = SPECIES_POOLS.get(getattr(spec, "species", ""), None)
    if species_opts and bpm is None:
        lo, hi = species_opts["bpm"]
        bpm = rng.randrange(lo, hi+1, 2)
    # jitter a touch so neighbors differ
    bpm = int(max(84, min(140, (bpm or 120) + rng.choice([-4,-2,0,2,4]))))

    key_mode = getattr(spec, "key_mode", None)
    if not key_mode:
        key_mode = f"{rng.choice(TONICS)} {rng.choice(MODES)}"

    # Instruments
    pool_instruments = (species_opts or {}).get("instruments", [])
    base_instr = pool_instruments[:]
    if getattr(spec, "instruments", None):
        base_instr += spec.instruments
    # sample 2..4 instruments
    k_instr = 2 + int( (2 * intensity) )
    instr = rng.sample(list(set(base_instr + EXTRA_INSTR)), k=min(k_instr, max(2, len(set(base_instr + EXTRA_INSTR)))))

    # Textures (descriptors)
    pool_textures = (species_opts or {}).get("textures", [])
    base_tex = pool_textures[:]
    if getattr(spec, "descriptors", None):
        base_tex += spec.descriptors
    k_tex = 2 + int( (1 * intensity) )
    tex = rng.sample(list(set(base_tex + EXTRA_TEXTS)), k=min(k_tex, max(2, len(set(base_tex + EXTRA_TEXTS)))))

    # Contrast tag: pick ONE side only
    side = rng.choice(CONTREATE := CONTRAST_PAIRS)
    tags = [rng.choice(side)]
    # optional second stylistic tag from textures
    tags.append(rng.choice(tex))

    # Mood and biome lines
    mood = getattr(spec, "mood", "energetic")
    biome = getattr(spec, "biome", "unknown biome")
    species = getattr(spec, "species", "unknown species")
    name = getattr(spec, "name", "Unknown Zone")

    # Compose text
    desc = ", ".join(tex[:3])
    inst = ", ".join(instr[:4])
    tagline = ", ".join(tags[:2])

    body = (
        f"{mood} vibes in the {biome}, home of the {species}. "
        f"{desc}. Instruments: {inst}. "
        f"Style tags: {tagline}. "
        f"Loopable {bars} bars, {bpm} BPM, {key_mode}. "
        f"Clean downbeat; seamless loop; minimal silence at edges; tight low end."
    )
    return body

# ================= Stable Audio provider =================
class LocalStableAudioProvider(AudioProvider):
    def __init__(self):
        try:
            import torch
            from stable_audio_tools import get_pretrained_model
            self._torch = torch
            self._get_pretrained_model = get_pretrained_model
        except Exception as e:
            raise RuntimeError(
                "Stable Audio backend unavailable. Inside your venv:\n"
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
        seed = secrets.randbits(31)  # fresh each time
        prompt = build_prompt_from_spec(zone, bars=zone.bars, rng=random.Random(), intensity=0.55)
        print(f"[StableLocal] seed={seed} | {zone.species} @ {zone.biome} | '{zone.name}'", flush=True)

        t0 = time.time()
        with self._torch.inference_mode():
            audio = generate_diffusion_cond(
                self.model,
                steps=8, cfg_scale=1.0, sampler_type="pingpong",
                conditioning=[{"prompt": prompt, "seconds_total": seconds_total}],
                sample_size=sample_size, device=self.device, seed=seed
            )
        # (B, D, T) -> (T, D)
        audio = audio.squeeze(0).to(self._torch.float32).cpu()
        audio = audio.transpose(0, 1).numpy().copy()
        peak = float(np.max(np.abs(audio))) or 1.0
        audio = (audio / peak) * 0.98
        audio = self._ensure_length(audio, self.sr_model, seconds_total)
        audio, out_sr = self._resample_linear(audio, self.sr_model, ENGINE_SR)

        out_path = os.path.join(self.tmpdir, f"{uuid.uuid4().hex}.wav")
        sf.write(out_path, audio, out_sr, subtype="PCM_16")
        print(f"[StableLocal] gen {zone.name} in {time.time()-t0:.2f}s → {os.path.basename(out_path)}", flush=True)
        return GeneratedLoop(out_path, seconds_total, zone.bpm, zone.key_mode, prompt,
                             {"backend":"stable_local","seed":seed,"sr":out_sr})

# ================= Playback/record =================
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

# ================= Procedural engine =================
class ProcGen:
    """Deterministic per-session/per-tile param builder; inspired by your notes."""
    ALIENS = {
        "Chillaxians":  dict(biomes=["Polar Ice", "Crystal Canyons"], bpm=(80,100), mood=["calm","triumphant"], tags=["slow","retro","soft"], inst=["warm pads","ice chimes","soft bass"]),
        "Glorpals":     dict(biomes=["Slime Caves","Groove Pits"],   bpm=(92,118), mood=["playful","melancholy"], tags=["wet","slurpy","layered"], inst=["gel bells","drip fx","squelch synth"]),
        "Bzaris":       dict(biomes=["Centerspire","Crystal Canyons"], bpm=(120,142), mood=["energetic","angry"], tags=["fast","glitchy","futuristic"], inst=["bitcrush blips","granular stutters","noisy hats"]),
        "Shagdeliacs":  dict(biomes=["Groove Pits","Centerspire"],   bpm=(96,120), mood=["funky","playful"], tags=["brassy","jazzy","retro"], inst=["wah guitar","clavinet","horn section"]),
        "Rockheads":    dict(biomes=["Groove Pits","Slime Caves"],   bpm=(100,126), mood=["brooding","triumphant"], tags=["mechanical","metallic","loud"], inst=["metal gongs","anvil hits","deep toms"]),
    }
    BIOMES = ["Crystal Canyons","Slime Caves","Centerspire","Groove Pits","Polar Ice","Mag-Lev Yards","Electro Marsh"]
    CONTRAST = [
        ("fast","slow"), ("funky","classic"), ("brassy","woodwinds"),
        ("loud","soft"), ("layered","simple"), ("retro","futuristic")
    ]
    DESCRIPTORS = ["wet","sproingy","furry","hairy","hirsute","hair-clad","fuzzy","slurpy","crystalline","laser","electric","mechanical","nuclear","magnetic","chemical"]
    INSTR_EXTRA  = ["kalimba","woodblocks","808 subs","sub-bass","tape echo","space choir","gamelan bells","analog brass","vocoder chops","hand drums","dub chords"]

    MODES  = ["major","minor","Dorian","Phrygian","Mixolydian","Lydian","Aeolian"]
    TONICS = ["A","B","C","D","E","F","G","A♭","B♭","C♯","D♯","F♯","G♯"]

    def __init__(self, session_seed: int):
        self.session_seed = session_seed

    def _rng(self, zx: int, zy: int) -> random.Random:
        return random.Random((self.session_seed << 1) ^ (zx * 73856093) ^ (zy * 19349663))

    def build_spec(self, coord: Tuple[int,int]) -> ZoneSpec:
        zx, zy = coord
        rng = self._rng(zx, zy)

        # biome: banded by y with noise twist
        biome = rng.choice(self.BIOMES)
        # species: weight by biome affinity
        species = rng.choice(list(self.ALIENS.keys()))
        # prefer alien whose biomes include the chosen biome
        for _ in range(2):
            pick = rng.choice(list(self.ALIENS.keys()))
            if biome in self.ALIENS[pick]["biomes"]:
                species = pick; break

        a = self.ALIENS[species]
        bpm = rng.randrange(a["bpm"][0], a["bpm"][1]+1, 2)
        mood = rng.choice(a["mood"])
        tags = list(set(a["tags"] + [rng.choice(self.DESCRIPTORS) for _ in range(2)]))
        # Add one side of a contrast pair, randomly choosing which side
        tag_pair = rng.choice(self.CONTRAST)
        tags.append(rng.choice(tag_pair))

        # instruments
        instruments = list(set(a["inst"] + rng.sample(self.INSTR_EXTRA, k=min(2, len(self.INSTR_EXTRA)))))

        # name: themed
        if coord == START_ZONE_COORD:
            name = START_ZONE_NAME
        else:
            A = ["Neon","Crystal","Slime","Chrome","Velvet","Magnetic","Shatter","Rusty","Electric","Gleaming","Polar","Echo"]
            L = ["Bazaar","Canyons","Caves","Centerspire","Grove","Pits","Foundry","Sprawl","Galleries","Arcade","Causeway","Yards"]
            T = ["Funk","Pulse","Flux","Jitter","Parade","Breaks","Mallets","Reverb","Drift","Chromatics","Riff"]
            base = f"{rng.choice(A)} {rng.choice(L)}"
            if rng.random() < 0.4: base = f"{base} {rng.choice(T)}"
            name = base

        key_mode = f"{rng.choice(self.TONICS)} {rng.choice(self.MODES)}"

        # scene sentence (kept short; descriptors will add color)
        scene = rng.choice([
            "glittering arps and clacky percussion",
            "dubby chords and foghorn pads",
            "glassy mallets and airy choirs",
            "breakbeat kit and gnarly clavs",
            "hand percussion and kalimba",
            "trip-hop haze with vinyl crackle",
            "UKG shuffle with vocal chops",
            "parade brass and wah guitar",
            "holo-arcade bleeps and neon hum",
        ])

        # descriptor list (texture words)
        descriptors = rng.sample(self.DESCRIPTORS, k=3)

        return ZoneSpec(
            name=name, bpm=bpm, key_mode=key_mode, scene=scene, mood=mood,
            biome=biome, species=species, descriptors=descriptors,
            instruments=instruments, tags=tags
        )

# ================= World Model (infinite) =================
class WorldModel:
    def __init__(self):
        self.player = Player(0, 0, speed=6.0)
        self.zones: Dict[Tuple[int,int], ZoneRuntime] = {}
        self.current_coord = START_ZONE_COORD
        self._listeners = []
        self.pg = ProcGen(SESSION_SEED)
        self.set_start_coord(START_ZONE_COORD)

    def get_zone(self, coord: Tuple[int,int]) -> ZoneRuntime:
        if coord not in self.zones:
            spec = self.pg.build_spec(coord)
            self.zones[coord] = ZoneRuntime(coord=coord, spec=spec)
        return self.zones[coord]

    def set_start_coord(self, coord: Tuple[int,int]):
        self.current_coord = coord
        self.player.x = coord[0]*TILE_W + TILE_W/2
        self.player.y = coord[1]*TILE_H + TILE_H/2

    def add_zone_changed_listener(self, fn): self._listeners.append(fn)

    def move_player(self, dx: float, dy: float):
        self.player.x += dx; self.player.y += dy
        zx = math.floor(self.player.x / TILE_W)
        zy = math.floor(self.player.y / TILE_H)
        newc = (zx, zy)
        if newc != self.current_coord:
            old = self.current_coord
            self.current_coord = newc
            for fn in self._listeners: fn(old, newc)

# ================= View =================
class GameView:
    def __init__(self, model: WorldModel):
        self.m = model
        self.font = pygame.font.SysFont("consolas", 18)

    def _draw_glow(self, surf: pygame.Surface, rect: pygame.Rect, pulse_t: float):
        glow_color = (170 + int(60*(0.5+0.5*math.sin(pulse_t*2.0))), 60, 220)
        glow = pygame.Surface((rect.w+16, rect.h+16), pygame.SRCALPHA)
        base = pygame.Rect(8, 8, rect.w, rect.h)
        for i, alpha in enumerate((90, 60, 30)):
            pygame.draw.rect(glow, (*glow_color, alpha), base.inflate(8+i*6, 8+i*6), width=3, border_radius=14)
        surf.blit(glow, (rect.x-8, rect.y-8))

    def draw(self, screen: pygame.Surface, record_armed: bool, recorder_active: bool,
         show_prompt: bool = False, prompt_text: str = ""):
        screen.fill((14,10,18))
        pulse_t = pygame.time.get_ticks()/1000.0
        cam_x = self.m.player.x - SCREEN_W/2
        cam_y = self.m.player.y - SCREEN_H/2
        min_zx = math.floor(cam_x / TILE_W) - 1
        max_zx = math.floor((cam_x + SCREEN_W) / TILE_W) + 1
        min_zy = math.floor(cam_y / TILE_H) - 1
        max_zy = math.floor((cam_y + SCREEN_H) / TILE_H) + 1

        for zy in range(min_zy, max_zy+1):
            for zx in range(min_zx, max_zx+1):
                coord = (zx, zy)
                z = self.m.get_zone(coord)
                sx = int(zx*TILE_W - cam_x); sy = int(zy*TILE_H - cam_y)
                rect = pygame.Rect(sx+2, sy+2, TILE_W-4, TILE_H-4)
                active = (coord == self.m.current_coord)
                base_col = (40, 42, 70) if not active else (70, 88, 140)
                pygame.draw.rect(screen, base_col, rect, border_radius=12)

                if z.loop is not None and z.error is None:
                    self._draw_glow(screen, rect, pulse_t)

                pygame.draw.circle(screen, (220,200,140), (rect.centerx, rect.centery), 14)

                name = self.font.render(z.spec.name, True, (240,240,240))
                sub  = self.font.render(f"{z.spec.species} | {z.spec.biome} | {z.spec.mood} | {z.spec.bpm} BPM | {z.spec.key_mode}", True, (200,220,255))
                screen.blit(name, (rect.x+10, rect.y+10))
                screen.blit(sub,  (rect.x+10, rect.y+34))

                if z.generating: st = "generating…"
                elif z.error:    st = f"error: {z.error[:26]}…"
                elif not z.loop: st = "not generated"
                else:            st = os.path.basename(z.loop.wav_path)
                screen.blit(self.font.render(st, True, (180,180,180)), (rect.x+10, rect.bottom - 28))

        pygame.draw.circle(screen, (255,100,120), (int(self.m.player.x - cam_x), int(self.m.player.y - cam_y)), 10)

        hud = ["Move: WASD/Arrows | G regenerate | M cycle mood | E edit mood | R arm/stop record | I inventory | Esc quit"]
        if record_armed:    hud.append("REC ARMED: will start at next loop boundary.")
        if recorder_active: hud.append("RECORDING… R to stop manually (or auto-stop at loop end).")
        y = SCREEN_H - 22*len(hud) - 8
        for line in hud:
            screen.blit(self.font.render(line, True, (240,240,240)), (10, y)); y += 22
        # --- Prompt overlay ---
        if show_prompt and prompt_text:
            # Layout
            max_w = int(SCREEN_W * 0.8)
            pad   = 14
            title_font = pygame.font.SysFont("consolas", 20, bold=True)
            body_font  = pygame.font.SysFont("consolas", 18)
            title = "Audio Prompt"
            lines = self._wrap_text(prompt_text, body_font, max_w)
        
            # Measure
            title_w, title_h = title_font.size(title)
            body_h = sum(body_font.size(line)[1] + 4 for line in lines)
            box_w = min(max_w, max(title_w, max((body_font.size(l)[0] for l in lines), default=0))) + pad*2
            box_h = title_h + 10 + body_h + pad*2
        
            # Position (bottom-left)
            box_x = 10
            box_y = SCREEN_H - box_h - 10
        
            # Background (translucent) + border
            surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            pygame.draw.rect(surf, (20, 16, 28, 220), (0, 0, box_w, box_h), border_radius=10)
            pygame.draw.rect(surf, (180, 120, 255, 255), (0, 0, box_w, box_h), width=2, border_radius=10)
            screen.blit(surf, (box_x, box_y))
        
            # Text
            screen.blit(title_font.render(title, True, (240, 230, 255)), (box_x + pad, box_y + pad))
            y = box_y + pad + title_h + 6
            for line in lines:
                screen.blit(body_font.render(line, True, (230, 230, 240)), (box_x + pad, y))
                y += body_font.size(line)[1] + 4
                
    def _wrap_text(self, text: str, font: pygame.font.Font, max_width: int) -> list[str]:
        words = text.split()
        lines, cur = [], ""
        for w in words:
            test = w if not cur else cur + " " + w
            if font.size(test)[0] <= max_width:
                cur = test
            else:
                if cur: lines.append(cur)
                cur = w
        if cur: lines.append(cur)
        return lines

# ================= Audio Service (priority queue) =================
class AudioService:
    """
    Priority rules (recomputed on every zone change):
      tier 0 = current zone (must be next)
      tier 1 = neighbors (N/E/S/W)
      tier 2 = everything else (previously queued)
    Within a tier, nearer distance wins; ties use FIFO.
    """
    def __init__(self, model: WorldModel):
        
        self.m = model
        self.provider = LocalStableAudioProvider()  # hard requirement
        self.player = AudioPlayer()
        self.recorder = Recorder()
        self.auto_stop_at_end = True
        self.record_armed = False

        # --- scheduling structures ---
        self._heap = []                     # heap of (priority_tuple, seq, coord, token)
        self._counter = itertools.count()   # monotonic sequence, also used as token
        self._pending: Dict[Tuple[int,int], int] = {}  # coord -> latest token (for stale-skip)
        self._lock = threading.Lock()

        # worker
        self._worker_stop = False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.m.add_zone_changed_listener(self.on_zone_changed)

        # Start: preload theme (if any) or generate current, then neighbors
        if not self._maybe_preload(self.m.current_coord):
            self.request_generate(self.m.current_coord, priority=0, force=True)
        self._reprioritize_all()  # ensure heap ordering is correct right away
        self.prefetch_neighbors(self.m.current_coord, priority=1)

    # ------------- priority computation -------------
    def _priority_for(self, coord: Tuple[int,int], center: Tuple[int,int]) -> Tuple[int,int]:
        cx, cy = center
        x, y = coord
        dist = abs(x - cx) + abs(y - cy)  # Manhattan
        if dist == 0:        tier = 0
        elif dist == 1:      tier = 1
        else:                tier = 2
        return (tier, dist)

    # ------------- queue API -------------
    def request_generate(self, coord: Tuple[int,int], priority: int = None, force: bool = False):
        """
        Enqueue (or re-enqueue) a task.
        priority: None = compute from current position;
                  0 = force 'current zone' tier;
                  1 = force 'neighbor' tier;
                  others fall back to computed priority.
        """
        z = self.m.get_zone(coord)
        if not force and (z.generating or z.loop is not None):
            return

        # compute priority tuple
        if priority == 0:
            prio = (0, 0)
        elif priority == 1:
            prio = (1, 0)
        else:
            prio = self._priority_for(coord, self.m.current_coord)

        with self._lock:
            token = next(self._counter)
            self._pending[coord] = token
            heapq.heappush(self._heap, (prio, token, coord, token))

    def _pop_task(self):
        import heapq, time
        with self._lock:
            while self._heap:
                prio, seq, coord, token = heapq.heappop(self._heap)
                # skip stale entries (re-enqueued with a newer token)
                if self._pending.get(coord) != token:
                    continue
                return (prio, seq, coord, token)
        return None

    def _worker_loop(self):
        import time, pygame
        while not self._worker_stop:
            item = self._pop_task()
            if item is None:
                time.sleep(0.01)
                continue
            _, _, coord, token = item
            z = self.m.get_zone(coord)
            # safety: if it just got generated or is generating, skip
            if z.generating or z.loop is not None:
                continue

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
            self._prune_cache()

    # ------------- reprioritize on zone change -------------
    def _reprioritize_all(self):
        """
        Completely rebuild the heap based on the *current* position.
        Keeps only tasks that are not generated and not generating.
        Ensures current (tier 0) then neighbors (tier 1) then backlog (tier 2).
        """
        import heapq
        with self._lock:
            coords = list(self._pending.keys())
            self._heap.clear()
            self._pending.clear()
        # re-enqueue with fresh tokens and new priorities
        # First: current tile
        c = self.m.current_coord
        zc = self.m.get_zone(c)
        if zc.loop is None and not zc.generating:
            self.request_generate(c, priority=0, force=True)
        # Then: neighbors
        for n in self.neighbor_coords(c):
            zn = self.m.get_zone(n)
            if zn.loop is None and not zn.generating:
                self.request_generate(n, priority=1, force=False)
        # Finally: backlog
        for coord in coords:
            if coord == c or coord in self.neighbor_coords(c):
                continue
            z = self.m.get_zone(coord)
            if z.loop is None and not z.generating:
                self.request_generate(coord, priority=None, force=False)

    # ------------- preload theme (start tile only) -------------
    def _maybe_preload(self, coord: Tuple[int,int]) -> bool:
        if coord != START_ZONE_COORD: return False
        path = START_THEME_WAV
        if not os.path.isfile(path): return False
        try:
            import soundfile as sf
            with sf.SoundFile(path) as f:
                duration = len(f) / float(f.samplerate)
        except Exception:
            return False
        z = self.m.get_zone(coord)
        z.loop = GeneratedLoop(path, duration, z.spec.bpm, z.spec.key_mode,
                               f"Preloaded theme: {z.spec.name}", {"backend":"preloaded"})
        if coord == self.m.current_coord:
            self.player.play_loop(z.loop.wav_path, z.loop.duration_sec, fade_ms=140)
        print(f"[Preload] {z.spec.name} @ {coord} → {os.path.basename(path)} ({duration:.2f}s)")
        return True

    # ------------- zone change -------------
    def on_zone_changed(self, oldc: Tuple[int,int], newc: Tuple[int,int]):
        # stop current playback; disarm recording
        self.player.stop(fade_ms=160)
        self.record_armed = False
        if self.recorder.is_recording(): self.recorder.stop()

        # ensure new current is queued first (or play immediately if loop ready)
        z = self.m.get_zone(newc)
        if z.loop:
            self.player.play_loop(z.loop.wav_path, z.loop.duration_sec, fade_ms=180)
        else:
            if not self._maybe_preload(newc):
                self.request_generate(newc, priority=0, force=True)

        # rebuild priorities so neighbors of *newc* come next, backlog after
        self._reprioritize_all()

    # ------------- neighbors / prefetch -------------
    def neighbor_coords(self, center: Tuple[int,int]) -> List[Tuple[int,int]]:
        cx, cy = center
        return [(cx, cy-1), (cx+1, cy), (cx, cy+1), (cx-1, cy)]

    def prefetch_neighbors(self, center: Tuple[int,int], priority: int = 1):
        for c in self.neighbor_coords(center):
            z = self.m.get_zone(c)
            if z.loop is None and not z.generating:
                self.request_generate(c, priority=priority, force=False)

    # ------------- cache pruning -------------
    def _prune_cache(self):
        if len(self.m.zones) <= MAX_ZONE_CACHE: return
        cx, cy = self.m.current_coord
        items = sorted(self.m.zones.items(),
                       key=lambda kv: abs(kv[0][0]-cx)+abs(kv[0][1]-cy),
                       reverse=True)
        to_remove = len(self.m.zones) - MAX_ZONE_CACHE
        removed = 0
        for coord, z in items:
            if coord == self.m.current_coord: continue
            if z.generating: continue
            if abs(coord[0]-cx)+abs(coord[1]-cy) <= 1: continue
            z.loop = None
            removed += 1
            if removed >= to_remove: break

    # ------------- loop boundary for recording -------------
    def on_boundary_tick(self):
        if self.record_armed and not self.recorder.is_recording():
            z = self.m.get_zone(self.m.current_coord)
            if z.loop:
                self.recorder.start(z.loop.wav_path, z.loop.duration_sec,
                                    {"zone": z.spec.name, "bpm": z.spec.bpm, "key": z.spec.key_mode, "mood": z.spec.mood})
                self.record_armed = False
        elif self.recorder.is_recording() and self.auto_stop_at_end:
            self.recorder.stop()


# ================= Controller =================
class GameController:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Alien DJ – Infinite (Priority)")
        self.fullscreen = DEFAULT_FULLSCREEN

        self.screen = self._apply_display_mode()  # sets SCREEN_W/H
        self.clock  = pygame.time.Clock()

        self.model = WorldModel()
        self.show_prompt = False
        self.prompt_text = ""
        # Auto-hide prompt on zone change
        self.model.add_zone_changed_listener(lambda oldc, newc: setattr(self, "show_prompt", False))

        self.view  = GameView(self.model)
        self.audio = AudioService(self.model)
        pygame.display.set_caption(f"Alien DJ – Backend: {type(self.audio.provider).__name__}")

        pygame.key.set_repeat(250, 30)

    def _apply_display_mode(self):
        """Fullscreen uses real desktop res (no SCALED) so you see more tiles."""
        global SCREEN_W, SCREEN_H
        if self.fullscreen:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            screen = pygame.display.set_mode((1200, 700), pygame.RESIZABLE)
        SCREEN_W, SCREEN_H = screen.get_size()   # renderer uses these -> bigger viewport
        return screen

    def cycle_mood(self):
        order = ["calm","energetic","angry","triumphant","melancholy","playful","brooding","gritty","glittery","funky"]
        z = self.model.get_zone(self.model.current_coord)
        cur = z.spec.mood.lower()
        z.spec.mood = order[(order.index(cur)+1) % len(order)] if cur in order else "calm"

    def edit_mood_text(self):
        pygame.key.set_repeat(0)
        font = pygame.font.SysFont("consolas", 20)
        entered = ""; done = False
        while not done:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: pygame.quit(); sys.exit(0)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_RETURN: done = True
                    elif e.key == pygame.K_ESCAPE: done = True
                    elif e.key == pygame.K_BACKSPACE: entered = entered[:-1]
                    else:
                        if e.unicode and 32 <= ord(e.unicode) < 127: entered += e.unicode
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
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.VIDEORESIZE and not self.fullscreen:
                    global SCREEN_W, SCREEN_H
                    SCREEN_W, SCREEN_H = e.w, e.h
                    self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.RESIZABLE)
                elif e.type == self.audio.player.boundary_event:
                    self.audio.on_boundary_tick()
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif e.key == pygame.K_p:
                        # toggle prompt overlay
                        if self.show_prompt:
                            self.show_prompt = False
                        else:
                            z = self.model.get_zone(self.model.current_coord)
                            self.prompt_text = z.loop.prompt if (z.loop and z.loop.prompt) else "(No audio prompt yet — this zone’s loop hasn’t been generated.)"
                            self.show_prompt = True
                    elif e.key == pygame.K_F11 or (e.key == pygame.K_RETURN and (e.mod & pygame.KMOD_ALT)):
                        self.fullscreen = not self.fullscreen
                        self.screen = self._apply_display_mode()
                        pygame.display.set_caption(f"Alien DJ – Backend: {type(self.audio.provider).__name__}")
                        pygame.display.set_caption(f"Alien DJ – Backend: {type(self.audio.provider).__name__}")
                    elif e.key == pygame.K_g:
                        c = self.model.current_coord
                        z = self.model.get_zone(c); z.loop = None
                        self.audio.request_generate(c, priority=0, force=True)
                    elif e.key == pygame.K_m:
                        self.cycle_mood()
                        self.audio.request_generate(self.model.current_coord, priority=0, force=True)
                    elif e.key == pygame.K_e:
                        self.edit_mood_text()
                        self.audio.request_generate(self.model.current_coord, priority=0, force=True)
                    elif e.key == pygame.K_r:
                        if self.audio.recorder.is_recording():
                            self.audio.recorder.stop(); self.audio.record_armed = False
                            print("[REC] stopped.")
                        else:
                            self.audio.record_armed = True
                            print("[REC] armed: start at next loop boundary.")
                    elif e.key == pygame.K_i:
                        print_inventory("./inventory")

            keys = pygame.key.get_pressed()
            dx = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
            dy = (keys[pygame.K_DOWN]  or keys[pygame.K_s]) - (keys[pygame.K_UP]   or keys[pygame.K_w])
            if dx or dy:
                self.model.move_player(dx*self.model.player.speed, dy*self.model.player.speed)

            self.view.draw(self.screen, self.audio.record_armed, self.audio.recorder.is_recording(),
                           self.show_prompt, self.prompt_text)
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()


# ================= Utils =================
def print_inventory(inv_dir: str):
    try:
        files = sorted(f for f in os.listdir(inv_dir) if f.lower().endswith(".wav"))
    except FileNotFoundError:
        files = []
    if not files:
        print("[INV] (empty)"); return
    print("[INV] Recorded clips:")
    for f in files: print("  -", f)

# ================= Entrypoint =================
if __name__ == "__main__":
    try:
        GameController().run()
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
