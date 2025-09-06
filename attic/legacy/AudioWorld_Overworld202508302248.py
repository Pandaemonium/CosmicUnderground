#!/usr/bin/env python3
# Alien DJ – Overworld zones + POIs, priority audio (2 workers), immediate crossfade
# Requirements: pygame, numpy, soundfile, torch, stable_audio_tools, promptgen.py in same folder

import os, sys, math, time, uuid, tempfile, wave, threading, queue, hashlib, secrets, random, itertools, heapq
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import pygame
import soundfile as sf

import promptgen  # your slim ~10–15 word prompt builder

# --- Display/config ---
DEFAULT_FULLSCREEN = True
SCREEN_W, SCREEN_H = 1200, 700
FPS = 60
TILE_W, TILE_H = 100, 120
ENGINE_SR = 44100

# --- World/map config ---
MAP_W, MAP_H = 96, 96              # huge finite map
ZONE_MIN, ZONE_MAX = 10, 40         # contiguous blob size per zone
AVG_ZONE = 20                       # target avg (used for seed count)
POIS_NPC_RANGE = (0, 2)
POIS_OBJ_RANGE = (0, 1)

START_TILE = (MAP_W // 2, MAP_H // 2)
START_ZONE_NAME  = "Scrapyard Funk"
START_THEME_WAV  = r"C:\Games\CosmicUnderground\inventory\rec_1756545018_Scrapyard Funk_d5ae11.wav"

PLAYER_SPRITE = r"C:\Games\CosmicUnderground\sprites\character1.png"
PLAYER_SPRITE_COMPLETE = "C:\Games\CosmicUnderground\sprites\laser_bunny.png"  # swap to a different file later if you like

# Generation/caching
MAX_ACTIVE_LOOPS = 120             # LRU cap
GEN_WORKERS = 1                    # concurrent Stable Audio generations





# ======= Data =======
@dataclass
class ZoneSpec:
    name: str
    bpm: int
    key_mode: str
    scene: str
    mood: str
    bars: int = 8
    timesig: Tuple[int,int] = (4,4)
    prompt_override: Optional[str] = None
    biome: str = "Unknown"
    species: str = "Unknown"
    descriptors: List[str] = field(default_factory=list)
    instruments: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

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
    id: int
    spec: ZoneSpec
    tiles: List[Tuple[int,int]]
    centroid: Tuple[float,float]
    loop: Optional[GeneratedLoop] = None
    generating: bool = False
    error: Optional[str] = None

@dataclass
class POI:
    pid: int
    kind: str          # "npc" | "object"
    name: str
    role: str          # "performer" | "resonator" | "boss" | etc.
    tile: Tuple[int, int]
    zone_id: int

    rarity: int = 0
    kind_key: Optional[str] = None
    mood_hint: Optional[str] = None
    bpm_hint: Optional[int] = None
    bars_hint: Optional[int] = None

    # runtime (must exist so worker can set these)
    generating: bool = False
    loop: Optional["GeneratedLoop"] = None
    error: Optional[str] = None
    last_seed: Optional[int] = None


@dataclass
class Player:
    tile_x: int
    tile_y: int
    px: float            # pixel coords for smooth camera
    py: float
    speed: float = 6.0

@dataclass
class Quest:
    giver_pid: int
    target_pid: int
    target_name: str
    target_tile: Tuple[int,int]
    target_zone: int
    target_zone_name: str
    accepted: bool = True  # we pop the card immediately on interact


# ======= Prompt helpers (light biasing without changing your promptgen) =======
def tokens_for_poi(poi: POI) -> List[str]:
    """Optional prepend tokens to bias short prompts without changing promptgen."""
    if poi.name == "Boss Skuggs":
        return ["cosmic", "talkbox", "boogie"]  # signature
    if poi.kind == "npc":
        return ["alien", "funk", "bass"]        # lean funkier
    if poi.kind == "object":
        return ["weird", "motif"]               # sparser
    return []

# ======= Stable Audio backend =======
class LocalStableAudioProvider:
    def __init__(self, context_fn=None):
        try:
            import torch
            from stable_audio_tools import get_pretrained_model
            self._torch = torch
            self._get_pretrained_model = get_pretrained_model
        except Exception as e:
            raise RuntimeError(
                "Stable Audio backend unavailable. In your venv:\n"
                "  pip install --no-deps stable-audio-tools==0.0.19\n"
                "  pip install numpy einops soundfile huggingface_hub pygame\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu121  (or cpu)"
            ) from e

        os.environ.setdefault("OMP_NUM_THREADS", "2")
        self._torch.set_num_threads(max(1, os.cpu_count() // 2))
        self.device = (
            "cuda" if self._torch.cuda.is_available() else
            ("mps" if getattr(self._torch.backends, "mps", None)
                     and self._torch.backends.mps.is_available() else "cpu")
        )
        print(f"[StableLocal] Loading model… (device={self.device})", flush=True)
        self.model, self.cfg = self._get_pretrained_model("stabilityai/stable-audio-open-small")
        self.model = self.model.to(self.device).eval()
        self.sr_model = int(self.cfg["sample_rate"])
        print(f"[StableLocal] Ready. sr={self.sr_model}", flush=True)

        self.tmpdir = tempfile.mkdtemp(prefix="alien_dj_local_")
        self.context_fn = (context_fn if callable(context_fn) else (lambda: {}))

    @staticmethod
    def _duration_for(bpm: int, bars: int, timesig: Tuple[int,int]) -> float:
        beats = bars * (timesig[0] if timesig and len(timesig) else 4)
        return beats * (60.0 / max(1, bpm))

    def _ensure_length(self, wav_np: np.ndarray, sr: int, seconds_total: float) -> np.ndarray:
        target = int(round(seconds_total * sr))
        T = wav_np.shape[0]
        if T < target:
            reps = int(math.ceil(target / max(1, T)))
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

    def generate(self, spec_like: Any, prepend: Optional[List[str]] = None) -> GeneratedLoop:
        """spec_like can be a ZoneSpec or a dict carrying similar fields."""
        from stable_audio_tools.inference.generation import generate_diffusion_cond
        import torch

        # Coerce fields
        bpm      = int(getattr(spec_like, "bpm", 112) if hasattr(spec_like, "bpm") else spec_like.get("bpm", 112))
        bars     = int(getattr(spec_like, "bars", 8) if hasattr(spec_like, "bars") else spec_like.get("bars", 8))
        timesig  = getattr(spec_like, "timesig", (4,4)) if hasattr(spec_like, "timesig") else spec_like.get("timesig", (4,4))

        ctx = {}
        try:
            ctx = self.context_fn() or {}
        except Exception:
            ctx = {}

        # Prompt
        prompt, meta = promptgen.build(spec_like, bars=bars, rng=random.Random(), intensity=0.6, **ctx)
        if prepend:
            # Prepend bias tokens without blowing up length too much
            prompt = " ".join(prepend[:3]) + " " + prompt

        seconds_total = self._duration_for(bpm, bars, timesig)
        sample_size   = int(round(seconds_total * self.sr_model))
        seed = secrets.randbits(31)
        print(f"[StableLocal] seed={seed} | '{getattr(spec_like, 'name', meta.get('zone_name','Zone'))}'", flush=True)

        t0 = time.time()
        with self._torch.inference_mode():
            audio = generate_diffusion_cond(
                self.model,
                steps=8, cfg_scale=1.0, sampler_type="pingpong",
                conditioning=[{"prompt": prompt, "seconds_total": seconds_total}],
                sample_size=sample_size, device=self.device, seed=seed
            )
        # (B, D, T) -> (T, D)
        audio = audio.squeeze(0).to(self._torch.float32).cpu().transpose(0,1).numpy().copy()
        peak = float(np.max(np.abs(audio))) or 1.0
        audio = (audio / peak) * 0.98
        audio = self._ensure_length(audio, self.sr_model, seconds_total)
        audio, out_sr = self._resample_linear(audio, self.sr_model, ENGINE_SR)
        duration_actual = float(audio.shape[0]) / float(out_sr)

        out_path = os.path.join(self.tmpdir, f"{uuid.uuid4().hex}.wav")
        sf.write(out_path, audio, out_sr, subtype="PCM_16")
        print(f"[StableLocal] gen in {time.time()-t0:.2f}s → {os.path.basename(out_path)}", flush=True)
        return GeneratedLoop(
            out_path, duration_actual, bpm,
            getattr(spec_like, "key_mode", "Unknown"),
            prompt,
            {"backend":"stable_local","seed":seed,"sr":out_sr, **meta}
        )

# ======= Audio player with A/B crossfade =======
# --- Playback/record ---
class AudioPlayer:
    def __init__(self):
        pygame.mixer.init(frequency=ENGINE_SR, size=-16, channels=2, buffer=1024)
        self.boundary_event = pygame.USEREVENT + 1
        self.current: Optional[str] = None
        self.loop_ms = 0
        pygame.mixer.init(frequency=ENGINE_SR, size=-16, channels=2, buffer=2048)

    def play_loop(self, wav_path: str, duration_sec: float, **kwargs):
        """
        Plays/loops the given wav.

        Accepts any of these fade aliases (first one present wins):
          - cross_ms
          - crossfade_ms
          - xfade_ms
          - fade_ms

        Example calls that all work:
          play_loop(path, dur, cross_ms=200)
          play_loop(path, dur, crossfade_ms=200)
          play_loop(path, dur, fade_ms=200)
        """
        # normalize fade/crossfade argument
        xfade = 0
        for key in ("cross_ms", "crossfade_ms", "xfade_ms", "fade_ms"):
            if key in kwargs and kwargs[key] is not None:
                try:
                    xfade = max(0, int(kwargs[key]))
                except Exception:
                    xfade = 0
                break

        # schedule loop boundary event for record-at-loop-start behavior
        self.loop_ms = max(1, int(duration_sec * 1000))
        pygame.time.set_timer(self.boundary_event, self.loop_ms)

        # if something is already playing, fade it out (or stop) before new loop
        if pygame.mixer.music.get_busy():
            if xfade > 0:
                pygame.mixer.music.fadeout(xfade)
            else:
                pygame.mixer.music.stop()

        try:
            pygame.mixer.music.load(wav_path)
        except pygame.error as e:
            print(f"[Audio] load failed: {e}")
            return

        # fade in the new loop
        pygame.mixer.music.play(loops=-1, fade_ms=xfade)
        self.current = wav_path

    def stop(self, fade_ms: int = 200):
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.fadeout(max(0, int(fade_ms)))
        finally:
            pygame.time.set_timer(self.boundary_event, 0)
            self.current = None
            self.loop_ms = 0



# ======= Recorder (unchanged) =======
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

# ======= Region-growing map + ProcGen =======
class RegionMap:
    """Builds contiguous zone blobs and places POIs, including Boss Skuggs in start zone."""
    ALIENS = {
        "Chillaxians": dict(biomes=["Polar Ice","Crystal Canyons"], bpm=(84,102), mood=["calm","triumphant"],
                            tags=["slow","retro","soft"], inst=["warm pads","ice chimes","soft bass"]),
        "Glorpals":    dict(biomes=["Slime Caves","Groove Pits"],   bpm=(96,122), mood=["playful","melancholy"],
                            tags=["wet","slurpy","layered"], inst=["gel bells","drip fx","squelch synth"]),
        "Bzaris":      dict(biomes=["Centerspire","Crystal Canyons"], bpm=(120,140), mood=["energetic","angry"],
                            tags=["fast","glitchy","futuristic"], inst=["bitcrush blips","granular stutters","noisy hats"]),
        "Shagdeliacs": dict(biomes=["Groove Pits","Centerspire"],   bpm=(96,120), mood=["funky","playful"],
                            tags=["brassy","jazzy","retro"], inst=["wah guitar","clavinet","horn section"]),
        "Rockheads":   dict(biomes=["Groove Pits","Slime Caves"],   bpm=(100,126), mood=["brooding","triumphant"],
                            tags=["mechanical","metallic","loud"], inst=["metal gongs","anvil hits","deep toms"]),
    }
    BIOMES = ["Crystal Canyons","Slime Caves","Centerspire","Groove Pits","Polar Ice","Mag-Lev Yards","Electro Marsh"]
    MODES  = ["major","minor","Dorian","Phrygian","Mixolydian","Lydian","Aeolian"]
    TONICS = ["A","B","C","D","E","F","G","A♭","B♭","C♯","D♯","F♯","G♯"]
    
    ALIEN_FIRST = ["Skuggs", "Jax", "Zor", "Blee", "Kru", "Vex", "Talla", "Moxo", "Rz-7", "Floop"]
    ALIEN_LAST  = ["of Bazaria", "the Fuzzy", "from Groovepit", "Chrome-Snout", "Mag-Skipper", "Buzzwing", "Chilldraft"]
    OBJECT_NAMES = {
        "amp": ["Chromeblaster", "Groove Reactor", "Ion Stack", "Neon Cab", "Riff Turbine"],
        "boombox": ["Slimebox", "Funk Beacon", "Pulse Cube", "Laser Luggable"],
        "gong": ["Shard Gong", "Cavern Gong", "Mag Gong", "Phase Gong"],
        "terminal": ["Beat Kiosk", "Loop Terminal", "Wave Vender", "Rhythm Post"],
    }

    def __init__(self, w: int, h: int, seed: int):
            self.w, self.h = w, h
            self.seed = seed
            self.rng = random.Random(seed)
    
            self.zone_of: List[List[int]] = [[-1]*h for _ in range(w)]
            self.zones: Dict[int, ZoneRuntime] = {}
            self.pois: Dict[int, POI] = {}
            self.pois_at: Dict[Tuple[int,int], int] = {}
    
            # ✅ make this BEFORE _procgen_zone_specs uses it
            self.zone_color: Dict[int, Tuple[int,int,int]] = {}
    
            self._build_regions()
            self._procgen_zone_specs()
            self._place_pois()

    # --- Region growing ---
    def _build_regions(self):
        total_tiles = self.w * self.h
        target_zones = max(1, total_tiles // AVG_ZONE)
        # Build zone budgets 10..40 until we cover the map
        budgets = []
        s = 0
        while s < total_tiles:
            k = self.rng.randint(ZONE_MIN, ZONE_MAX)
            budgets.append(k); s += k
        # Seeds: choose distinct unassigned tiles
        unassigned = {(x, y) for x in range(self.w) for y in range(self.h)}
        seeds = []
        for _ in range(len(budgets)):
            if not unassigned: break
            t = self.rng.choice(tuple(unassigned))
            seeds.append(t)
            unassigned.remove(t)

        # Initialize frontier per zone id
        zone_id = 0
        frontiers: Dict[int, List[Tuple[int,int]]] = {}
        want: Dict[int, int] = {}
        for s_tile, budget in zip(seeds, budgets):
            x, y = s_tile
            self.zone_of[x][y] = zone_id
            want[zone_id] = max(ZONE_MIN, min(ZONE_MAX, budget))
            frontiers[zone_id] = [s_tile]
            zone_id += 1

        # Multi-source random BFS growth
        active = set(frontiers.keys())
        while active:
            zid = self.rng.choice(tuple(active))
            if want[zid] <= 0:
                active.remove(zid); continue
            if not frontiers[zid]:
                active.remove(zid); continue

            fx, fy = frontiers[zid].pop(0)
            # neighbors (4-neighborhood; diagonals allowed to "touch only at corners" naturally)
            for nx, ny in ((fx+1,fy),(fx-1,fy),(fx,fy+1),(fx,fy-1)):
                if 0 <= nx < self.w and 0 <= ny < self.h and self.zone_of[nx][ny] == -1:
                    self.zone_of[nx][ny] = zid
                    frontiers[zid].append((nx, ny))
                    want[zid] -= 1
                    if want[zid] <= 0: break
            if want[zid] <= 0:
                active.remove(zid)

        # Assign any leftovers to nearest assigned neighbor
        for x in range(self.w):
            for y in range(self.h):
                if self.zone_of[x][y] != -1: continue
                # pick random neighboring assigned
                cand = []
                for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1),(x+1,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1)):
                    if 0 <= nx < self.w and 0 <= ny < self.h and self.zone_of[nx][ny] != -1:
                        cand.append(self.zone_of[nx][ny])
                self.zone_of[x][y] = self.rng.choice(cand) if cand else 0

        # Build zone tile lists
        tiles_by_zone: Dict[int, List[Tuple[int,int]]] = {}
        for x in range(self.w):
            for y in range(self.h):
                zid = self.zone_of[x][y]
                tiles_by_zone.setdefault(zid, []).append((x,y))
        # Make ZoneRuntime shells (specs filled next)
        for zid, tl in tiles_by_zone.items():
            cx = sum(t[0] for t in tl) / len(tl)
            cy = sum(t[1] for t in tl) / len(tl)
            self.zones[zid] = ZoneRuntime(zid, None, tl, (cx, cy), None, False, None)
        
        # Stable label anchor per zone (top-left tile in world coords)
        self.zone_anchor: Dict[int, Tuple[int,int]] = {}
        for zid, tl in tiles_by_zone.items():
            self.zone_anchor[zid] = min(tl)  # lexicographic = top-left
        
        self.neighbors: Dict[int, set[int]] = {zid:set() for zid in self.zones}
        for x in range(self.w):
            for y in range(self.h):
                a = self.zone_of[x][y]
                for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                    if 0 <= nx < self.w and 0 <= ny < self.h:
                        b = self.zone_of[nx][ny]
                        if a != b:
                            self.neighbors[a].add(b)
                            self.neighbors[b].add(a)

    def _procgen_zone_specs(self):
        for zid, zr in self.zones.items():
            rng = random.Random((self.seed << 1) ^ (zid * 2654435761))
            biome = rng.choice(self.BIOMES)
            species = rng.choice(list(self.ALIENS.keys()))
            # prefer matching biome
            for _ in range(2):
                p = rng.choice(list(self.ALIENS.keys()))
                if biome in self.ALIENS[p]["biomes"]:
                    species = p; break
            a = self.ALIENS[species]
            bpm = rng.randrange(a["bpm"][0], a["bpm"][1]+1, 2)
            mood = rng.choice(a["mood"])
            tags = list(set(a["tags"]))
            instruments = list(set(a["inst"]))
            # name
            if START_TILE in zr.tiles:
                name = START_ZONE_NAME
            else:
                A = ["Neon","Crystal","Slime","Chrome","Velvet","Magnetic","Shatter","Rusty","Electric","Gleaming","Polar","Echo"]
                L = ["Bazaar","Canyons","Caves","Centerspire","Grove","Pits","Foundry","Sprawl","Galleries","Arcade","Causeway","Yards"]
                T = ["Funk","Pulse","Flux","Jitter","Parade","Breaks","Mallets","Reverb","Drift","Chromatics","Riff"]
                base = f"{rng.choice(A)} {rng.choice(L)}"
                if rng.random() < 0.4: base = f"{base} {rng.choice(T)}"
                name = base
            key_mode = f"{rng.choice(self.TONICS)} {rng.choice(self.MODES)}"
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
            zr.spec = ZoneSpec(
                name=name, bpm=bpm, key_mode=key_mode, scene=scene, mood=mood,
                biome=biome, species=species, descriptors=[], instruments=instruments, tags=tags
            )
            rr = random.Random(zid*1337)
            self.zone_color[zid] = (40 + rr.randint(0,40), 42 + rr.randint(0,30), 70 + rr.randint(0,40))

    # --- POIs ---
    def _place_pois(self):
        next_id = 1
    
        # --- helpers (define before use) ---
        def interior_tiles(zr: ZoneRuntime) -> List[Tuple[int,int]]:
            tl = zr.tiles
            if len(tl) < 4: return tl
            sx = sum(x for x,_ in tl)/len(tl); sy = sum(y for _,y in tl)/len(tl)
            return sorted(tl, key=lambda t: abs(t[0]-sx)+abs(t[1]-sy))
    
        def make_npc_name(rng: random.Random, species: str) -> str:
            if species == "Bzaris":
                return f"{rng.choice(['Bz-','Zz-','Q-'])}{rng.choice(['Skug','Zap','Kli','Vrr'])}{rng.randrange(10,99)}"
            if species == "Glorpals":
                return f"{rng.choice(['Glo','Slu','Dro'])}{rng.choice(['rpo','rma','opa'])}{rng.choice(['x','z'])}"
            base = f"{rng.choice(self.ALIEN_FIRST)} {rng.choice(self.ALIEN_LAST)}"
            return base
    
        def make_object_name(rng: random.Random, kind: str) -> str:
            pool = self.OBJECT_NAMES.get(kind, ["Artifact", "Thingy", "Widget"])
            return rng.choice(pool)
    
        # --- Boss Skuggs in start zone ---
        for zid, zr in self.zones.items():
            if START_TILE in zr.tiles:
                home = interior_tiles(zr)[0]
                poi = POI(next_id, "npc", "Boss Skuggs", "boss", home, zid, rarity=10)
                self.pois[next_id] = poi; self.pois_at[home] = next_id
                next_id += 1
                break
    
        rng = self.rng
        for zid, zr in self.zones.items():
            # counts
            if START_TILE in zr.tiles:
                npc_n = rng.randint(max(0, POIS_NPC_RANGE[0]-1), max(1, POIS_NPC_RANGE[1]))  # already 1 boss exists
                obj_n = rng.randint(*POIS_OBJ_RANGE)
            else:
                npc_n = rng.randint(*POIS_NPC_RANGE)
                obj_n = rng.randint(*POIS_OBJ_RANGE)
    
            candidates = [t for t in interior_tiles(zr) if t not in self.pois_at]
            rng.shuffle(candidates)
    
            # NPCs (✅ use alien names + ✅ increment next_id)
            for _ in range(npc_n):
                if not candidates: break
                tile = candidates.pop(0)
                name = make_npc_name(rng, zr.spec.species if zr.spec else "Unknown")
                poi = POI(next_id, "npc", name, "performer", tile, zid, rarity=rng.randint(0,3))
                self.pois[next_id] = poi; self.pois_at[tile] = next_id
                next_id += 1
    
            # Objects
            for _ in range(obj_n):
                if not candidates: break
                tile = candidates.pop(0)
                name = rng.choice(["Crystal Resonator","Slime Drum","Mag Lev Bell","Arcade Cabinet"])
                poi = POI(next_id, "object", name, "resonator", tile, zid, rarity=rng.randint(0,2))
                self.pois[next_id] = poi; self.pois_at[tile] = next_id
                next_id += 1
    
        # --- Place special quest_giver beacon in a neighbor zone of the start ---
        start_zid = self.zone_of[START_TILE[0]][START_TILE[1]]
        neighs = list(self.neighbors.get(start_zid, []))
        rng.shuffle(neighs)
        for qz in neighs:
            zr = self.zones[qz]
            tiles = [t for t in interior_tiles(zr) if t not in self.pois_at]
            if not tiles: 
                continue
            tile = tiles[0]
            beacon = POI(next_id, "object", "Beacon of Names", "quest_giver", tile, qz, rarity=99, kind_key="beacon")
            self.pois[next_id] = beacon
            self.pois_at[tile] = next_id
            next_id += 1
            break

                
        def make_npc_name(rng: random.Random, species: str) -> str:
            # Species can bias prefixes if you want; simple for now
            if species == "Bzaris":
                return f"{rng.choice(['Bz-', 'Zz-', 'Q-'])}{rng.choice(['Skug','Zap','Kli','Vrr'])}{rng.randrange(10,99)}"
            if species == "Glorpals":
                return f"{rng.choice(['Glo','Slu','Dro'])}{rng.choice(['rpo','rma','opa'])}{rng.choice(['x','z'])}"
            base = f"{rng.choice(self.ALIEN_FIRST)} {rng.choice(self.ALIEN_LAST)}"
            return base
        
        def make_object_name(rng: random.Random, kind: str) -> str:
            pool = self.OBJECT_NAMES.get(kind, ["Artifact", "Thingy", "Widget"])
            return rng.choice(pool)
        
        # --- SPECIAL: place a quest-giving beacon in a zone adjacent to start ---
        # figure out starting zone
        start_zid = self.zone_of[START_TILE[0]][START_TILE[1]]
        ngh = list(self.neighbors.get(start_zid, []))
        if ngh:
            rng.shuffle(ngh)
            qz = ngh[0]  # pick one adjacent zone
            zr_q = self.zones[qz]
            # find an interior tile not already used
            cand = [t for t in zr_q.tiles if t not in self.pois_at]
            if not cand:
                cand = zr_q.tiles[:]  # fallback
            rng.shuffle(cand)
            if cand:
                q_tile = cand[0]
                q_name = "Laser Beacon"
                poi = POI(next_id, "object", q_name, "quest_giver", q_tile, qz, rarity=999)
                self.pois[next_id] = poi
                self.pois_at[q_tile] = next_id
                # expose for outside
                self.quest_giver_pid = next_id
                next_id += 1

# ======= World model =======
class WorldModel:
    def __init__(self):
        self.world_seed = secrets.randbits(32)
        self.map = RegionMap(MAP_W, MAP_H, self.world_seed)

        sx, sy = START_TILE
        self.player = Player(sx, sy, sx*TILE_W + TILE_W/2, sy*TILE_H + TILE_H/2, speed=6.0)
        self.current_zone_id = self.map.zone_of[sx][sy]

        # runtime flags/context
        self.time_of_day = "night"
        self.weather = None
        self.heat = 0.15
        self.debt_pressure = 0.4
        self.festival = False
        self.cosmic_event = False

        # listeners
        self._tile_listeners = []
        self._zone_listeners: List[Any] = []
        
        self.quest: Optional[Quest] = None
        self.active_quest: Optional[Quest] = None
        self.quest_giver_pid = getattr(self.map, "quest_giver_pid", None)
        self.quest_completed = False
        
        def add_tile_changed_listener(self, fn):
            self._tile_listeners.append(fn)
        def move_player(self, dx, dy):
            old_tx, old_ty = self.player.tile_x, self.player.tile_y
            self.player.px += dx; self.player.py += dy
            self.player.tile_x = max(0, min(MAP_W-1, int(self.player.px // TILE_W)))
            self.player.tile_y = max(0, min(MAP_H-1, int(self.player.py // TILE_H)))
            if (self.player.tile_x, self.player.tile_y) != (old_tx, old_ty):
                for fn in self._tile_listeners: fn((old_tx,old_ty), (self.player.tile_x, self.player.tile_y))

    def add_tile_changed_listener(self, fn): self._tile_listeners.append(fn)
    def add_zone_changed_listener(self, fn): self._zone_listeners.append(fn)

    def move_player(self, dx: float, dy: float):
        # pixel move
        self.player.px += dx; self.player.py += dy
        # clamp into map
        self.player.px = max(0, min(self.player.px, MAP_W*TILE_W-1))
        self.player.py = max(0, min(self.player.py, MAP_H*TILE_H-1))
        # tile
        nx = int(self.player.px // TILE_W)
        ny = int(self.player.py // TILE_H)
        if (nx,ny) != (self.player.tile_x, self.player.tile_y):
            oldt = (self.player.tile_x, self.player.tile_y)
            self.player.tile_x, self.player.tile_y = nx, ny
            for fn in self._tile_listeners: fn(oldt, (nx,ny))

            zid = self.map.zone_of[nx][ny]
            if zid != self.current_zone_id:
                oldz = self.current_zone_id; self.current_zone_id = zid
                for fn in self._zone_listeners: fn(oldz, zid)

# ======= Audio service (2 workers, zone+POI sources, priority) =======
class AudioService:
    """
    Priority tiers recomputed on tile change:
      0 = current audible source (zone or poi)
      1 = POIs within radius 1 of the player
      2 = zone neighbors in view
      3 = backlog
    """
    def __init__(self, model: WorldModel):
        self.m = model
        # context → promptgen
        def _ctx():
            return dict(
                time_of_day=self.m.time_of_day,
                weather=self.m.weather,
                heat=self.m.heat,
                debt_pressure=self.m.debt_pressure,
                festival=self.m.festival,
                cosmic_event=self.m.cosmic_event,
            )
        self.provider = LocalStableAudioProvider(context_fn=_ctx)
        self.player = AudioPlayer()
        self.recorder = Recorder()
        self.record_armed = False
        self.auto_stop_at_end = True

        # Task queue
        self._heap = []  # (prio_tuple, seq, task_dict)
        self._counter = itertools.count()
        self._pending: Dict[Tuple[str,int], int] = {}  # (kind,id) -> latest token
        self._lock = threading.Lock()

        # Active source
        self.active_source: Tuple[str,int] = ("zone", self.m.current_zone_id)
        


        # initialize once at startup
        self.active_source = self._current_active_source()
        rt0 = self._get_runtime(self.active_source)
        if rt0.loop:
            self.player.play_loop(rt0.loop.wav_path, rt0.loop.duration_sec, fade_ms=140)
        else:
            # try preload for start zone, fall back to generate
            if not self._maybe_preload_start_zone():
                self.request_generate(self.active_source, priority=(0,0), force=True)
        self._reprioritize_all()
        
        # now start the worker (provider must already exist)
        self._worker_stop = False
        self._workers = []
        for _ in range(max(1, GEN_WORKERS)):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)


        # listeners
        self.m.add_tile_changed_listener(self.on_tile_changed)
        self.m.add_zone_changed_listener(self.on_zone_changed)

        # preload theme for start zone
        self._maybe_preload_zone(self.m.current_zone_id)
        # ensure initial audible source is playing
        self._ensure_audible_playing()

        # prefetch immediate neighbors
        self._reprioritize_all()

    # ---------- audible source selection ----------
    @staticmethod
    def _chebyshev(a: Tuple[int,int], b: Tuple[int,int]) -> int:
        return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

    def _nearby_pois(self) -> List[POI]:
        px, py = self.m.player.tile_x, self.m.player.tile_y
        out = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                tx, ty = px+dx, py+dy
                if not (0 <= tx < MAP_W and 0 <= ty < MAP_H): continue
                pid = self.m.map.pois_at.get((tx,ty))
                if pid: out.append(self.m.map.pois[pid])
        # Order by distance, then kind preference (npc over object), then rarity desc
        out.sort(key=lambda p: (abs(p.tile[0]-px)+abs(p.tile[1]-py), 0 if p.kind=="npc" else 1, -p.rarity))
        return out

    def _pick_audible(self) -> Tuple[str,int]:
        cands = self._nearby_pois()
        if cands:
            return ("poi", cands[0].pid)
        return ("zone", self.m.current_zone_id)

    def _ensure_audible_playing(self):
        kind, idv = self.active_source
        if kind == "zone":
            zr = self.m.map.zones[idv]
            if zr.loop:
                self.player.play_loop(zr.loop.wav_path, zr.loop.duration_sec, cross_ms=220)
            else:
                self.request_zone(idv, priority=(0,0), force=True)
        else:
            poi = self.m.map.pois[idv]
            if poi.loop:
                self.player.play_loop(poi.loop.wav_path, poi.loop.duration_sec, cross_ms=220)
            else:
                self.request_poi(idv, priority=(0,0), force=True)

    # ---------- tasks ----------
    def request_zone(self, zid: int, priority: Optional[Tuple[int,int]]=None, force: bool=False):
        prio = priority or self._priority_for(("zone", zid))
        self.request_generate(("zone", zid), priority=prio, force=force)

    def request_poi(self, pid: int, priority: Optional[Tuple[int,int]]=None, force: bool=False):
        prio = priority or self._priority_for(("poi", pid))
        self.request_generate(("poi", pid), priority=prio, force=force)

    def _pop_task(self):
        with self._lock:
            while self._heap:
                prio, token, src = heapq.heappop(self._heap)
                if self._pending.get(src) != token:
                    continue  # stale
                return prio, token, src
        return None

    # ---------- worker ----------
    def _worker_loop(self):
        import time
        while not self._worker_stop:
            item = self._pop_task()
            if item is None:
                time.sleep(0.01); continue
            _, _, src = item
            rt = self._get_runtime(src)
            if rt.generating or rt.loop is not None:
                continue
            rt.generating = True; rt.error = None
            try:
                loop = self._generate_for(src)
                rt.loop = loop
                # if the thing we just generated is the *currently audible* source, play it now
                if src == self.active_source:
                    self.player.play_loop(loop.wav_path, loop.duration_sec, fade_ms=200)
            except Exception as e:
                rt.error = str(e)
                print(f"[FATAL][GEN] {src}: {e}", flush=True)
            finally:
                rt.generating = False
            self._prune_cache()

    # ---------- priority ----------
    def _priority_for_zone(self, zid: int) -> Tuple[int,int,int]:
        """(tier, manhattan_dist_to_centroid, -recency) — here recency omitted for simplicity."""
        px, py = self.m.player.tile_x, self.m.player.tile_y
        kind, aid = self.active_source
        tier = 3
        if kind == "zone" and aid == zid: tier = 0
        cx, cy = self.m.map.zones[zid].centroid
        dist = int(abs(px - cx) + abs(py - cy))
        # neighbors (by centroid approx) get tier 2
        if tier != 0 and dist <= 4: tier = 2
        return (tier, dist, 0)

    def _priority_for_poi(self, pid: int) -> Tuple[int,int,int]:
        px, py = self.m.player.tile_x, self.m.player.tile_y
        poi = self.m.map.pois[pid]
        kind, aid = self.active_source
        tier = 3
        if kind == "poi" and aid == pid: tier = 0
        dist = abs(px - poi.tile[0]) + abs(py - poi.tile[1])
        if tier != 0 and self._chebyshev((px,py), poi.tile) <= 1: tier = 1
        return (tier, dist, -poi.rarity)
    
    def _priority_for(self, src: Tuple[str,int]) -> Tuple[int,int]:
        # tier 0: current active source
        if src == self.active_source:
            return (0, 0)
        # tier 1: zones adjacent to player’s zone
        px, py = self.m.player.tile_x, self.m.player.tile_y
        cz = self.m.map.zone_of[px][py]
        adj = set()
        for (nx, ny) in ((px+1,py),(px-1,py),(px,py+1),(px,py-1)):
            if 0 <= nx < self.m.map.w and 0 <= ny < self.m.map.h:
                adj.add(self.m.map.zone_of[nx][ny])
        if src[0] == "zone" and src[1] in adj:
            # distance ~1 for all adjacent zones
            return (1, 1)
        # everything else
        return (2, 99)
        
    def _reprioritize_all(self):
        with self._lock:
            items = list(self._pending.keys())
            self._heap.clear()
            self._pending.clear()
    
        # ensure active first
        rt = self._get_runtime(self.active_source)
        if rt.loop is None and not rt.generating:
            self.request_generate(self.active_source, priority=(0,0), force=True)
    
        # adjacent zones next (no tile scan!)
        if self.active_source[0] == "zone":
            cz = self.active_source[1]
        else:
            # fall back to player's current zone
            cz = self.m.current_zone_id
        for zid in self.m.map.neighbors.get(cz, ()):
            rt = self._get_runtime(("zone", zid))
            if rt.loop is None and not rt.generating:
                self.request_generate(("zone", zid), priority=(1,1), force=False)
    
        # backlog
        for src in items:
            rt = self._get_runtime(src)
            if rt.loop is None and not rt.generating:
                self.request_generate(src, priority=self._priority_for(src), force=False)


    # ---------- cache pruning ----------
    def _prune_cache(self):
        # Count loops
        zone_loops = sum(1 for z in self.m.map.zones.values() if z.loop)
        poi_loops  = sum(1 for p in self.m.map.pois.values() if p.loop)
        total = zone_loops + poi_loops
        if total <= MAX_ACTIVE_LOOPS: return
        # Evict farthest POIs first, then zones
        px, py = self.m.player.tile_x, self.m.player.tile_y
        # Make eviction list of (distance, kind, id)
        cand: List[Tuple[int,str,int]] = []
        for p in self.m.map.pois.values():
            if p.loop and not p.generating:
                d = abs(px-p.tile[0])+abs(py-p.tile[1])
                cand.append((d,"poi",p.pid))
        for zid, z in self.m.map.zones.items():
            if z.loop and not z.generating and zid != self.m.current_zone_id:
                cx, cy = z.centroid
                d = int(abs(px-cx)+abs(py-cy))
                cand.append((d,"zone",zid))
        cand.sort(reverse=True)
        to_remove = total - MAX_ACTIVE_LOOPS
        for _,k,i in cand[:to_remove]:
            if k=="poi":
                self.m.map.pois[i].loop = None
            else:
                self.m.map.zones[i].loop = None

    # ---------- events ----------
    def on_zone_changed(self, old_z, new_z):
        self._maybe_preload_zone(new_z)
        # Force a fresh loop request for the new zone if we don't have one yet
        zr = self.m.map.zones[new_z]
        if zr.loop is None and not zr.generating:
            self.request_zone(new_z, priority=(0,0), force=True)
        self._reprioritize_all()
        self._maybe_switch()


    def on_tile_changed(self, old_t: Tuple[int,int], new_t: Tuple[int,int]):
        self._reprioritize_all()
        self._maybe_switch()

    def _maybe_switch(self):
        next_src = self._pick_audible()
        if next_src != self.active_source:
            self.active_source = next_src
            self._ensure_audible_playing()

    # ---------- preload theme ----------
    def _maybe_preload_zone(self, zid: int):
        z = self.m.map.zones[zid]
        if START_TILE not in z.tiles or z.loop: return
        path = START_THEME_WAV
        if not os.path.isfile(path): return
        try:
            with sf.SoundFile(path) as f:
                duration = len(f) / float(f.samplerate)
        except Exception:
            return
        z.loop = GeneratedLoop(path, duration, z.spec.bpm, z.spec.key_mode, f"Preloaded theme: {z.spec.name}", {"backend":"preloaded"})
        if self.active_source == ("zone", zid):
            self.player.play_loop(z.loop.wav_path, z.loop.duration_sec, cross_ms=180)

    def _maybe_preload_start_zone(self) -> bool:
        k, i = self.active_source
        if k != "zone": return False
        # reuse your START_THEME_WAV logic here
        return self._maybe_preload_zone(i)

    # ---------- recording ----------
    def on_boundary_tick(self):
        if self.record_armed and not self.recorder.is_recording():
            kind, idv = self.active_source
            if kind == "zone":
                z = self.m.map.zones[idv]
                if z.loop:
                    self.recorder.start(z.loop.wav_path, z.loop.duration_sec,
                                        {"zone": z.spec.name, "bpm": z.spec.bpm, "key": z.spec.key_mode, "mood": z.spec.mood})
                    self.record_armed = False
            else:
                p = self.m.map.pois[idv]
                if p.loop:
                    home = self.m.map.zones[p.zone_id].spec.name
                    self.recorder.start(p.loop.wav_path, p.loop.duration_sec,
                                        {"zone": f"{home}/{p.name}", "bpm": self.m.map.zones[p.zone_id].spec.bpm, "key": self.m.map.zones[p.zone_id].spec.key_mode, "mood": "funky"})
                    self.record_armed = False
        elif self.recorder.is_recording() and self.auto_stop_at_end:
            self.recorder.stop()
    
    # Which source is audible right now? (POI within radius 1 beats zone)
    def _current_active_source(self) -> Tuple[str, int]:
        px, py = self.m.player.tile_x, self.m.player.tile_y
        # find nearest POI with Chebyshev distance <= 1
        best = None
        best_d = 999
        for pid, poi in self.m.map.pois.items():
            dx = abs(poi.tile[0] - px)
            dy = abs(poi.tile[1] - py)
            d  = max(dx, dy)
            if d <= 1:
                # prefer NPC over object if tie; otherwise nearest
                rank = (0 if poi.kind == "npc" else 1, d)
                if best is None or rank < best_d:
                    best = ("poi", pid)
                    best_d = rank
        if best: return best
        zid = self.m.map.zone_of[px][py]
        return ("zone", zid)
    
    def _get_runtime(self, src: Tuple[str,int]):
        k, i = src
        if k == "zone":
            return self.m.map.zones[i]
        else:
            return self.m.map.pois[i]
    
    def request_generate(self, src: Tuple[str,int], priority: Tuple[int,int]=(0,0), force: bool=False):
        rt = self._get_runtime(src)
        if not force and (rt.generating or rt.loop is not None or rt.error):
            return
        with self._lock:
            token = next(self._counter)
            self._pending[src] = token
            heapq.heappush(self._heap, (priority, token, src))
            
    def _generate_for(self, src: Tuple[str,int]) -> Optional["GeneratedLoop"]:
        k, i = src
        if k == "zone":
            zr = self.m.map.zones[i]
            return self.provider.generate(zr.spec)
        else:
            poi = self.m.map.pois[i]
            base = self.m.map.zones[poi.zone_id].spec
            bpm  = poi.bpm_hint or random.randint(96, 126)
            bars = poi.bars_hint or base.bars
            zspec = ZoneSpec(
                name=poi.name, bpm=bpm, key_mode=base.key_mode, scene=base.scene,
                mood=(poi.mood_hint or ("energetic" if poi.kind=="npc" else "mysterious")),
                bars=bars, timesig=base.timesig,
                biome=base.biome, species=base.species,
                descriptors=(base.descriptors + (["syncopated","funky"] if poi.kind=="npc" else ["textural"])),
                instruments=(base.instruments + (["rubber synth bass","clavinet","wah guitar"] if poi.kind=="npc" else ["mallets","resonant metal"])),
                tags=list(set(base.tags + (["fast","layered"] if poi.kind=="npc" else ["retro","soft"]))),
            )
            return self.provider.generate(zspec, prepend=tokens_for_poi(poi))


# ======= View =======
class GameView:
    def __init__(self, model: WorldModel, audio: AudioService):
        self.m = model
        self.audio = audio
        self.font = pygame.font.SysFont("consolas", 18)
        self.small = pygame.font.SysFont("consolas", 14)
        
        # --- NEW: zone title fonts (per-biome), cache, and sizing
        self.zone_font_cache: dict[tuple[str,int], pygame.font.Font] = {}
        self.zone_name_px = 30  # size for big zone names (tweak to taste)
        
        # Pick families that usually exist on Windows/macOS/Linux; will fallback if missing.
        self.biome_font_family = {
            "Crystal Canyons": "georgia",       # serif, crystalline vibe
            "Slime Caves": "tahoma",            # clean sans
            "Centerspire": "consolas",          # techy mono
            "Groove Pits": "verdana",           # chunky groove
            "Polar Ice": "trebuchetms",         # airy sans
            "Mag-Lev Yards": "couriernew",      # industrial mono
            "Electro Marsh": "segoe ui",        # modern sans
        }
        
        self.player_img = self._load_sprite(PLAYER_SPRITE, target_h=48)
        self.player_img_complete = self._load_sprite(PLAYER_SPRITE_COMPLETE, target_h=48)
        
        # If a family isn’t found, SysFont will pick “best available”.
    def _get_biome_font(self, biome: str, size: int) -> pygame.font.Font:
        key = (biome, size)
        if key in self.zone_font_cache:
            return self.zone_font_cache[key]
        family = self.biome_font_family.get(biome, "trebuchetms")
        try:
            font = pygame.font.SysFont(family, size, bold=True)
        except Exception:
            font = pygame.font.SysFont(None, size, bold=True)
        self.zone_font_cache[key] = font
        return font
    
    def _blit_text_outline(self, surf: pygame.Surface, text: str, font: pygame.font.Font,
                           x: int, y: int, fill=(240,240,255), outline=(255,255,255), px: int = 2):
        # Draw an outline by rendering the text offset in 8 directions, then fill on top
        txt = font.render(text, True, fill)
        out = font.render(text, True, outline)
        for ox, oy in ((-px,0),(px,0),(0,-px),(0,px),(-px,-px),(px,-px),(-px,px),(px,px)):
            surf.blit(out, (x+ox, y+oy))
        surf.blit(txt, (x, y))

    def _draw_glow(self, surf: pygame.Surface, rect: pygame.Rect, pulse_t: float):
        glow_color = (170 + int(60*(0.5+0.5*math.sin(pulse_t*2.0))), 60, 220)
        glow = pygame.Surface((rect.w+16, rect.h+16), pygame.SRCALPHA)
        base = pygame.Rect(8, 8, rect.w, rect.h)
        for i, alpha in enumerate((90, 60, 30)):
            pygame.draw.rect(glow, (*glow_color, alpha), base.inflate(8+i*6, 8+i*6), width=3, border_radius=14)
        surf.blit(glow, (rect.x-8, rect.y-8))
    
    def _draw_beacon_glow(self, surf: pygame.Surface, cx: int, cy: int, t: float):
        """Strong pulsating laser-blue glow for the quest giver."""
        # laser blue
        base = (80, 200, 255)
        # fast pulse
        s = (math.sin(t*3.2) * 0.5 + 0.5)  # 0..1
        # 3 concentric rings + soft disc
        for i, alpha in enumerate((180, 120, 70)):
            r = int(22 + i*10 + s*8)
            pygame.draw.circle(surf, (*base, alpha), (cx, cy), r, width=3)
        # soft inner disc
        inner = pygame.Surface((60, 60), pygame.SRCALPHA)
        pygame.draw.circle(inner, (*base, 80 + int(70*s)), (30, 30), 24)
        surf.blit(inner, (cx-30, cy-30), special_flags=pygame.BLEND_PREMULTIPLIED)

    def draw(self, screen: pygame.Surface, record_armed: bool, recorder_active: bool,
         show_prompt: bool=False, prompt_text: str="",
         show_quest: bool=False, quest_text: str=""):

        screen.fill((14,10,18))
        pulse_t = pygame.time.get_ticks()/1000.0
        cam_x = self.m.player.px - SCREEN_W/2
        cam_y = self.m.player.py - SCREEN_H/2

        min_zx = max(0, int(cam_x // TILE_W) - 1)
        max_zx = min(MAP_W-1, int((cam_x + SCREEN_W) // TILE_W) + 1)
        min_zy = max(0, int(cam_y // TILE_H) - 1)
        max_zy = min(MAP_H-1, int((cam_y + SCREEN_H) // TILE_H) + 1)

        active_k, active_id = self.audio.active_source
        active_tile = None
        if active_k == "zone":
            # pick nearest tile of that zone to player (approx: use player's tile)
            active_tile = (self.m.player.tile_x, self.m.player.tile_y)
        else:
            active_tile = self.m.map.pois[active_id].tile

        # Draw tiles
        # Build visible tiles grouped by zone id
        visible_by_zone: Dict[int, List[Tuple[int,int]]] = {}
        visible_pois: List[Tuple[POI, int, int]] = []  # (poi, cx, cy)
        
        for zy in range(min_zy, max_zy+1):
            for zx in range(min_zx, max_zx+1):
                zid = self.m.map.zone_of[zx][zy]
                zr = self.m.map.zones[zid]
                sx = int(zx*TILE_W - cam_x); sy = int(zy*TILE_H - cam_y)
                rect = pygame.Rect(sx+2, sy+2, TILE_W-4, TILE_H-4)
        
                # base tile color per zone
                base_col = self.m.map.zone_color[zid]
                if (zx,zy) == (self.m.player.tile_x, self.m.player.tile_y):
                    base_col = (70, 88, 140)
                pygame.draw.rect(screen, base_col, rect, border_radius=12)
        
                # audible source marker ring
                if active_tile == (zx,zy):
                    pygame.draw.rect(screen, (200,120,255), rect, width=3, border_radius=12)
        
                # collect per-zone tiles
                visible_by_zone.setdefault(zid, []).append((zx, zy))
        
                # collect POIs on this tile (for post-pass)
                pid = self.m.map.pois_at.get((zx,zy))
                if pid:
                    poi = self.m.map.pois[pid]
                    cx = rect.centerx; cy = rect.centery
                    visible_pois.append((poi, cx, cy))
                    # simple marker (we'll do glow in overlay pass)
                    color = (255,220,90) if poi.name=="Boss Skuggs" else ((200,255,200) if poi.kind=="npc" else (200,200,255))
                    pygame.draw.circle(screen, color, (cx, cy), 10)
        
        # --- 2) FX overlay (zone perimeter + beacon glow), then blit once ---
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        
        # zone perimeters (only for ready zones)
        for zid, tiles in visible_by_zone.items():
            zr = self.m.map.zones[zid]
            if zr.loop and not zr.error:
                self._stroke_zone_edges(overlay, tiles, cam_x, cam_y, pulse_t)
        
        # quest beacon strong laser-blue glow
        for poi, cx, cy in visible_pois:
            if poi.role == "quest_giver":
                self._draw_beacon_glow(overlay, cx, cy, pulse_t)
        
        screen.blit(overlay, (0, 0))
        
        # --- NEW: draw BIG zone names (outlined) after overlay, so they sit above tiles
        for zid, tiles in visible_by_zone.items():
            ax, ay = self.m.map.zone_anchor[zid]
            if not (min_zx <= ax <= max_zx and min_zy <= ay <= max_zy):
                continue  # anchor not visible; skip
            px = int(ax*TILE_W - cam_x) + 12
            py = int(ay*TILE_H - cam_y) + 10
            self._blit_label(screen, self.m.map.zones[zid].spec.name, px, py)

        
        # --- 4) POI name labels (centered UNDER marker) ---
        for poi, cx, cy in visible_pois:
            text_w, _ = self.small.size(poi.name)
            pad = 4
            x = int(cx - (text_w + pad*2)//2)
            y = int(cy + 12)  # below the circle (radius ~10) with small gap
            self._blit_label(screen, poi.name, x, y)

        
        

        # player (sprite)
        px = int(self.m.player.px - cam_x)
        py = int(self.m.player.py - cam_y)
        
        img = self.player_img_complete if getattr(self.m, "quest_completed", False) else self.player_img
        if img:
            # center sprite on player position
            rect = img.get_rect(center=(px, py))
            screen.blit(img, rect.topleft)
        else:
            # fallback if sprite failed to load
            pygame.draw.circle(screen, (255,100,120), (px, py), 10)

        # HUD
        hud = ["Move: WASD/Arrows | F interact | G regen zone | M cycle mood | E edit mood | N panel | P prompt | R record | Esc quit"]

        if record_armed:    hud.append("REC ARMED: starts at next loop boundary.")
        if recorder_active: hud.append("RECORDING… R to stop (auto-stops at boundary).")
        y = SCREEN_H - 22*len(hud) - 8
        for line in hud:
            screen.blit(self.font.render(line, True, (240,240,240)), (10, y)); y += 22
        
        # Quest status line
        q = getattr(self.m, "active_quest", None)
        if q:
            hud.append(f"QUEST: Find {q.target_name} in {q.target_zone_name}")

        # Prompt overlay
        if show_prompt and prompt_text:
            # wrap
            max_w = int(SCREEN_W * 0.8)
            pad   = 14
            title_font = pygame.font.SysFont("consolas", 20, bold=True)
            body_font  = pygame.font.SysFont("consolas", 18)
            title = "Audio Prompt"
            lines = self._wrap_text(prompt_text, body_font, max_w)

            title_w, title_h = title_font.size(title)
            body_h = sum(body_font.size(line)[1] + 4 for line in lines)
            box_w = min(max_w, max(title_w, max((body_font.size(l)[0] for l in lines), default=0))) + pad*2
            box_h = title_h + 10 + body_h + pad*2

            box_x = 10
            box_y = SCREEN_H - box_h - 10

            surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            pygame.draw.rect(surf, (20, 16, 28, 220), (0, 0, box_w, box_h), border_radius=10)
            pygame.draw.rect(surf, (180, 120, 255, 255), (0, 0, box_w, box_h), width=2, border_radius=10)
            screen.blit(surf, (box_x, box_y))

            screen.blit(title_font.render(title, True, (240, 230, 255)), (box_x + pad, box_y + pad))
            yy = box_y + pad + title_h + 6
            for line in lines:
                screen.blit(body_font.render(line, True, (230, 230, 240)), (box_x + pad, yy))
                yy += body_font.size(line)[1] + 4
        # Quest overlay
        if show_quest and quest_text:
            max_w = int(SCREEN_W * 0.8)
            pad   = 14
            title_font = pygame.font.SysFont("consolas", 20, bold=True)
            body_font  = pygame.font.SysFont("consolas", 18)
            title = "Quest"
            lines = self._wrap_text(quest_text, body_font, max_w)
        
            title_w, title_h = title_font.size(title)
            body_h = sum(body_font.size(line)[1] + 4 for line in lines)
            box_w = min(max_w, max(title_w, max((body_font.size(l)[0] for l in lines), default=0))) + pad*2
            box_h = title_h + 10 + body_h + pad*2
        
            box_x = 10
            box_y = SCREEN_H - box_h - 10
        
            surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            pygame.draw.rect(surf, (16, 20, 30, 235), (0, 0, box_w, box_h), border_radius=10)
            pygame.draw.rect(surf, (90, 190, 255, 255), (0, 0, box_w, box_h), width=2, border_radius=10)
            screen.blit(surf, (box_x, box_y))
        
            screen.blit(title_font.render(title, True, (220, 240, 255)), (box_x + pad, box_y + pad))
            yy = box_y + pad + title_h + 6
            for line in lines:
                screen.blit(body_font.render(line, True, (225, 235, 245)), (box_x + pad, yy))
                yy += body_font.size(line)[1] + 4


    def _wrap_text(self, text: str, font: pygame.font.Font, max_width: int) -> List[str]:
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
    
    def _load_sprite(self, path: str, target_h: int = 48) -> Optional[pygame.Surface]:
        """Load a sprite with alpha, scale to target height preserving aspect. Returns None if load fails."""
        try:
            img = pygame.image.load(path).convert_alpha()
            w, h = img.get_size()
            if h != target_h:
                scale = target_h / float(h)
                img = pygame.transform.smoothscale(img, (int(w*scale), target_h))
            return img
        except Exception as e:
            print(f"[Sprite] load failed: {path} ({e})")
            return None

    
    def _blit_label(self, screen: pygame.Surface, text: str, x: int, y: int,
                bgcolor=(20,16,28), fg=(240,240,255)):
        surf = self.small.render(text, True, fg)
        pad = 4
        box = pygame.Surface((surf.get_width()+pad*2, surf.get_height()+pad*2), pygame.SRCALPHA)
        pygame.draw.rect(box, (*bgcolor, 210), box.get_rect(), border_radius=6)
        pygame.draw.rect(box, (180,120,255), box.get_rect(), width=1, border_radius=6)
        box.blit(surf, (pad, pad))
        screen.blit(box, (x, y))
    
    def _stroke_zone_edges(self, overlay: pygame.Surface, tiles: List[Tuple[int,int]],
                           cam_x: float, cam_y: float, t: float):
        S = set(tiles)
        base = (170 + int(60*(0.5 + 0.5*math.sin(t*2.0))), 60, 220)
        # draw thick→thin on the SAME overlay
        for width, alpha in ((8,70), (5,110), (2,180)):
            col = (*base, alpha)
            for (tx, ty) in tiles:
                rx = int(tx*TILE_W - cam_x) + 2
                ry = int(ty*TILE_H - cam_y) + 2
                rw = TILE_W - 4; rh = TILE_H - 4
                if (tx, ty-1) not in S: pygame.draw.line(overlay, col, (rx, ry), (rx+rw, ry), width)
                if (tx+1, ty) not in S: pygame.draw.line(overlay, col, (rx+rw, ry), (rx+rw, ry+rh), width)
                if (tx, ty+1) not in S: pygame.draw.line(overlay, col, (rx, ry+rh), (rx+rw, ry+rh), width)
                if (tx-1, ty) not in S: pygame.draw.line(overlay, col, (rx, ry), (rx, ry+rh), width)



# ======= Controller =======
class GameController:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Alien DJ – Overworld")
        self.fullscreen = DEFAULT_FULLSCREEN
        self.screen = self._apply_display_mode()
        self.clock  = pygame.time.Clock()

        self.model = WorldModel()
        self.audio = AudioService(self.model)
        self.view  = GameView(self.model, self.audio)

        self.show_prompt = False
        self.prompt_text = ""
        self.show_panel = False  # toggleable Now Playing (stub – we keep prompt for now)

        # Auto-hide prompt when tile changes (optional)
        self.model.add_tile_changed_listener(lambda oldt, newt: setattr(self, "show_prompt", False))
        pygame.key.set_repeat(250, 30)
        
        self.show_quest = False
        self.quest_text = ""
        
    def _adjacent_pois(self) -> List[POI]:
        px, py = self.model.player.tile_x, self.model.player.tile_y
        out = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                tx, ty = px+dx, py+dy
                if 0 <= tx < MAP_W and 0 <= ty < MAP_H:
                    pid = self.model.map.pois_at.get((tx,ty))
                    if pid:
                        out.append(self.model.map.pois[pid])
        return out
    

    

    def _apply_display_mode(self):
        global SCREEN_W, SCREEN_H
        if self.fullscreen:
            screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
        else:
            screen = pygame.display.set_mode((1200,700), pygame.RESIZABLE)
        SCREEN_W, SCREEN_H = screen.get_size()
        return screen

    def cycle_mood(self):
        order = ["calm","energetic","angry","triumphant","melancholy","playful","brooding","gritty","glittery","funky"]
        zid = self.model.current_zone_id
        z = self.model.map.zones[zid]
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
            self.model.map.zones[self.model.current_zone_id].spec.mood = entered.strip()
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
                    elif e.key == pygame.K_F11 or (e.key == pygame.K_RETURN and (e.mod & pygame.KMOD_ALT)):
                        self.fullscreen = not self.fullscreen
                        self.screen = self._apply_display_mode()
                    elif e.key == pygame.K_n:
                        self.show_panel = not self.show_panel
                    elif e.key == pygame.K_p:
                        # Toggle prompt overlay for the *current audible* source
                        if self.show_prompt:
                            self.show_prompt = False
                        else:
                            kind, aid = self.audio.active_source
                            if kind == "zone":
                                z = self.model.map.zones[aid]
                                self.prompt_text = z.loop.prompt if (z.loop and z.loop.prompt) else "(No prompt yet.)"
                            else:
                                p = self.model.map.pois[aid]
                                self.prompt_text = p.loop.prompt if (p.loop and p.loop.prompt) else f"(No prompt yet for {p.name}.)"
                            self.show_prompt = True
                    elif e.key == pygame.K_F5:
                        import importlib, promptgen as _pg
                        importlib.reload(_pg)
                        print(f"[PromptGen] Reloaded {getattr(_pg, 'SCHEMA_VERSION', '?')}")
                        # Optional: regen current audible with new prompt
                        k, i = self.audio.active_source
                        if k=="zone":
                            self.model.map.zones[i].loop = None
                            self.audio.request_zone(i, priority=(0,0,0), force=True)
                        else:
                            self.model.map.pois[i].loop = None
                            self.audio.request_poi(i, priority=(0,0,0), force=True)
                    elif e.key == pygame.K_f:
                        # Interact: if adjacent to quest giver, pop quest card (create quest if needed)
                        px, py = self.model.player.tile_x, self.model.player.tile_y
                        # find any POI within 1-tile Chebyshev radius that's a quest giver
                        q_pid = None
                        for dx in (-1,0,1):
                            for dy in (-1,0,1):
                                tx, ty = px+dx, py+dy
                                pid = self.model.map.pois_at.get((tx,ty))
                                if not pid: continue
                                poi = self.model.map.pois[pid]
                                if poi.role == "quest_giver":
                                    q_pid = pid
                                    break
                            if q_pid: break
                    
                        if q_pid:
                            # if we don't already have a quest, pick a target NPC anywhere (not Boss Skuggs)
                            if self.model.active_quest is None:
                                # pick a target npc
                                npcs = [p for p in self.model.map.pois.values() if p.kind=="npc" and p.name != "Boss Skuggs"]
                                if npcs:
                                    target = random.choice(npcs)
                                    tz = self.model.map.zones[target.zone_id]
                                    self.model.active_quest = Quest(
                                        giver_pid=q_pid,
                                        target_pid=target.pid,
                                        target_name=target.name,
                                        target_tile=target.tile,
                                        target_zone=target.zone_id,
                                        target_zone_name=tz.spec.name,
                                        accepted=True
                                    )
                            # show card (existing or just-created)
                            if self.model.active_quest:
                                q = self.model.active_quest
                                tx, ty = q.target_tile
                                self.quest_text = (
                                    f"Find the alien '{q.target_name}'.\n"
                                    f"Location: {q.target_zone_name} at tile ({tx}, {ty}).\n\n"
                                    f"Tip: stand next to them to hear their tune."
                                )
                                self.show_quest = True
                        else:
                            # Optional: click F again to dismiss the quest box if it's open
                            if self.show_quest:
                                self.show_quest = False

                    elif e.key == pygame.K_g:
                        # Regenerate current zone
                        zid = self.model.current_zone_id
                        self.model.map.zones[zid].loop = None
                        self.audio.request_zone(zid, priority=(0,0,0), force=True)
                    elif e.key == pygame.K_m:
                        self.cycle_mood()
                        self.audio.request_zone(self.model.current_zone_id, priority=(0,0,0), force=True)
                    elif e.key == pygame.K_e:
                        self.edit_mood_text()
                        self.audio.request_zone(self.model.current_zone_id, priority=(0,0,0), force=True)
                    elif e.key == pygame.K_r:
                        if self.audio.recorder.is_recording():
                            self.audio.recorder.stop(); self.audio.record_armed = False
                            print("[REC] stopped.")
                        else:
                            self.audio.record_armed = True
                            print("[REC] armed: will start at next loop boundary.")
                    elif e.key == pygame.K_i:
                        print_inventory("./inventory")

            keys = pygame.key.get_pressed()
            dx = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
            dy = (keys[pygame.K_DOWN]  or keys[pygame.K_s]) - (keys[pygame.K_UP]   or keys[pygame.K_w])
            if dx or dy:
                self.model.move_player(dx*self.model.player.speed, dy*self.model.player.speed)
                # Quest auto-complete if adjacent to target
                # --- quest completion check ---
                q = self.model.active_quest
                if q:
                    px, py = self.model.player.tile_x, self.model.player.tile_y
                    tx, ty = q.target_tile
                    if max(abs(px - tx), abs(py - ty)) <= 1:
                        # Completed!
                        self.quest_text = f"Quest complete! You found {q.target_name} in {q.target_zone_name}."
                        self.show_quest = True  # reuse the quest modal to announce completion
                        self.model.active_quest = None
                        self.model.quest_completed = True


            self.view.draw(self.screen,
               self.audio.record_armed,
               self.audio.recorder.is_recording(),
               self.show_prompt, self.prompt_text,
               self.show_quest, self.quest_text)

            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()

# ======= utils =======
def print_inventory(inv_dir: str):
    try:
        files = sorted(f for f in os.listdir(inv_dir) if f.lower().endswith(".wav"))
    except FileNotFoundError:
        files = []
    if not files:
        print("[INV] (empty)"); return
    print("[INV] Recorded clips:")
    for f in files: print("  -", f)

# ======= entry =======
if __name__ == "__main__":
    try:
        GameController().run()
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
