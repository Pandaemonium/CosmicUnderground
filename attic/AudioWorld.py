#!/usr/bin/env python3
"""
Alien DJ Bootlegger – tiny pygame prototype with on-the-fly audio generation.

Features:
- 8 zones on a 2x4 grid; each has an NPC and its own prompt/mood/BPM/key.
- Entering a zone starts/continues that zone's loop.
- Press 'G' to (re)generate a loop for the current zone (cloud or dummy backend).
- Press 'M' to cycle mood tags, 'E' to type custom mood text.
- Press 'R' to arm recording: it begins at the next loop boundary (bar 1).
  Press 'R' again to stop manually, or let it auto-stop at end-of-loop.
- Press 'I' to list inventory (recorded clips saved to ./inventory).

Audio backends:
- Replicate (Stable Audio Open): set AUDIO_BACKEND=replicate,
  REPLICATE_API_TOKEN, REPLICATE_MODEL_VERSION.
- Dummy: no network; produces synthetic loop WAVs for testing.

This is a *prototype*: one loop plays at a time (no mixing), recording copies the
currently playing loop file "live" from the boundary for realism without tapping the OS mixer.
"""

import os, sys, math, time, json, threading, queue, uuid, tempfile, wave, struct
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import pygame
import requests
import hashlib
import numpy as np

# ------------------------------ Configuration --------------------------------

SCREEN_W, SCREEN_H = 1200, 700
GRID_COLS, GRID_ROWS = 4, 2
FPS = 60

DEFAULT_BARS = 8           # bars per loop (4/4)
DEFAULT_TIMESIG = (4, 4)
SAMPLE_RATE = 44100
CHANNELS = 2
BITS = 16

# --- Zones: base prompts (edit freely) ---
DEFAULT_ZONES = [
    # name, bpm, key_mode, base scene/ensemble, mood words
    ("Neon Bazaar",      120, "G Mixolydian",  "glittering synth arps, clacky percussion, funky market", "energetic"),
    ("Moss Docks",        96, "F Phrygian",    "dubby bass, foghorn pads, cavernous reverb",             "brooding"),
    ("Crystal Canyon",   140, "C Lydian",      "glassy mallets, airy choirs, shimmering echoes",         "triumphant"),
    ("Scrapyard Funk",   108, "E minor",       "breakbeat kit, clavinet riffs, grimey amp",              "gritty"),
    ("Fungal Grove",     100, "D major",       "hand percussion, kalimba, wooden clicks",                "playful"),
    ("Neon Rain",         92, "F# Aeolian",    "trip-hop, vinyl crackle, moody Rhodes",                  "melancholy"),
    ("Hover Bazaar",     132, "B Dorian",      "UK garage shuffle, vocal chops, subby bass",             "energetic"),
    ("Slime Parade",     120, "D minor",       "big-band brass, clav, wah guitar, parade snare",         "happy"),
]

# ------------------------------ Utility Types --------------------------------

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

# ------------------------------ Audio Providers ------------------------------

class AudioProvider:
    def generate(self, zone: ZoneSpec) -> GeneratedLoop:
        raise NotImplementedError

    @staticmethod
    def compute_duration_sec(bars: int, bpm: float, timesig=(4,4)) -> float:
        # bars * beats_per_bar * seconds_per_beat
        beats = bars * timesig[0]
        return beats * (60.0 / bpm)

class DummyProvider(AudioProvider):
    """Generates a synthetic loop (sine + noise) into a WAV file for testing."""
    def __init__(self):
        self.tmpdir = tempfile.mkdtemp(prefix="alien_dj_dummy_")

    def generate(self, zone: ZoneSpec) -> GeneratedLoop:
        dur = self.compute_duration_sec(zone.bars, zone.bpm, zone.timesig)
        path = os.path.join(self.tmpdir, f"{uuid.uuid4().hex}.wav")
        freq_left  = 220.0 + (hash(zone.name) % 7) * 30 + (hash(zone.mood) % 20)
        freq_right = 330.0 + (hash(zone.scene) % 5) * 35
        amp = 0.2
        with wave.open(path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            total = int(dur * SAMPLE_RATE)
            for n in range(total):
                t = n / SAMPLE_RATE
                # simple mood tilt
                mood_bias = {"calm":0.0, "energetic":0.5, "angry":0.8, "triumphant":0.6}.get(zone.mood.lower(), 0.3)
                sL = amp * math.sin(2*math.pi*freq_left*t) * (0.7 + 0.3*math.sin(2*math.pi*zone.bpm/60.0*t))
                sR = amp * math.sin(2*math.pi*freq_right*t) * (0.7 + 0.3*math.sin(2*math.pi*zone.bpm/60.0*t + mood_bias))
                # mild gate at bar boundaries to make downbeats clear
                beat = (t * zone.bpm / 60.0)
                gate = 0.3 + 0.7*(1.0 if (beat % 1.0) < 0.2 else 0.6)
                sL *= gate; sR *= gate
                # clip & write
                sL_i = max(-1.0, min(1.0, sL))
                sR_i = max(-1.0, min(1.0, sR))
                wf.writeframes(struct.pack("<hh", int(sL_i*32767), int(sR_i*32767)))
        prompt = zone.prompt_override or f"{zone.mood} {zone.scene}, loopable {zone.bars} bars, {zone.bpm} BPM, {zone.key_mode}"
        return GeneratedLoop(path, dur, zone.bpm, zone.key_mode, prompt, provider_meta={"backend": "dummy"})

class ReplicateProvider(AudioProvider):
    """
    Calls Replicate's REST API for the 'stackadoc/stable-audio-open-1.0' model.
    Requires:
      - REPLICATE_API_TOKEN
      - REPLICATE_MODEL_VERSION (from the model's 'Versions' page)
    Returns a downloaded WAV file path.
    """
    def __init__(self):
        self.token = os.environ.get("REPLICATE_API_TOKEN")
        self.version = os.environ.get("REPLICATE_MODEL_VERSION")  # e.g., "9aff84a6..." (update from Versions page)
        if not self.token or not self.version:
            raise RuntimeError("ReplicateProvider requires REPLICATE_API_TOKEN and REPLICATE_MODEL_VERSION env vars.")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/json",
        })
        self.tmpdir = tempfile.mkdtemp(prefix="alien_dj_rep_")

    def generate(self, zone: ZoneSpec) -> GeneratedLoop:
        dur = self.compute_duration_sec(zone.bars, zone.bpm, zone.timesig)
        prompt = zone.prompt_override or (
            f"{zone.mood} alien scene: {zone.scene}. "
            f"Groove-focused, clean downbeats, loopable {zone.bars} bars. "
            f"{zone.bpm} BPM, {zone.key_mode}. Minimal silence at start/end."
        )
        # Build request payload for Replicate
        payload = {
            "version": self.version,
            "input": {
                # The exact input keys depend on the model’s Cog; these commonly work:
                "description": prompt,
                "duration": round(dur),   # seconds
                # Some variants accept "seed", "top_k", "cfg_scale" etc.; add as needed.
            }
        }
        resp = self.session.post("https://api.replicate.com/v1/predictions", data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        prediction = resp.json()
        # Poll for completion
        status = prediction.get("status")
        get_url = prediction["urls"]["get"]
        while status in ("starting", "processing"):
            time.sleep(1.0)
            pr = self.session.get(get_url, timeout=30).json()
            status = pr.get("status")
            prediction = pr
        if status != "succeeded":
            raise RuntimeError(f"Replicate generation failed: {status} {prediction}")
        # Output: usually list of URLs; pick first WAV
        outputs = prediction.get("output", [])
        if not outputs:
            raise RuntimeError("No output from Replicate.")
        # Heuristic: first item is WAV URL
        wav_url = outputs[0] if isinstance(outputs[0], str) else outputs[0].get("audio")
        if not isinstance(wav_url, str):
            raise RuntimeError("Unexpected output format from Replicate.")
        # Download
        r = requests.get(wav_url, stream=True, timeout=120)
        r.raise_for_status()
        out_path = os.path.join(self.tmpdir, f"{uuid.uuid4().hex}.wav")
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024 * 512):
                if chunk:
                    f.write(chunk)
        return GeneratedLoop(out_path, dur, zone.bpm, zone.key_mode, prompt, provider_meta={"backend": "replicate", "prediction_id": prediction.get("id")})

class StabilityProvider(AudioProvider):
    """
    Placeholder for Stability Platform 'Stable Audio (Small/Open/2.0)' REST.
    Fill in your account’s endpoint/parameters and auth.

    Typical shape (pseudo):
      POST https://api.stability.ai/v2beta/audio/generate
      Headers: Authorization: Bearer <STABILITY_API_KEY>
      JSON: { "prompt": "...", "duration": 10, ... }
    See your Stability dashboard/docs for the exact path & fields.
    """
    def __init__(self):
        self.api_key = os.environ.get("STABILITY_API_KEY")
        if not self.api_key:
            raise RuntimeError("StabilityProvider requires STABILITY_API_KEY.")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self.tmpdir = tempfile.mkdtemp(prefix="alien_dj_stab_")

    def generate(self, zone: ZoneSpec) -> GeneratedLoop:
        dur = self.compute_duration_sec(zone.bars, zone.bpm, zone.timesig)
        prompt = zone.prompt_override or (
            f"{zone.mood} alien world: {zone.scene}. loopable {zone.bars} bars, {zone.bpm} BPM, {zone.key_mode}."
        )
        # TODO: Replace with your actual endpoint + params.
        raise NotImplementedError("Fill StabilityProvider.generate() with your Stable Audio Small endpoint/params.")



class LocalStableAudioProvider(AudioProvider):
    """
    Local Stable Audio Small/Open via 'stable_audio_tools'.
    Creates exact-length, loop-friendly WAVs for each zone request.
    Select with: AUDIO_BACKEND=stable_local
    Optional: HUGGINGFACE_HUB_TOKEN in env if you need HF login.
    """
    def __init__(self):
        # Lazy-import heavy deps so other backends don’t pay the cost
        import torch
        from stable_audio_tools import get_pretrained_model

        self.torch = torch
        self.device = self._pick_device()
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        self.torch.set_num_threads(max(1, os.cpu_count() // 2))

        # Load once; keep in eval.
        # You can swap to "stabilityai/stable-audio-open-small" or your exact repo
        print("[LocalStableAudio] Loading model…")
        self.model, self.cfg = get_pretrained_model("stabilityai/stable-audio-open-small")
        self.model = self.model.to(self.device).eval()
        self.sr_model = int(self.cfg["sample_rate"])
        print(f"[LocalStableAudio] Ready. sr={self.sr_model}")

        self.tmpdir = tempfile.mkdtemp(prefix="alien_dj_local_")

    def _pick_device(self):
        import torch
        # Use CUDA if available, else CPU (your script hard-coded 'cuda' with a CPU comment)
        if torch.cuda.is_available():
            return "cuda"
        try:
            # Apple MPS if you’re on a Mac w/ Metal
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _stable_hash_seed(self, text: str) -> int:
        # Reproducible 31-bit seed from text (zone+prompt); avoids full randomness when you want determinism
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(h[:8], 16) & ((1 << 31) - 1)

    def _build_prompt(self, zone: ZoneSpec) -> str:
        if zone.prompt_override:
            return zone.prompt_override
        # Keep it short & directive for diffusion: “loopable N bars”, “minimal silence”, etc.
        return (f"{zone.mood} alien scene: {zone.scene}. "
                f"Groove-forward, clean downbeats, loopable {zone.bars} bars. "
                f"{zone.bpm} BPM, {zone.key_mode}. Minimal silence at start and end.")

    def _ensure_length(self, wav_np: np.ndarray, sr: int, seconds_total: float) -> np.ndarray:
        """Crop/tile to exact length in samples, then micro-fade edges."""
        target = int(round(seconds_total * sr))
        T = wav_np.shape[0]
        if T < target:
            reps = int(math.ceil(target / T))
            wav_np = np.tile(wav_np, (reps, 1))[:target]
        elif T > target:
            wav_np = wav_np[:target]

        # 3 ms micro-fades (equal-power-ish)
        fade = max(2, int(sr * 0.003))
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wav_np[:fade] *= ramp[:, None]
        wav_np[-fade:] *= ramp[::-1][:, None]
        return wav_np

    def _maybe_resample(self, wav_np: np.ndarray, sr_in: int, sr_out: int) -> Tuple[np.ndarray, int]:
        if sr_in == sr_out:
            return wav_np, sr_in
        # Lightweight linear resample (good enough for prototyping)
        dur = wav_np.shape[0] / sr_in
        tgt_len = int(round(dur * sr_out))
        t_in = np.linspace(0.0, dur, wav_np.shape[0], endpoint=False, dtype=np.float64)
        t_out = np.linspace(0.0, dur, tgt_len, endpoint=False, dtype=np.float64)
        out = np.empty((tgt_len, wav_np.shape[1]), dtype=np.float32)
        for ch in range(wav_np.shape[1]):
            out[:, ch] = np.interp(t_out, t_in, wav_np[:, ch]).astype(np.float32)
        return out, sr_out

    def generate(self, zone: ZoneSpec) -> GeneratedLoop:
        import torch
        from einops import rearrange
        from stable_audio_tools.inference.generation import generate_diffusion_cond
        import soundfile as sf

        beats_per_bar = zone.timesig[0]
        seconds_total = zone.bars * beats_per_bar * (60.0 / float(zone.bpm))

        prompt = self._build_prompt(zone)
        # Use deterministic seed per (zone, mood, bpm, key) so re-entry is stable; press G to change mood/regenerate
        seed_text = f"{zone.name}|{zone.mood}|{zone.bpm}|{zone.key_mode}|{zone.scene}"
        seed = self._stable_hash_seed(seed_text)

        # **Important**: Set sample_size to match desired seconds_total
        sample_size = int(round(seconds_total * self.sr_model))

        t0 = time.time()
        with torch.inference_mode():
            audio = generate_diffusion_cond(
                self.model,
                steps=8,
                cfg_scale=1.0,
                sampler_type="pingpong",
                conditioning=[{"prompt": prompt, "seconds_total": seconds_total}],
                sample_size=sample_size,
                device=self.device,
                seed=seed
            )

        # -> (T, C) float32 in [-1, 1]
        audio = rearrange(audio, "b d t -> t d").to(torch.float32).cpu().numpy()
        # normalize conservatively
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-6:
            audio = (audio / peak) * 0.98

        # length fix + micro-fades
        audio = self._ensure_length(audio, self.sr_model, seconds_total)

        # resample to engine SR if needed (pygame set to 44100)
        target_sr = SAMPLE_RATE
        audio, out_sr = self._maybe_resample(audio, self.sr_model, target_sr)

        out_path = os.path.join(self.tmpdir, f"{uuid.uuid4().hex}.wav")
        sf.write(out_path, audio, out_sr, subtype="PCM_16")

        print(f"[LocalStableAudio] gen {zone.name} in {time.time()-t0:.2f}s → {os.path.basename(out_path)}")
        return GeneratedLoop(
            wav_path=out_path,
            duration_sec=seconds_total,   # authoritative loop length for boundary timer
            bpm=zone.bpm,
            key_mode=zone.key_mode,
            prompt=prompt,
            provider_meta={"backend": "stable_local", "seed": seed, "sr": out_sr}
        )


def make_provider() -> AudioProvider:
    backend = os.environ.get("AUDIO_BACKEND", "dummy").lower()
    if backend == "replicate":
        return ReplicateProvider()
    elif backend in ("stability", "stable_local", "local", "stability_local"):
        return LocalStableAudioProvider()
    else:
        return DummyProvider()

# ------------------------------ Game State -----------------------------------

@dataclass
class ZoneRuntime:
    spec: ZoneSpec
    loop: Optional[GeneratedLoop] = None
    generating: bool = False
    last_error: Optional[str] = None

class Recorder:
    """Copies the current zone's loop WAV to inventory starting at boundary; can stop early."""
    def __init__(self):
        self.active_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.inventory_dir = os.path.abspath("./inventory")
        os.makedirs(self.inventory_dir, exist_ok=True)
        self.recording_path: Optional[str] = None

    def is_recording(self) -> bool:
        return self.active_thread is not None and self.active_thread.is_alive()

    def start(self, loop_path: str, max_seconds: float, meta: Dict):
        if self.is_recording():
            return
        self.stop_flag.clear()
        out_name = f"rec_{int(time.time())}_{meta.get('zone','zone')}_{uuid.uuid4().hex[:6]}.wav"
        out_path = os.path.join(self.inventory_dir, out_name)
        self.recording_path = out_path

        def _run_copy():
            try:
                with wave.open(loop_path, "rb") as src, wave.open(out_path, "wb") as dst:
                    dst.setnchannels(src.getnchannels())
                    dst.setsampwidth(src.getsampwidth())
                    dst.setframerate(src.getframerate())
                    frames_total = int(max_seconds * src.getframerate())
                    chunk = 2048
                    written = 0
                    while written < frames_total and not self.stop_flag.is_set():
                        to_read = min(chunk, frames_total - written)
                        data = src.readframes(to_read)
                        if not data:
                            break
                        dst.writeframes(data)
                        written += to_read
                print(f"[REC] Saved: {out_path}")
            except Exception as e:
                print(f"[REC] Error: {e}")

        t = threading.Thread(target=_run_copy, daemon=True)
        t.start()
        self.active_thread = t

    def stop(self):
        self.stop_flag.set()
        if self.active_thread:
            self.active_thread.join(timeout=2.0)
        self.active_thread = None
        return self.recording_path

class AudioPlayer:
    """Controls pygame.mixer.music for seamless looping, boundary notifications."""
    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=1024)
        self.current_path: Optional[str] = None
        self.loop_len_ms: int = 0
        self.next_boundary_ms: int = 0
        self.loop_started_ticks: int = 0
        self.boundary_event = pygame.USEREVENT + 1

    def play_loop(self, wav_path: str, duration_sec: float):
        if self.current_path == wav_path and pygame.mixer.music.get_busy():
            # Already playing; just update loop length/timing.
            self.loop_len_ms = max(1, int(duration_sec * 1000))
            return
        pygame.mixer.music.stop()
        pygame.mixer.music.load(wav_path)
        pygame.mixer.music.play(loops=-1)  # seamless loop
        self.current_path = wav_path
        self.loop_len_ms = max(1, int(duration_sec * 1000))
        self.loop_started_ticks = pygame.time.get_ticks()
        self.next_boundary_ms = self.loop_len_ms
        pygame.time.set_timer(self.boundary_event, self.loop_len_ms)  # periodic boundary

    def stop(self):
        pygame.mixer.music.stop()
        pygame.time.set_timer(self.boundary_event, 0)

# ------------------------------ Main App -------------------------------------

class AlienDJApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Alien DJ Bootlegger – Prototype")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

        self.provider = make_provider()
        self.player = AudioPlayer()
        self.recorder = Recorder()

        # Build zones
        self.zones: list[ZoneRuntime] = []
        for name, bpm, key_mode, scene, mood in DEFAULT_ZONES:
            self.zones.append(ZoneRuntime(ZoneSpec(name, bpm, key_mode, scene, mood)))

        self.player_pos = [SCREEN_W // 2, SCREEN_H // 2]
        self.speed = 6
        self.col_w = SCREEN_W // GRID_COLS
        self.row_h = SCREEN_H // GRID_ROWS
        self.current_index = self._zone_index_from_pos(*self.player_pos)

        # Recording FSM
        self.record_armed = False
        self.auto_stop_at_end = True  # stop at next boundary if not manually stopped

        # Async gen queue
        self.gen_queue = queue.Queue()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

        # Ensure something is playing at start
        self._ensure_zone_loop(self.current_index, force_generate=True)

    # -------------------------- Worker for generation -------------------------

    def _worker_loop(self):
        while True:
            idx = self.gen_queue.get()
            if idx is None:
                return
            zr = self.zones[idx]
            if zr.generating:
                continue
            zr.generating = True; zr.last_error = None
            try:
                loop = self.provider.generate(zr.spec)
                zr.loop = loop
                print(f"[GEN] {zr.spec.name} -> {os.path.basename(loop.wav_path)} ({loop.duration_sec:.2f}s)")
                # If this zone is active, (re)play it
                if idx == self.current_index:
                    self.player.play_loop(loop.wav_path, loop.duration_sec)
            except Exception as e:
                zr.last_error = str(e)
                print(f"[GEN][ERR] {zr.spec.name}: {e}")
            finally:
                zr.generating = False

    def _ensure_zone_loop(self, idx: int, force_generate=False):
        zr = self.zones[idx]
        if force_generate or zr.loop is None:
            self.gen_queue.put(idx)
        elif zr.loop:
            self.player.play_loop(zr.loop.wav_path, zr.loop.duration_sec)

    # ----------------------------- Helpers ------------------------------------

    def _zone_index_from_pos(self, x, y) -> int:
        col = min(GRID_COLS-1, max(0, x // self.col_w))
        row = min(GRID_ROWS-1, max(0, y // self.row_h))
        return int(row * GRID_COLS + col)

    def _cycle_mood(self, idx: int):
        order = ["calm", "energetic", "angry", "triumphant"]
        cur = self.zones[idx].spec.mood.lower()
        nxt = order[(order.index(cur) + 1) % len(order)] if cur in order else "calm"
        self.zones[idx].spec.mood = nxt

    def _edit_mood_text(self, idx: int):
        pygame.key.set_repeat(0)
        entered = ""
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        done = True
                    elif event.key == pygame.K_BACKSPACE:
                        entered = entered[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        done = True
                    else:
                        ch = event.unicode
                        if ch and 32 <= ord(ch) < 127:
                            entered += ch
            # draw prompt
            self.screen.fill((10,10,15))
            txt1 = self.font.render("Type mood/descriptor and press Enter (Esc to cancel):", True, (240,240,240))
            txt2 = self.font.render(entered, True, (180,255,180))
            self.screen.blit(txt1, (40, SCREEN_H//2 - 30))
            self.screen.blit(txt2, (40, SCREEN_H//2 + 10))
            pygame.display.flip()
            self.clock.tick(30)
        if entered.strip():
            self.zones[idx].spec.mood = entered.strip()
        pygame.key.set_repeat(250, 30)

    # ----------------------------- Main Loop ----------------------------------

    def run(self):
        pygame.key.set_repeat(250, 30)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == self.player.boundary_event:
                    # Loop boundary has occurred
                    if self.record_armed and not self.recorder.is_recording():
                        # Start recording for exactly one loop (or until manual stop)
                        z = self.zones[self.current_index]
                        if z.loop:
                            self.recorder.start(z.loop.wav_path, z.loop.duration_sec, {
                                "zone": z.spec.name, "bpm": z.spec.bpm, "key": z.spec.key_mode, "mood": z.spec.mood
                            })
                            self.record_armed = False
                    elif self.recorder.is_recording() and self.auto_stop_at_end:
                        # stop automatically at boundary if not manually stopped
                        self.recorder.stop()

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_g:
                        self._ensure_zone_loop(self.current_index, force_generate=True)
                    elif event.key == pygame.K_m:
                        self._cycle_mood(self.current_index)
                    elif event.key == pygame.K_e:
                        self._edit_mood_text(self.current_index)
                    elif event.key == pygame.K_r:
                        if self.recorder.is_recording():
                            self.recorder.stop()
                            self.record_armed = False
                            print("[REC] Manually stopped.")
                        else:
                            self.record_armed = True
                            print("[REC] Armed: will start at next loop boundary.")
                    elif event.key == pygame.K_i:
                        print_inventory("./inventory")

            # movement
            keys = pygame.key.get_pressed()
            dx = dy = 0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]: dx -= self.speed
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: dx += self.speed
            if keys[pygame.K_UP] or keys[pygame.K_w]: dy -= self.speed
            if keys[pygame.K_DOWN] or keys[pygame.K_s]: dy += self.speed
            self.player_pos[0] = int(max(0, min(SCREEN_W-1, self.player_pos[0] + dx)))
            self.player_pos[1] = int(max(0, min(SCREEN_H-1, self.player_pos[1] + dy)))

            # zone switching
            idx = self._zone_index_from_pos(*self.player_pos)
            if idx != self.current_index:
                self.current_index = idx
                self._ensure_zone_loop(idx, force_generate=True)  # generate a fresh loop on entry

                # Disarm recording when switching zones (design choice)
                self.record_armed = False
                if self.recorder.is_recording():
                    self.recorder.stop()

            self._draw()
            self.clock.tick(FPS)

        pygame.quit()

    # ------------------------------ Rendering ---------------------------------

    def _draw(self):
        self.screen.fill((14, 10, 18))
        # grid
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                idx = r*GRID_COLS + c
                x = c*self.col_w; y = r*self.row_h
                rect = pygame.Rect(x+2, y+2, self.col_w-4, self.row_h-4)
                active = (idx == self.current_index)
                col = (40, 42, 70) if not active else (70, 88, 140)
                pygame.draw.rect(self.screen, col, rect, border_radius=12)
                # NPC dot
                pygame.draw.circle(self.screen, (220, 200, 140), (x + self.col_w//2, y + self.row_h//2), 16)
                # labels
                zr = self.zones[idx]
                name = self.font.render(zr.spec.name, True, (240,240,240))
                sub = self.font.render(f"{zr.spec.mood} | {zr.spec.bpm} BPM | {zr.spec.key_mode}", True, (200,220,255))
                self.screen.blit(name, (x+10, y+10))
                self.screen.blit(sub, (x+10, y+34))
                # status
                status = "ready"
                if zr.generating:
                    status = "generating..."
                elif zr.last_error:
                    status = f"error: {zr.last_error[:26]}..."
                elif zr.loop is None:
                    status = "needs generate (G)"
                else:
                    status = os.path.basename(zr.loop.wav_path)
                st = self.font.render(status, True, (180,180,180))
                self.screen.blit(st, (x+10, y + self.row_h - 28))

        # player
        pygame.draw.circle(self.screen, (255, 100, 120), self.player_pos, 10)

        # HUD
        hud_lines = [
            "Arrows/WASD move | G generate | M cycle mood | E edit mood | R arm/stop record | I inventory | Esc quit",
        ]
        if self.record_armed:
            hud_lines.append("REC ARMED: recording will start at the next loop boundary.")
        if self.recorder.is_recording():
            hud_lines.append("RECORDING... press R to stop manually (or it will stop at end-of-loop).")
        y = SCREEN_H - 22*len(hud_lines) - 8
        for hl in hud_lines:
            self.screen.blit(self.font.render(hl, True, (240, 240, 240)), (10, y))
            y += 22

        pygame.display.flip()

def print_inventory(inv_dir: str):
    try:
        files = sorted(os.listdir(inv_dir))
    except FileNotFoundError:
        files = []
    if not files:
        print("[INV] (empty)")
        return
    print("[INV] Recorded clips:")
    for f in files:
        if f.lower().endswith(".wav"):
            print("  -", f)

# ------------------------------ Entrypoint -----------------------------------

if __name__ == "__main__":
    try:
        app = AlienDJApp()
        app.run()
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
