import os, math, uuid, time, tempfile, secrets
import random
import threading
import numpy as np
import soundfile as sf
from typing import Optional, Tuple, Dict, List, Any

from cosmic_underground.core import config as C
from cosmic_underground.core.models import GeneratedLoop, ZoneSpec
from cosmic_underground.core.logger import logger
import promptgen


class LocalStableAudioProvider:
    def __init__(self, context_fn=None):
        try:
            import torch
            self._torch = torch
        except Exception as e:
            raise RuntimeError(
                "Stable Audio backend unavailable. In your venv:\n"
                "  pip install --no-deps stable-audio-tools==0.0.19\n"
                "  pip install numpy einops soundfile huggingface_hub pygame\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu121  (or cpu)"
            ) from e

        self._torch = torch
        from stable_audio_tools import get_pretrained_model
        self._local = threading.local()
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        self._torch.set_num_threads(max(1, os.cpu_count() // 2))
        self.device = (
            "cuda" if self._torch.cuda.is_available() else
            ("mps" if getattr(self._torch.backends, "mps", None)
                     and self._torch.backends.mps.is_available() else "cpu")
        )
        
        print(f"[StableLocal] Loading model… (device={self.device})", flush=True)
        self.model, self.cfg = get_pretrained_model("stabilityai/stable-audio-open-small")
        self.model = self.model.to(self.device).eval()
        self.sr_model = int(self.cfg["sample_rate"])

        self.tmpdir = tempfile.mkdtemp(prefix="alien_dj_local_")
        self.context_fn = (context_fn if callable(context_fn) else (lambda: {}))

    @staticmethod
    def _duration_for(bpm: int, bars: int, timesig: Tuple[int,int]) -> float:
        beats = bars * (timesig[0] if timesig and len(timesig) else 4)
        return beats * (60.0 / max(1, bpm))

    def _get_thread_local_model(self):
        """Each worker thread gets its own model instance."""
        # The legacy provider used a single, shared model instance.
        # This is generally not thread-safe for inference unless the model itself is stateless
        # and the library handles internal locking, which stable-audio-tools seems to do.
        return self.model, self.cfg, self.sr_model

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
        
        model, cfg, sr_model = self._get_thread_local_model()

        seconds_total = self._duration_for(bpm, bars, timesig)
        sample_size   = int(round(seconds_total * self.sr_model))
        seed = secrets.randbits(31)
        log_msg = f"tid={threading.get_ident()} seed={seed} prompt='{prompt}'"
        logger.log(f"REQUEST: {log_msg}")

        t0 = time.time()
        with self._torch.inference_mode():
            audio = generate_diffusion_cond(
                model,
                steps=8, cfg_scale=1.0, sampler_type="pingpong",
                conditioning=[{"prompt": prompt, "seconds_total": seconds_total}],
                sample_size=sample_size, device=self.device, seed=seed,
            )
        # (B, D, T) -> (T, D)
        audio = audio.squeeze(0).to(self._torch.float32).cpu().transpose(0,1).numpy().copy()
        peak = float(np.max(np.abs(audio))) or 1.0
        audio = (audio / peak) * 0.98
        audio = self._ensure_length(audio, self.sr_model, seconds_total)
        audio, out_sr = self._resample_linear(audio, self.sr_model, C.ENGINE_SR)
        duration_actual = float(audio.shape[0]) / float(out_sr)

        out_path = os.path.join(self.tmpdir, f"{uuid.uuid4().hex}.wav")
        sf.write(out_path, audio, out_sr, subtype="PCM_16")
        gen_time = time.time() - t0
        logger.log(f"SUCCESS: tid={threading.get_ident()} time={gen_time:.2f}s file='{os.path.basename(out_path)}'")
        return GeneratedLoop(
            out_path, duration_actual, bpm,
            getattr(spec_like, "key_mode", "Unknown"),
            prompt,
            {"backend":"stable_local","seed":seed,"sr":out_sr, **meta}
        )