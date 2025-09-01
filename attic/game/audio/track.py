from typing import Dict, Tuple, Optional
import pygame
import numpy as np
from ..core.config import SR, MIN_CLIP_SEC
from ..core.utils import seconds_to_samples

class Track:
    def __init__(self, name: str, array: np.ndarray):
        self.name = name
        self.array = array  # int16 stereo (n,2)
        self.sound = pygame.mixer.Sound(buffer=array)
        self.playing = False
        self.channel: Optional[pygame.mixer.Channel] = None
        self.loop_started_ms: int = 0
        self._wave_cache: Dict[Tuple[int, int, float, float], pygame.Surface] = {}
        # selection in normalized [0,1]
        self.sel_start = 0.0
        self.sel_end = min(1.0, max(MIN_CLIP_SEC / max(1e-6, self.duration), 2.0 / max(1e-6, self.duration)))

    @property
    def duration(self) -> float:
        return self.array.shape[0] / SR

    def play(self):
        if not self.playing:
            ch = self.sound.play(loops=-1)
            if ch:
                self.channel = ch
                self.loop_started_ms = pygame.time.get_ticks()
                self.playing = True

    def stop(self):
        if self.playing:
            try:
                if self.channel: self.channel.stop()
                else: self.sound.fadeout(120)
            except Exception:
                pass
            self.playing = False
            self.channel = None

    def make_slice_sound(self, start_t: float, end_t: float) -> pygame.mixer.Sound:
        s0 = seconds_to_samples(start_t, SR)
        s1 = seconds_to_samples(end_t, SR)
        s0 = max(0, min(self.array.shape[0]-1, s0))
        s1 = max(s0+1, min(self.array.shape[0], s1))
        seg = self.array[s0:s1, :]
        try:
            return pygame.mixer.Sound(buffer=seg)
        except Exception:
            return pygame.mixer.Sound(buffer=seg.copy())

    def wave_surface(self, w: int, h: int, win_start: float, win_end: float) -> pygame.Surface:
        key = (w, h, round(win_start,4), round(win_end,4))
        if key in self._wave_cache:
            return self._wave_cache[key]
        # mono envelope for visible window
        n_total = self.array.shape[0]
        i0 = int(max(0.0, min(1.0, win_start)) * n_total)
        i1 = int(max(0.0, min(1.0, win_end)) * n_total)
        i0, i1 = min(i0, i1), max(i0, i1)
        if i1 - i0 < 1: i1 = min(n_total, i0 + 1)
        arr = self.array[i0:i1, :]
        mono = arr.mean(axis=1).astype(np.float32) / 32768.0
        n = mono.shape[0]
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        step = max(1, n // max(1, w))
        mid = h // 2
        for x in range(w):
            j0 = x * step
            j1 = min(n, j0 + step)
            seg = mono[j0:j1]
            if seg.size == 0: continue
            lo = float(np.min(seg)); hi = float(np.max(seg))
            y0 = int(mid - hi * (h * 0.48))
            y1 = int(mid - lo * (h * 0.48))
            pygame.draw.line(surf, (180, 210, 255), (x, y0), (x, y1))
        self._wave_cache[key] = surf
        return surf
