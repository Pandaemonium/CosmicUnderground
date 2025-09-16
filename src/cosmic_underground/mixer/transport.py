from __future__ import annotations
import time
from dataclasses import dataclass

@dataclass
class Transport:
    bpm: float = 120.0
    timesig_n: int = 4
    timesig_d: int = 4
    playing: bool = False
    loop_enabled: bool = False
    loop_start_bar: float = 0.0
    loop_end_bar: float = 8.0

    # runtime
    _pos_beats: float = 0.0
    _last_monotonic: float = 0.0

    def set_from_project(self, bpm: float, timesig: tuple[int,int], loop_region):
        self.bpm = float(bpm)
        self.timesig_n, self.timesig_d = timesig
        if loop_region:
            self.loop_start_bar, self.loop_end_bar = loop_region

    @property
    def pos_beats(self) -> float:
        return self._pos_beats

    @property
    def pos_bars(self) -> float:
        return self._pos_beats / self.timesig_n

    def play(self):
        if not self.playing:
            self.playing = True
            self._last_monotonic = time.monotonic()

    def stop(self):
        self.playing = False

    def seek_to_bar(self, bar: float):
        self._pos_beats = max(0.0, bar) * self.timesig_n
        self._last_monotonic = time.monotonic()

    def toggle_play(self):
        if self.playing: self.stop()
        else: self.play()

    def toggle_loop(self):
        self.loop_enabled = not self.loop_enabled

    def update(self):
        if not self.playing: return
        now = time.monotonic()
        dt = now - self._last_monotonic
        self._last_monotonic = now
        beats_per_sec = self.bpm / 60.0
        self._pos_beats += dt * beats_per_sec
        # loop
        if self.loop_enabled:
            bar_len_beats = self.timesig_n
            loop_start_beats = self.loop_start_bar * bar_len_beats
            loop_end_beats = self.loop_end_bar * bar_len_beats
            if self._pos_beats >= loop_end_beats:
                self._pos_beats = loop_start_beats + (self._pos_beats - loop_end_beats)

    # --- helpers: time <-> bars ---
    def seconds_per_bar(self) -> float:
        beats_per_bar = float(self.timesig_n)
        return (60.0 / max(1e-6, float(self.bpm))) * beats_per_bar

    def bars_to_seconds(self, bars: float) -> float:
        return float(bars) * self.seconds_per_bar()

    def seconds_to_bars(self, sec: float) -> float:
        spb = self.seconds_per_bar()
        return float(sec) / spb if spb > 0 else 0.0
