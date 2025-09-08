from __future__ import annotations
import os, math, pygame
from typing import Optional, Set, List
from dataclasses import dataclass

from cosmic_underground.core.music import Song
from cosmic_underground.core import config as C

@dataclass
class BroadcastTrack:
    path: str
    tags: Set[str]
    base_quality: float = 0.0

class BroadcastDJ:
    def __init__(self, mixer_channel: int = 3):
        if pygame.mixer.get_init() is None:
            pygame.mixer.init(frequency=C.ENGINE_SR, size=-16, channels=2, buffer=2048)

        need = mixer_channel + 1
        if pygame.mixer.get_num_channels() < need:
            pygame.mixer.set_num_channels(need)

        self.chan_index = mixer_channel
        self.chan = pygame.mixer.Channel(self.chan_index)

        self.state: str = "stopped"    # "playing" | "paused" | "stopped"
        self.volume: float = 0.8
        self.fade_ms: int = 150

        # library & selection
        self.library: List[Song] = []
        self.sel_index: int = -1

        # current playback
        self.current: Optional[Song] = None
        self._sound: Optional[pygame.mixer.Sound] = None

        # Influence tuning
        self.radius_tiles = 6
        self.kw_weight    = 0.7
        self.qual_weight  = 0.3
        self.per_sec_gain = 22.0
        self.per_sec_decay= 6.0
        self.start_thresh = 110
        self.stop_thresh  = 95

    # --- library ---
    def load_library(self, songs: List[Song]):
        self.library = list(songs)
        self.sel_index = 0 if songs else -1

    def select_relative(self, step: int):
        if not self.library:
            self.sel_index = -1
            return
        self.sel_index = (self.sel_index + step) % len(self.library)

    # --- playback ---
    def play(self, song: Song | None = None, *, loop: bool = True):
        if song is None:
            if 0 <= self.sel_index < len(self.library):
                song = self.library[self.sel_index]
            else:
                return
        if not os.path.isfile(song.wav_path):
            return

        try:
            snd = pygame.mixer.Sound(song.wav_path)
        except Exception:
            return

        snd.set_volume(self.volume)
        loops = -1 if loop else 0
        self.chan.play(snd, loops=loops, fade_ms=self.fade_ms)
        self.current = song
        self._sound = snd
        self.state = "playing"

    def stop(self, fade_ms: int = 120):
        try:
            self.chan.fadeout(max(0, int(fade_ms)))
        finally:
            self.state = "stopped"
            self.current = None
            self._sound = None

    def pause(self):
        if self.state == "playing":
            self.chan.pause()
            self.state = "paused"

    def resume(self):
        if self.state == "paused":
            self.chan.unpause()
            self.state = "playing"

    def is_playing(self) -> bool:
        return self.state == "playing" and self.chan.get_busy()

    def set_volume(self, vol: float):
        self.volume = max(0.0, min(1.0, float(vol)))
        self.chan.set_volume(self.volume)

    # --- metadata for influence ---
    def _pref_match(self, song: Song, prefs: Set[str]) -> float:
        raw = (getattr(song, "tags", None) or getattr(song, "keywords", None) or [])
        song_tags = {str(t).strip().lower() for t in raw if t}
        prefs_l   = {str(p).strip().lower() for p in (prefs or set()) if p}
        if not song_tags or not prefs_l:
            return 0.0
        inter = song_tags & prefs_l
        union = song_tags | prefs_l
        return len(inter) / max(1, len(union))

    def current_tags(self) -> Set[str]:
        if self.current:
            raw = getattr(self.current, "tags", None) or getattr(self.current, "keywords", None) or []
            return {str(t).strip().lower() for t in raw if t}
        return set()

    def current_base_quality(self) -> float:
        if self.current:
            return float(getattr(self.current, "base_quality", 0.0))
        return 0.0

    # --- influence ---
    def apply_influence(self, *, world_model, dt_sec: float):
        if self.state != "playing" or not self.current:
            self._decay_all(world_model, dt_sec)
            return

        px, py = world_model.player.tile_x, world_model.player.tile_y
        song = self.current

        for poi in world_model.map.pois.values():
            if poi.kind != "npc":
                continue
            mind = getattr(poi, "mind", None)
            if mind is None:
                continue

            dx = abs(poi.tile[0] - px)
            dy = abs(poi.tile[1] - py)
            if max(dx, dy) > self.radius_tiles:
                self._decay_one(poi, dt_sec)
                continue

            kwm  = self._pref_match(song, getattr(mind, "prefs", set()))
            qual = (float(getattr(song, "base_quality", 0.0)) * 0.5) + 0.5  # -1..1 -> 0..1
            score = self.kw_weight * kwm + self.qual_weight * qual

            signed = (score - 0.45) * 2.0
            delta = signed * self.per_sec_gain * dt_sec
            mind.affinity = max(-100.0, min(200.0, mind.affinity + delta))

            if not mind.is_dancing and mind.disposition > self.start_thresh:
                mind.is_dancing = True
            elif mind.is_dancing and mind.disposition < self.stop_thresh:
                mind.is_dancing = False

    def _decay_one(self, poi, dt_sec: float):
        mind = getattr(poi, "mind", None)
        if not mind:
            return
        target = float(mind.disposition_base * 2)  # -100..+100
        cur = mind.affinity
        step = self.per_sec_decay * dt_sec
        if cur < target:
            mind.affinity = min(target, cur + step)
        elif cur > target:
            mind.affinity = max(target, cur - step)

    def _decay_all(self, world_model, dt_sec: float):
        for poi in world_model.map.pois.values():
            if poi.kind == "npc":
                self._decay_one(poi, dt_sec)
