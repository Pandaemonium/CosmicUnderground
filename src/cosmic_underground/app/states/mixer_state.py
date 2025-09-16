from __future__ import annotations
import pygame
from .game_state import GameState
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cosmic_underground.app.context import GameContext

class MixerState(GameState):
    def on_enter(self, **kwargs):
        self.context.audio.disable_env_playback()
        # Mute broadcast audio so it doesn't compete with the DAW
        try:
            self._prev_broadcast_vol = getattr(self.context.audio.broadcast, "volume", None)
            if self._prev_broadcast_vol is not None:
                self.context.audio.broadcast.set_volume(0.0)
        except Exception:
            self._prev_broadcast_vol = None

    def on_exit(self):
        # Stop DAW audio and resume environment sounds
        self.context.mixer.engine.stop_all()
        self.context.mixer.transport.stop()
        self.context.audio.enable_env_playback()
        # Restore broadcast volume
        if hasattr(self, "_prev_broadcast_vol") and self._prev_broadcast_vol is not None:
            try:
                self.context.audio.broadcast.set_volume(self._prev_broadcast_vol)
            except Exception:
                pass

    def handle_event(self, event):
        if self.context.mixer.handle_event(event):
            return
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_TAB):
            self.context.controller.change_state("overworld")

    def update(self, dt):
        self.context.mixer.update(dt)

    def draw(self, screen):
        self.context.mixer.draw(screen)
