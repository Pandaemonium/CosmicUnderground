from __future__ import annotations
import pygame
from .game_state import GameState
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cosmic_underground.app.context import GameContext

class DanceState(GameState):
    def on_enter(self, **kwargs):
        # Stop any other music and start the minigame
        self.context.audio.player.stop(fade_ms=0)
        self.context.dance_minigame.start_for_loop(
            loop=kwargs.get("loop"),
            src_title=kwargs.get("src_title"),
            player=self.context.audio.player,
            zone_spec=kwargs.get("zone_spec")
        )

    def handle_event(self, event):
        # The dance engine needs key events and its custom boundary event
        if event.type in (pygame.KEYDOWN, pygame.KEYUP, self.context.audio.player.boundary_event):
            self.context.dance_minigame.handle_event(event)
        # The dance engine handles its own exit via the on_finish callback, but we can add a manual escape
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
            self.context.dance_minigame.handle_event(event) # Let it finalize
            self.context.controller.change_state("overworld")

    def update(self, dt):
        self.context.dance_minigame.update(dt)

    def draw(self, screen):
        screen.fill((10, 10, 14))
        self.context.dance_minigame.draw(screen)
