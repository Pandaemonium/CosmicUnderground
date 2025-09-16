from __future__ import annotations
from abc import ABC, abstractmethod
import pygame
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cosmic_underground.app.context import GameContext

class GameState(ABC):
    """Abstract base class for a game state."""
    def __init__(self, context: GameContext):
        self.context = context

    def on_enter(self, **kwargs):
        """Called when entering the state."""
        pass

    def on_exit(self):
        """Called when exiting the state."""
        pass

    @abstractmethod
    def handle_event(self, event: pygame.event.Event):
        raise NotImplementedError

    @abstractmethod
    def update(self, dt: int):
        raise NotImplementedError

    @abstractmethod
    def draw(self, screen: pygame.Surface):
        raise NotImplementedError
