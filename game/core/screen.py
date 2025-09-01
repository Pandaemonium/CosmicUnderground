import pygame
from typing import List, Optional

class Screen:
    def handle_event(self, ev: pygame.event.EventType) -> None: ...
    def update(self, dt: float) -> None: ...
    def draw(self, surf: pygame.Surface) -> None: ...

class ScreenManager:
    def __init__(self):
        self._stack: List[Screen] = []

    def push(self, s: Screen) -> None:
        self._stack.append(s)

    def pop(self) -> Optional[Screen]:
        return self._stack.pop() if self._stack else None

    def replace(self, s: Screen) -> None:
        if self._stack: self._stack.pop()
        self._stack.append(s)

    @property
    def top(self) -> Optional[Screen]:
        return self._stack[-1] if self._stack else None

    def handle_event(self, ev): 
        if self.top: self.top.handle_event(ev)

    def update(self, dt: float):
        if self.top: self.top.update(dt)

    def draw(self, surf: pygame.Surface):
        if self.top: self.top.draw(surf)
