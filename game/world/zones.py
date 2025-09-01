import pygame
from ..core.config import ZONES, ZONE_COLOR, LISTEN_COLOR, TEXT_COLOR, ZONE_COLORS, ZONE_OUTLINE

class Zones:
    def __init__(self):
        self.rects = []
        self.labels = []
        for name, (x, y, w, h) in ZONES.items():
            self.labels.append(name)
            self.rects.append(pygame.Rect(x, y, w, h))
            print(f"🎯 Zone loaded: {name} at ({x}, {y}, {w}, {h})")
        
        print(f"🎯 Total zones loaded: {len(self.labels)}")
        print(f"🎯 Zone labels: {self.labels}")

    def which(self, pos) -> int:
        for i, r in enumerate(self.rects):
            if r.collidepoint(pos):
                return i
        return -1

    def draw(self, screen, FONT):
        for i, r in enumerate(self.rects):
            name = self.labels[i]
            color = ZONE_COLORS.get(name, (120, 120, 180))  # fallback
            pygame.draw.rect(screen, color, r)
            pygame.draw.rect(screen, ZONE_OUTLINE, r, 1)
            # Use dark label on bright zones if you prefer:
            screen.blit(FONT.render(name, True, TEXT_COLOR), (r.x + 10, r.y + 10))