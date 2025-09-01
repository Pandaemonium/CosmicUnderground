import pygame
from ..core.config import TILE_SIZE, PLAYER_SCALE, PLAYER_COLOR
from ..assets.loader import load_sprite

class Player:
    def __init__(self, sprite_path: str):
        self.size = int(TILE_SIZE * PLAYER_SCALE)
        self.sprite = load_sprite(sprite_path, (self.size, self.size))
        self.pos = [120.0, 120.0]
        self.speed = 200.0
        self.radius = 12

    def bounds(self):
        if self.sprite:
            half_w = self.sprite.get_width() // 2
            half_h = self.sprite.get_height() // 2
        else:
            half_w = half_h = self.radius
        return half_w, half_h

    def update(self, dt, keys, width, height):
        dx = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        dy = (keys[pygame.K_s] or keys[pygame.K_DOWN]) - (keys[pygame.K_w] or keys[pygame.K_UP])
        if dx or dy:
            mag = (dx*dx + dy*dy) ** 0.5
            dx /= mag; dy /= mag
            half_w, half_h = self.bounds()
            self.pos[0] = max(half_w, min(width - half_w, self.pos[0] + dx * self.speed * dt))
            self.pos[1] = max(half_h, min(height - half_h, self.pos[1] + dy * self.speed * dt))

    def draw(self, screen):
        if self.sprite:
            half_w, half_h = self.bounds()
            screen.blit(self.sprite, (int(self.pos[0]) - half_w, int(self.pos[1]) - half_h))
        else:
            pygame.draw.circle(screen, PLAYER_COLOR, (int(self.pos[0]), int(self.pos[1])), self.radius)
    
    # Movement methods for the overworld screen
    def move_up(self):
        """Move player up"""
        half_w, half_h = self.bounds()
        self.pos[1] = max(half_h, self.pos[1] - self.speed * 0.016)  # 60 FPS = 0.016s per frame
    
    def move_down(self):
        """Move player down"""
        half_w, half_h = self.bounds()
        self.pos[1] = min(1024 - half_h, self.pos[1] + self.speed * 0.016)  # 60 FPS = 0.016s per frame
    
    def move_left(self):
        """Move player left"""
        half_w, half_h = self.bounds()
        self.pos[0] = max(half_w, self.pos[0] - self.speed * 0.016)  # 60 FPS = 0.016s per frame
    
    def move_right(self):
        """Move player right"""
        half_w, half_h = self.bounds()
        self.pos[0] = min(1600 - half_w, self.pos[0] + self.speed * 0.016)  # 60 FPS = 0.016s per frame
    
    def stop_movement(self):
        """Stop player movement (no-op for now, but keeps interface consistent)"""
        pass
