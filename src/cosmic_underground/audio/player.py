import pygame
from cosmic_underground.core import config as C

class AudioPlayer:
    def __init__(self):
        pygame.mixer.init(frequency=C.ENGINE_SR, size=-16, channels=2, buffer=1024)
        self.boundary_event = pygame.USEREVENT + 1
        self.current = None
        self.loop_ms = 0

    def play_loop(self, wav_path: str, duration_sec: float, **kwargs):
        xfade = 0
        for key in ("cross_ms","crossfade_ms","xfade_ms","fade_ms"):
            if key in kwargs and kwargs[key] is not None:
                try: xfade = max(0, int(kwargs[key])); break
                except: xfade = 0
        self.loop_ms = max(1, int(duration_sec * 1000))
        pygame.time.set_timer(self.boundary_event, self.loop_ms)
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.fadeout(xfade) if xfade>0 else pygame.mixer.music.stop()
        pygame.mixer.music.load(wav_path)
        pygame.mixer.music.play(loops=-1, fade_ms=xfade)
        self.current = wav_path

    def stop(self, fade_ms: int = 200):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.fadeout(max(0, int(fade_ms)))
        pygame.time.set_timer(self.boundary_event, 0)
        self.current = None
        self.loop_ms = 0
