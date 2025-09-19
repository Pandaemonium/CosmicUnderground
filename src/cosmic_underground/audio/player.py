import pygame
from cosmic_underground.core import config as C

class AudioPlayer:
    def __init__(self):
        if pygame.mixer.get_init() is None:
            pygame.mixer.init(frequency=C.ENGINE_SR, size=-16, channels=2, buffer=1024)
        self.boundary_event = pygame.USEREVENT + 1
        self.fade_finished_event = pygame.USEREVENT + 2
        self.current = None
        self.loop_ms = 0
        self._next_wav = None

    def play_loop(self, wav_path: str, duration_sec: float, **kwargs):
        xfade = 0
        for key in ("cross_ms","crossfade_ms","xfade_ms","fade_ms"):
            if key in kwargs and kwargs[key] is not None:
                try: xfade = max(0, int(kwargs[key])); break
                except: xfade = 0

        self.loop_ms = max(1, int(duration_sec * 1000))
        pygame.time.set_timer(self.boundary_event, self.loop_ms)

        if pygame.mixer.music.get_busy():
            # If something is playing, schedule the new track to play after the fadeout.
            self._next_wav = wav_path
            pygame.mixer.music.set_endevent(self.fade_finished_event)
            pygame.mixer.music.fadeout(xfade)
        else:
            # If nothing is playing, start immediately.
            pygame.mixer.music.load(wav_path)
            pygame.mixer.music.play(loops=-1, fade_ms=xfade)
            self.current = wav_path
            self._next_wav = None
            pygame.mixer.music.set_endevent() # Clear end event

    def stop(self, fade_ms: int = 200):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.fadeout(max(0, int(fade_ms)))
        pygame.time.set_timer(self.boundary_event, 0)
        self.current = None
        self.loop_ms = 0
        self._next_wav = None
        pygame.mixer.music.set_endevent()

    def handle_fade_finish(self):
        """Called when the fadeout completes."""
        if self._next_wav:
            try:
                self.play_loop(self._next_wav, self.loop_ms / 1000.0, fade_ms=220)
            finally:
                self._next_wav = None
