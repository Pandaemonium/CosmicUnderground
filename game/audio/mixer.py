import pygame
from typing import Dict
from typing import Optional

AUDITION_CH: Optional[pygame.mixer.Channel] = None  # <— give it a binding

def init_mixer():
    pygame.mixer.pre_init(44100, size=-16, channels=2, buffer=512)
    pygame.mixer.set_num_channels(64)
    global AUDITION_CH
    AUDITION_CH = pygame.mixer.Channel(63)

AUDITION_CH: pygame.mixer.Channel  # set in init_mixer()

def stop_all_audio(inventory, active_event_channels: Dict[int, pygame.mixer.Channel]):
    try:
        AUDITION_CH.stop()
    except Exception:
        pass
    for tr in inventory:
        tr.stop()
    for ch in list(active_event_channels.values()):
        try: ch.stop()
        except Exception: pass
    active_event_channels.clear()
