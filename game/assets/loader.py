import os
import pygame
import numpy as np
from typing import Optional, Tuple

def ensure_int16_stereo(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.int16:
        arr = arr.astype(np.int16)
    if arr.ndim == 1:
        arr = np.column_stack([arr, arr])
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
    return arr

def load_sprite(path: str, size: Tuple[int, int]) -> Optional[pygame.Surface]:
    try:
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(img, size)
    except Exception as e:
        print("Sprite load failed:", e)
        return None

def load_audio_to_array(path: str) -> Optional[Tuple[str, np.ndarray]]:
    if not path or not os.path.isfile(path):
        print("File not found:", path)
        return None
    try:
        snd = pygame.mixer.Sound(path)
        arr = pygame.sndarray.array(snd)
        arr = ensure_int16_stereo(arr)
        name = os.path.splitext(os.path.basename(path))[0]
        return name, arr
    except Exception as e:
        print("Audio load failed:", e)
        return None
