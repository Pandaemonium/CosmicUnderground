from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, Tuple
import math
import pygame

# -------- Public API --------

def match_score(play_tags: Set[str], prefs: Set[str]) -> float:
    """
    Return -1..+1-ish likeness score.
    >0 means they like it, <0 dislike.
    """
    if not play_tags or not prefs:
        return 0.0
    inter = len(play_tags & prefs)
    if inter == 0:
        return -0.3
    frac = inter / max(1, len(prefs))
    return 0.4 + 0.6 * min(1.0, frac)  # 0.4..1.0

def chebyshev(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

def update_npc_affinity(mind: "NPCMind", dt_ms: int, can_hear: bool, play_tags: Set[str]) -> None:
    """Update the running affinity values (dt in ms)."""
    alpha = 0.08
    raw = match_score(play_tags, mind.prefs) if can_hear else 0.0
    mind.last_match = (1 - alpha) * mind.last_match + alpha * raw

    disp = mind.disposition_base / 50.0  # -1..+1ish
    hear_boost = (1.0 + 0.6 * disp)      # 0.4..1.6

    drift = -0.015 if not can_hear else 0.0  # per second

    d_affinity = 0.0
    if can_hear:
        d_affinity = 12.0 * mind.last_match * hear_boost
        d_affinity -= 1.5  # tiny fatigue

    mind.affinity += (d_affinity + drift) * (dt_ms / 1000.0)
    mind.affinity = max(-100.0, min(200.0, mind.affinity))

    # dance thresholds
    if not mind.is_dancing:
        if mind.affinity >= 50.0 and mind.last_match >= 0.5 and can_hear:
            mind.is_dancing = True
    else:
        if (mind.last_match < 0.2) or (mind.affinity < 45.0) or (not can_hear):
            mind.is_dancing = False

# -------- Meter math (your exact mapping) --------

def meter_fractions(affinity: int) -> tuple[float,float,float,float]:
    """
    Map affinity [-100..200] -> (red, grey, green, purple) fractions that sum to 1.
    - -100  -> all red
    -  -50  -> half red, half grey
    -    0  -> all grey
    -   25  -> 1/4 green, rest grey
    -  100  -> all green
    -  150  -> half green, half purple
    -  200  -> all purple
    """
    a = max(-100, min(200, int(affinity)))
    if a <= 0:
        red = abs(a) / 100.0
        grey = 1.0 - red
        green = purple = 0.0
    else:
        green  = min(a, 100) / 100.0
        purple = max(0, a - 100) / 100.0
        grey   = max(0.0, 1.0 - green - purple)
        red    = 0.0
    return red, grey, green, purple


def _filled_wedge(surf: pygame.Surface, center: tuple[int,int], radius: int,
                  start_deg: float, sweep_deg: float, color: tuple[int,int,int]):
    """
    Draw a filled wedge by triangulating from the center.
    start_deg is clockwise degrees, 0° at right; we'll rotate so -90° = top.
    """
    if sweep_deg <= 0: 
        return
    cx, cy = center
    steps = max(8, int(abs(sweep_deg) / 4))  # quality vs perf
    step = sweep_deg / steps
    pts = [(cx, cy)]
    for i in range(steps + 1):
        a = math.radians(start_deg + i * step)
        x = int(cx + radius * math.cos(a))
        y = int(cy + radius * math.sin(a))
        pts.append((x, y))
    pygame.draw.polygon(surf, color, pts)

# -------- Minimal mind type (imported by models) --------

@dataclass
class NPCMind:
    disposition_base: int = 80               # -50..+50
    prefs: Set[str] = field(default_factory=set)
    affinity: float = 0.0                   # -100..+200
    is_dancing: bool = False
    last_match: float = 0.0
    last_heard_ms: int = 0

    @property
    def disposition(self) -> int:
        """
        Combine long-term baseline and live affinity.
        Result is clamped to [-100, 200] for the groove meter & logic.
        """
        total = int(round(self.affinity)) + int(self.disposition_base)  # -100..+200 + (-50..+50)
        return max(-100, min(200, total))