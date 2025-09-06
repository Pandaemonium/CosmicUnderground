from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, Tuple
import math

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

def meter_fractions(affinity: float) -> tuple[float,float,float,float]:
    """
    Return (red, grey, green, purple) fill fractions 0..1 for the 16-dot ring.

    Mapping requested:
      -100 => all red
       -50 => 0.5 red, 0.5 grey
         0 => all grey
        25 => 0.25 green
       100 => all green
       150 => 1.0 green base + 0.5 purple overlay
       200 => all purple overlay
    """
    if affinity <= 0.0:
        t = min(1.0, max(0.0, -affinity / 100.0))  # 0 at 0, 1 at -100
        red = t
        grey = 1.0 - t
        return (red, grey, 0.0, 0.0)
    else:
        green = min(1.0, affinity / 100.0)         # 0..1 at 0..100
        purple = min(1.0, max(0.0, (affinity - 100.0) / 100.0))  # 0..1 at 100..200
        return (0.0, 0.0, green, purple)

# -------- Minimal mind type (imported by models) --------

@dataclass
class NPCMind:
    disposition_base: int = 0               # -50..+50
    prefs: Set[str] = field(default_factory=set)
    affinity: float = 0.0                   # -100..+200
    is_dancing: bool = False
    last_match: float = 0.0
    last_heard_ms: int = 0
