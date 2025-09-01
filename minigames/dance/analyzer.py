# Metadata-only “analyzer”: just wraps grid building so engine code has a stable interface.
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class BeatGrid:
    bpm: float
    beat_ms: float
    beats_per_bar: int
    bars: int
    loop_ms: int

def build_grid(bpm: float, bars: int, timesig: Tuple[int,int]) -> BeatGrid:
    bpb = int(timesig[0]) if timesig else 4
    beat_ms = 60000.0 / max(1.0, bpm)
    total_beats = bpb * bars
    loop_ms = int(round(total_beats * beat_ms))
    return BeatGrid(bpm=bpm, beat_ms=beat_ms, beats_per_bar=bpb, bars=bars, loop_ms=loop_ms)
