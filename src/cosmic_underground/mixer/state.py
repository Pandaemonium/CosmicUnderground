from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

TrackColor = tuple[int, int, int]

@dataclass
class Clip:
    clip_id: str
    track_id: int
    source_path: str
    start_bar: float
    length_beats: float
    source_offset_sec: float = 0.0
    gain_db: float = 0.0
    time_stretch: float = 1.0
    transpose_semitones: int = 0
    fitted_path: str | None = None

@dataclass
class Track:
    track_id: int
    name: str
    color: TrackColor
    clips: List[Clip] = field(default_factory=list)
    gain_db: float = 0.0
    pan: float = 0.0
    mute: bool = False
    solo: bool = False

@dataclass
class Project:
    title: str = "New Project"
    bpm: float = 110.0
    timesig: Tuple[int, int] = (4, 4)
    key: str = "C major"
    tracks: List[Track] = field(default_factory=list)
    loop_region_bars: Tuple[float, float] | None = (0.0, 8.0)

def new_blank_project() -> Project:
    # 7 tracks with pleasant colors
    palette = [
        (235,120,120),(235,180,120),(235,235,120),
        (150,220,150),(120,200,235),(170,150,235),
        (230,150,200),
    ]
    tracks = [Track(i, f"Track {chr(65+i)}", palette[i % len(palette)]) for i in range(7)]
    return Project(tracks=tracks)

# simple increasing clip id for this process
_CLIP_ID = 1

def next_clip_id() -> int:
    global _CLIP_ID
    i = _CLIP_ID
    _CLIP_ID += 1
    return i
