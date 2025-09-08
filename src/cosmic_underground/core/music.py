# src/cosmic_underground/core/music.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, Iterable

@dataclass
class Song:
    title: str
    wav_path: str
    keywords: Set[str] = field(default_factory=set)   # e.g. {"funk", "brassy", "retro"}
    base_quality: float = 0.0                         # -1.0 .. +1.0

    @property
    def tags(self) -> Set[str]:
        """Back-compat alias so older code expecting .tags keeps working."""
        return set(self.keywords or [])

    @staticmethod
    def kw(xs: Iterable[str]) -> Set[str]:
        return {x.strip().lower() for x in xs if x and x.strip()}
