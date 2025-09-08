from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from cosmic_underground.core.affinity import NPCMind
import random

@dataclass
class ZoneSpec:
    name: str
    bpm: int
    key_mode: str
    scene: str
    mood: str
    bars: int = 8
    timesig: Tuple[int,int] = (4,4)
    prompt_override: Optional[str] = None
    biome: str = "Unknown"
    species: str = "Unknown"
    descriptors: List[str] = field(default_factory=list)
    instruments: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class GeneratedLoop:
    wav_path: str
    duration_sec: float
    bpm: float
    key_mode: str
    prompt: str
    provider_meta: Dict = field(default_factory=dict)

@dataclass
class ZoneRuntime:
    id: int
    spec: ZoneSpec
    tiles: List[Tuple[int,int]]
    centroid: Tuple[float,float]
    loop: Optional[GeneratedLoop] = None
    generating: bool = False
    error: Optional[str] = None

@dataclass
class POI:
    pid: int
    kind: str
    name: str
    role: str
    tile: Tuple[int,int]
    zone_id: int
    rarity: int = 0
    kind_key: Optional[str] = None
    mood_hint: Optional[str] = None
    bpm_hint: Optional[int] = None
    bars_hint: Optional[int] = None
    generating: bool = False
    loop: Optional[GeneratedLoop] = None
    error: Optional[str] = None
    last_seed: Optional[int] = None
    sprite_id: str = "shagdeliac1"
    mind: NPCMind | None = None    
    species: Optional[str] = None

@dataclass
class Player:
    tile_x: int
    tile_y: int
    px: float
    py: float
    speed: float = 6.0

@dataclass
class Quest:
    giver_pid: int
    target_pid: int
    target_name: str
    target_tile: Tuple[int,int]
    target_zone: int
    target_zone_name: str
    accepted: bool = True
