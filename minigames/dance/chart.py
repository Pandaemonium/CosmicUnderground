import json, os, math, hashlib, random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

# lanes are symbolic strings; engine maps them to keys
Lane = str

@dataclass
class Note:
    time_ms: int
    lane: Lane
    length_ms: int = 0      # 0 = tap, >0 = hold
    strength: float = 1.0   # visual weight/density

@dataclass
class ChartMeta:
    bpm: float
    bars: int
    timesig: Tuple[int, int]      # (beats_per_bar, beat_unit) e.g., (4,4)
    mood: str = "calm"
    source_hash: str = ""         # stable key for caching
    title: str = "Loop"

@dataclass
class Chart:
    notes: List[Note]
    loop_ms: int
    lanes: List[Lane]
    meta: ChartMeta

    def to_json(self) -> str:
        d = {
            "notes":[asdict(n) for n in self.notes],
            "loop_ms": self.loop_ms,
            "lanes": self.lanes,
            "meta": asdict(self.meta),
        }
        return json.dumps(d)

    @staticmethod
    def from_json(s: str) -> "Chart":
        d = json.loads(s)
        notes = [Note(**n) for n in d["notes"]]
        meta  = ChartMeta(**d["meta"])
        return Chart(notes=notes, loop_ms=d["loop_ms"], lanes=d["lanes"], meta=meta)

# --------- Helpers ---------
def _hash_loop_key(wav_path: str, bpm: float, bars: int, timesig: Tuple[int,int], mood: str, seed: int) -> str:
    h = hashlib.sha1()
    h.update(str(wav_path).encode("utf-8"))
    h.update(f"{bpm:.3f}|{bars}|{timesig}|{mood}|{seed}".encode("utf-8"))
    return h.hexdigest()

def _grid_times_ms(bpm: float, bars: int, timesig: Tuple[int,int]) -> Tuple[List[int], int, int]:
    beats_per_bar = int(timesig[0]) if timesig else 4
    beat_ms = 60000.0 / float(max(1, bpm))
    total_beats = int(beats_per_bar * bars)
    beat_times = [int(round(i * beat_ms)) for i in range(total_beats)]
    loop_ms = int(round(total_beats * beat_ms))
    return beat_times, loop_ms, beats_per_bar

# --------- Chart factory (metadata-only) ---------
LEFT_LANES  = ("A", "S", "D", "F")
RIGHT_LANES = ("LEFT", "DOWN", "UP", "RIGHT")
SPACE_LANE  = "SPACE"

MOOD_PRESETS = {
    # base density & flavor per bar (values are “how many events per bar” targets)
    "calm":        dict(perc_density=2, synco_prob=0.15, melody_density=1, kick_pattern="1__1", lane_spread=2),
    "energetic":   dict(perc_density=6, synco_prob=0.45, melody_density=3, kick_pattern="1_1_", lane_spread=4),
    "funky":       dict(perc_density=5, synco_prob=0.55, melody_density=2, kick_pattern="1__1", lane_spread=3),
    "playful":     dict(perc_density=4, synco_prob=0.35, melody_density=2, kick_pattern="1___", lane_spread=3),
    "angry":       dict(perc_density=7, synco_prob=0.30, melody_density=1, kick_pattern="11__",
                        lane_spread=4),
    "melancholy":  dict(perc_density=3, synco_prob=0.20, melody_density=2, kick_pattern="1___", lane_spread=2),
    "triumphant":  dict(perc_density=4, synco_prob=0.25, melody_density=2, kick_pattern="1__1", lane_spread=3),
    "brooding":    dict(perc_density=2, synco_prob=0.10, melody_density=1, kick_pattern="1___", lane_spread=2),
}

def _kick_beats_from_pattern(pattern: str, beats_per_bar: int) -> List[int]:
    """Pattern like '1__1' → indices [0,3] in a 4/4; stretches/loops if beats_per_bar != len(pattern)."""
    if beats_per_bar <= 0: return [0]
    out = []
    if not pattern:
        pattern = "1___"
    for i in range(beats_per_bar):
        ch = pattern[i % len(pattern)]
        if ch == "1":
            out.append(i)
    return out

def build_chart_from_metadata(*,
                              wav_path: str,
                              title: str,
                              bpm: float,
                              bars: int,
                              timesig: Tuple[int,int],
                              mood: str,
                              seed: int = 1337) -> Chart:
    """Create a loop-length chart with on-grid notes using just loop metadata."""
    mood_l = (mood or "calm").lower()
    preset = MOOD_PRESETS.get(mood_l, MOOD_PRESETS["calm"])

    rng = random.Random(seed ^ (hash(wav_path) & 0xFFFFFFFF))
    beat_times, loop_ms, beats_per_bar = _grid_times_ms(bpm, bars, timesig)

    # (1) Kicks (SPACE) on a simple pattern per bar
    kick_idxs = _kick_beats_from_pattern(preset["kick_pattern"], beats_per_bar)
    notes: List[Note] = []
    for bar in range(bars):
        for b in kick_idxs:
            i = bar*beats_per_bar + b
            if i < len(beat_times):
                notes.append(Note(time_ms=beat_times[i], lane=SPACE_LANE, length_ms=0, strength=1.0))

    # (2) Percussion (RIGHT lanes) — distribute across lanes, with optional syncopation on 8ths
    perc_target = preset["perc_density"] * bars
    perc_pool: List[int] = []

    # base: all quarter beats
    for i in range(len(beat_times)):
        perc_pool.append(beat_times[i])

    # add syncopated 8ths between beats with probability
    beat_ms = 60000.0 / max(1.0, bpm)
    for i in range(len(beat_times)-1):
        mid = int((beat_times[i] + beat_times[i+1]) * 0.5)
        if rng.random() < preset["synco_prob"]:
            perc_pool.append(mid)

    rng.shuffle(perc_pool)
    perc_pool = sorted(perc_pool[:perc_target])

    # distribute across right lanes
    right_cycle = list(RIGHT_LANES)
    ri = rng.randrange(len(right_cycle))
    for t in perc_pool:
        notes.append(Note(time_ms=t, lane=right_cycle[ri % len(right_cycle)], strength=0.9))
        ri += 1

    # (3) Melody proxy (LEFT lanes) — strong beats: start-of-bar and beat 3 in 4/4
    mel_target = preset["melody_density"] * bars
    melodic_times: List[int] = []
    for bar in range(bars):
        base = bar*beats_per_bar
        # start of bar
        if base < len(beat_times): melodic_times.append(beat_times[base])
        # strong mid-beat (only if 4/4)
        if beats_per_bar >= 4:
            mid = base + 2
            if mid < len(beat_times): melodic_times.append(beat_times[mid])

    rng.shuffle(melodic_times)
    melodic_times = sorted(melodic_times[:mel_target])

    left_cycle = list(LEFT_LANES)
    li = rng.randrange(len(left_cycle))
    for t in melodic_times:
        notes.append(Note(time_ms=t, lane=left_cycle[li % len(left_cycle)], strength=1.0))
        li += 1

    notes.sort(key=lambda n: n.time_ms)

    # lanes order for rendering (left block, space in middle, right block)
    lanes = list(LEFT_LANES) + [SPACE_LANE] + list(RIGHT_LANES)

    meta = ChartMeta(
        bpm=bpm, bars=bars, timesig=timesig, mood=mood_l,
        title=title, source_hash=_hash_loop_key(wav_path, bpm, bars, timesig, mood_l, seed),
    )
    return Chart(notes=notes, loop_ms=loop_ms, lanes=lanes, meta=meta)

# --------- Cache I/O ---------
def cache_path_for(chart_dir: str, meta: ChartMeta) -> str:
    os.makedirs(chart_dir, exist_ok=True)
    key = meta.source_hash or "chart"
    return os.path.join(chart_dir, f"{key}.json")

def save_chart(chart_dir: str, chart: Chart) -> str:
    path = cache_path_for(chart_dir, chart.meta)
    with open(path, "w", encoding="utf-8") as f:
        f.write(chart.to_json())
    return path

def load_chart_if_exists(chart_dir: str, meta: ChartMeta) -> Optional[Chart]:
    path = cache_path_for(chart_dir, meta)
    if not os.path.isfile(path): return None
    with open(path, "r", encoding="utf-8") as f:
        return Chart.from_json(f.read())
