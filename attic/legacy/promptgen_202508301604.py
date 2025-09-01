# promptgen.py
# Structured prompt builder for Stable Audio Small
# - Deterministic via caller-supplied rng
# - Clean, order-aware prompts: Format → (alien tag) → Genre → Sub-genre → Instruments → Moods → Styles → BPM → Key → Extras
# - Guards bar length to fit Stable Audio Small (~11s)
# - Emits meta for debugging / UI
#
# Expected caller:
#   prompt, meta = promptgen.build(zone_spec, bars=zone_spec.bars, rng=random.Random(), intensity=0.55, **context)
#
# Context kwargs (optional):
#   time_of_day: "day" | "dusk" | "night" | "dawn"
#   weather: e.g., "light rain" | "storm" | "snow"
#   heat: 0..1
#   debt_pressure: 0..1
#   festival: bool
#   cosmic_event: bool
#   neighbor_contrast_prev: e.g., "retro" if last tile used that tag (to nudge variety)

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import random

SCHEMA_VERSION = "promptgen/v2"

# ---- Small helpers ----------------------------------------------------------

ALIEN_WORDS = ["alien", "weird", "extraterrestrial", "cosmic", "otherworldly", "offworld", "xeno"]

CONTRAST_PAIRS: List[Tuple[str, str]] = [
    ("fast", "slow"),
    ("funky", "classic"),
    ("brassy", "woodwinds"),
    ("loud", "soft"),
    ("layered", "simple"),
    ("retro", "futuristic"),
]

GENRE_BY_SPECIES = {
    "Chillaxians": ("Ambient / Downtempo", ["Chillhop", "Coldwave", "Ice ambient"]),
    "Glorpals": ("Funk / Electronic", ["Acid funk", "Nu-disco", "Liquid breaks"]),
    "Bzaris": ("Electronic / Breaks", ["Glitch hop", "IDM", "Neuro break"]),
    "Shagdeliacs": ("Funk / Jazz", ["P-Funk", "Jazz-funk", "Boogie"]),
    "Rockheads": ("Industrial / Percussive", ["Tribal industrial", "Metallic dub", "Foundry beats"]),
}

GENRE_BY_BIOME = {
    "Crystal Canyons": ("Ambient / Electronic", ["Crystal ambient", "Glacial IDM"]),
    "Slime Caves": ("Funk / Experimental", ["Slime funk", "Bio-house"]),
    "Centerspire": ("Electronic / Synth", ["Synthwave", "Arcade breaks"]),
    "Groove Pits": ("Funk / House", ["Jackin' house", "P-Funk"]),
    "Polar Ice": ("Ambient / Downtempo", ["Arctic downtempo"]),
    "Mag-Lev Yards": ("Techno / Industrial", ["Mag-rail techno", "Electro foundry"]),
    "Electro Marsh": ("Bass / UKG", ["2-step garage", "Marsh bass"]),
}

EXTRA_INSTR = [
    "kalimba", "woodblocks", "808 subs", "tape delay stabs", "analog brass",
    "vocoder chops", "hand drums", "dub chords", "FM plucks", "bitcrush hats",
    "gel bells", "ice chimes", "metal gongs", "anvil hits", "wah guitar", "clavinet",
]

PRODUCTION_TERMS = [
    "punchy", "tape-saturated", "compressed", "lofi", "hi-fi", "gritty",
    "wet", "dry", "sproingy", "glassy", "mechanical", "magnetic", "laser",
    "slurpy", "fuzzy", "crystalline", "retro", "futuristic",
]

TONICS = ["A","B","C","D","E","F","G","A♭","B♭","C♯","D♯","F♯","G♯"]
MODES  = ["major","minor","Dorian","Phrygian","Mixolydian","Lydian","Aeolian"]

def _get(obj: Any, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)

def _uniq_keep_order(xs: Iterable[str]) -> List[str]:
    seen = set(); out = []
    for x in xs:
        if not x or x in seen: continue
        seen.add(x); out.append(x)
    return out

def _beats_per_bar(timesig: Tuple[int,int]) -> int:
    # treat numerator as beats count (common for 4/4, 3/4, etc.)
    return max(1, int(timesig[0] if isinstance(timesig, (tuple, list)) and len(timesig)>=1 else 4))

def _fit_bars_to_small(bpm: int, bars: int, timesig: Tuple[int,int], max_seconds: float = 11.0) -> Tuple[int, float]:
    """Clamp bars so total loop duration fits Stable Audio Small (~11s). Returns (bars, seconds)."""
    beats = bars * _beats_per_bar(timesig)
    sec = beats * (60.0 / max(1, bpm))
    if sec <= max_seconds:
        return bars, sec
    # shrink bars but keep at least 2
    per_bar = (60.0 / max(1, bpm)) * _beats_per_bar(timesig)
    if per_bar <= 0:  # fallback
        return max(2, bars), min(sec, max_seconds)
    max_bars = max(2, int(max_seconds // per_bar))
    max_bars = max(2, min(bars, max_bars))
    beats = max_bars * _beats_per_bar(timesig)
    sec = beats * (60.0 / max(1, bpm))
    return max_bars, sec

def _choose_contrast(rng: random.Random, prefer_opposite_of: Optional[str]) -> str:
    if prefer_opposite_of:
        for a, b in CONTRAST_PAIRS:
            if prefer_opposite_of == a: return b
            if prefer_opposite_of == b: return a
    # else pick a random side
    a, b = rng.choice(CONTRAST_PAIRS)
    return rng.choice([a, b])

def _derive_genre(species: Optional[str], biome: Optional[str], rng: random.Random) -> Tuple[str, str]:
    base, subs = GENRE_BY_SPECIES.get(species or "", (None, None))
    if not base:
        base, subs = GENRE_BY_BIOME.get(biome or "", ("Electronic", ["Experimental"]))
    sub = rng.choice(subs)
    return base, sub

def _pick_instruments(spec_instruments: List[str], species: Optional[str], intensity: float, rng: random.Random) -> List[str]:
    pool = list(spec_instruments or [])
    # gentle species-based hints
    hints = {
        "Chillaxians": ["warm pads","ice chimes","soft bass","brush kit","chorus keys"],
        "Glorpals": ["squelch synth","gel bells","drip fx","rubber bass","wet claps"],
        "Bzaris": ["bitcrush blips","granular stutters","noisy hats","FM lead","resonant zap"],
        "Shagdeliacs": ["wah guitar","clavinet","horn section","rimshot kit","upright-ish synth bass"],
        "Rockheads": ["metal gongs","anvil hits","deep toms","clang perc","industrial pad"],
    }.get(species or "", [])
    pool += hints + EXTRA_INSTR
    pool = _uniq_keep_order(pool)
    k = max(2, min(5, 2 + int(3 * intensity)))
    return rng.sample(pool, k=k) if len(pool) >= k else pool

def _pick_moods(mood: Optional[str], tags: List[str], descriptors: List[str], rng: random.Random, ctx: Dict) -> List[str]:
    out = []
    if mood: out.append(mood.strip().lower())
    # bias with a few tags/descriptors
    out += rng.sample(_uniq_keep_order((tags or []) + (descriptors or [])), k=min(3, max(0, len((tags or []) + (descriptors or [])))))
    # contextual seasoning
    tod = (ctx.get("time_of_day") or "").lower()
    weather = (ctx.get("weather") or "").lower()
    if tod in ("night","dusk"): out.append("nocturnal")
    if tod == "dawn": out.append("hazy")
    if "storm" in weather: out.append("stormy")
    if "snow" in weather: out.append("icy")
    if ctx.get("festival"): out.append("festive")
    if ctx.get("cosmic_event"): out.append("cosmic")
    # keep 2–4
    out = _uniq_keep_order(out)
    if len(out) < 2:
        out += rng.sample(["playful","energetic","calm","gritty","brooding","glittery"], k=2-len(out))
    if len(out) > 4: out = rng.sample(out, k=4)
    return out

def _pick_styles(tags: List[str], descriptors: List[str], contrast: str, rng: random.Random) -> List[str]:
    base = _uniq_keep_order(tags + descriptors + PRODUCTION_TERMS)
    picks = rng.sample(base, k=min(3, len(base))) if base else []
    # ensure contrast shows up
    if contrast not in picks:
        picks = ([contrast] + picks)[:4]
    return _uniq_keep_order(picks)[:4]

def _coerce_key(key_mode: Optional[str], rng: random.Random) -> str:
    if key_mode and isinstance(key_mode, str) and key_mode.strip():
        return key_mode
    return f"{rng.choice(TONICS)} {rng.choice(MODES)}"

def _format_field(name: str, value: str) -> str:
    return f"{name}: {value}"

# ---- Public API -------------------------------------------------------------

def build(spec: Any, bars: int = 4, rng: Optional[random.Random] = None, intensity: float = 0.5, **context) -> Tuple[str, Dict]:
    """
    Build a Stable Audio Small-friendly prompt string + meta.
    - spec: ZoneSpec-like object (attrs or dict keys: species, biome, mood, bpm, key_mode, descriptors, instruments, tags, timesig, name)
    - bars: desired bars; will be clamped to fit ~11s
    - rng: random.Random seeded upstream for determinism
    - intensity: 0..1 affects density of instruments/styles
    - context: time_of_day, weather, heat, debt_pressure, festival, cosmic_event, neighbor_contrast_prev
    """
    rng = rng or random.Random()

    species     = _get(spec, "species", "Unknown")
    biome       = _get(spec, "biome", "Unknown biome")
    mood        = _get(spec, "mood", "energetic")
    bpm         = int(_get(spec, "bpm", 120) or 120)
    key_mode    = _coerce_key(_get(spec, "key_mode", None), rng)
    timesig     = _get(spec, "timesig", (4,4))
    descs       = list(_get(spec, "descriptors", []) or [])
    instr_src   = list(_get(spec, "instruments", []) or [])
    tags        = list(_get(spec, "tags", []) or [])
    name        = _get(spec, "name", "Unknown Zone")

    # Contrast alternation for variety
    contrast = _choose_contrast(rng, context.get("neighbor_contrast_prev"))

    # Instruments / moods / styles
    instruments = _pick_instruments(instr_src, species, intensity, rng)
    moods       = _pick_moods(mood, tags, descs, rng, context)
    styles      = _pick_styles(tags, descs, contrast, rng)

    # Genre mapping
    genre, subgenre = _derive_genre(species, biome, rng)

    # Format: Solo if lean, Band/Ensemble if richer
    fmt = "Ensemble" if len(instruments) >= 4 else ("Band" if len(instruments) >= 3 else "Solo")

    # Stable Audio Small cap guard
    bars_fitted, loop_seconds = _fit_bars_to_small(bpm=bpm, bars=bars, timesig=timesig, max_seconds=11.0)

    # One of your requested words near the start
    alien_word = rng.choice(ALIEN_WORDS)

    # Assemble (order matters)
    fields = [
        _format_field("Format", fmt),
        alien_word,  # near-start placement per request
        _format_field("Genre", genre),
        _format_field("Sub-genre", subgenre),
        _format_field("Instruments", ", ".join(instruments)),
        _format_field("Moods", ", ".join(moods)),
        _format_field("Styles", ", ".join(styles)),
        _format_field("BPM", str(bpm)),
        _format_field("Key", key_mode),
        _format_field(
            "Extras",
            f"loopable {bars_fitted} bars, seamless loop, clean downbeat, minimal silence at edges, no vocals, tight low end"
        ),
    ]

    prompt = " | ".join(fields)

    meta = {
        "schema": SCHEMA_VERSION,
        "zone_name": name,
        "species": species,
        "biome": biome,
        "format": fmt,
        "genre": genre,
        "subgenre": subgenre,
        "instruments": instruments,
        "moods": moods,
        "styles": styles,
        "bpm": bpm,
        "key": key_mode,
        "bars_requested": bars,
        "bars_used": bars_fitted,
        "time_signature": timesig,
        "loop_seconds_est": round(loop_seconds, 2),
        "contrast": contrast,
        "alien_word": alien_word,
        "context": {
            "time_of_day": context.get("time_of_day"),
            "weather": context.get("weather"),
            "heat": context.get("heat"),
            "debt_pressure": context.get("debt_pressure"),
            "festival": context.get("festival"),
            "cosmic_event": context.get("cosmic_event"),
            "neighbor_contrast_prev": context.get("neighbor_contrast_prev"),
        },
    }

    return prompt, meta
