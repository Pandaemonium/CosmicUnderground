# promptgen.py
# Ultra-compact funk prompts (10–15 words) for Stable Audio Small
# Keeps alien vibe, strong groove, bass-forward. Same build(...) API.
from typing import Any, Dict, Iterable, List, Optional, Tuple
import random

SCHEMA_VERSION = "promptgen/v5_slimfunk"

ALIEN = ["alien", "weird", "extraterrestrial", "cosmic", "offworld", "xeno"]
GENRES = ["P-Funk", "Boogie", "Electro-funk", "Jazz-funk", "G-Funk", "Space-funk"]
BASS_PACKS = [
    ["slap", "bass"], ["rubber", "bass"], ["analog", "synth", "bass"],
    ["moog", "bass"], ["808", "subs"], ["talkbox", "bass"]
]
COLOR_INSTR = [
    ["wah", "guitar"], ["gamelan", "bells"], ["synth", "moog"],  ["horn", "stabs"], ["rimshot", "kit"],
    ["707", "drums"], ["909", "hats"], ["conga", "groove"], ["vocoder", "chops"]
]
GROOVE = ["swung", "syncopated", "in-the-pocket", "ghost", "notes", "octave", "jumps", "slides"]
TONICS = ["A","B","C","D","E","F","G","A♭","B♭","C♯","D♯","F♯","G♯"]
MODES  = ["minor","Dorian","Mixolydian","major"]

CONTRAST_PAIRS = [("funky","classic"),("retro","futuristic"),("brassy","woodwinds"),
                  ("layered","simple"),("loud","soft"),("fast","slow")]

def _get(obj: Any, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)

def _choose_contrast(rng: random.Random, prev: Optional[str]) -> str:
    if prev:
        for a,b in CONTRAST_PAIRS:
            if prev == a: return b
            if prev == b: return a
    a,b = rng.choice(CONTRAST_PAIRS)
    return rng.choice([a,b])

def _key(rng: random.Random, key_mode: Optional[str]) -> List[str]:
    if key_mode and isinstance(key_mode, str) and key_mode.strip():
        parts = key_mode.split()
        return parts[:2] if parts else [rng.choice(TONICS), rng.choice(MODES)]
    return [rng.choice(TONICS), rng.choice(MODES)]

def _swing(rng: random.Random) -> str:
    return f"{rng.randint(56,62)}%"

def _assemble(tokens: List[str], target_min=10, target_max=15) -> str:
    # Trim to <= target_max; if < target_min, just keep as is (model handles brevity).
    words = [t for t in tokens if t]
    if len(words) > target_max:
        words = words[:target_max]
    return " ".join(words)

def build(spec: Any, bars: int = 4, rng: Optional[random.Random] = None,
          intensity: float = 0.6, **context) -> Tuple[str, Dict]:
    rng = rng or random.Random()

    name     = _get(spec, "name", "Zone")
    species  = _get(spec, "species", "Unknown")
    biome    = _get(spec, "biome", "Unknown")
    bpm      = int(_get(spec, "bpm", 112) or 112)
    key_list = _key(rng, _get(spec, "key_mode", None))

    # Choose compact building blocks
    alien    = rng.choice(ALIEN)
    genre    = rng.choice(GENRES)
    bass     = rng.choice(BASS_PACKS)
    color    = rng.choice(COLOR_INSTR)
    groove   = rng.choice(["swung", "syncopated", "in-the-pocket"])
    swing    = _swing(rng)
    contrast = _choose_contrast(rng, context.get("neighbor_contrast_prev"))

    # Optional seasoning from context (only one short token to stay slim)
    ctx_tag = None
    if (context.get("festival")): ctx_tag = "festive"
    elif (context.get("time_of_day") in ("night","dusk")): ctx_tag = "nocturnal"
    elif (context.get("weather") and "storm" in str(context.get("weather")).lower()): ctx_tag = "stormy"

    # Compose ~10–15 words, highest-value info first
    tokens: List[str] = []
    tokens += [alien, genre]                  # 2
    tokens += bass                            # +2..3
    tokens += color                           # +1..2
    #tokens += [groove]               # +2
    tokens += [str(bpm), "BPM"]               # +2
    tokens += key_list                        # +2
    tokens += ["seamless", "loop"]            # +2
    if ctx_tag and len(tokens) < 14:
        tokens.append(ctx_tag)                # +1 (optional)

    prompt = _assemble(tokens, target_min=10, target_max=15)

    meta = {
        "schema": SCHEMA_VERSION,
        "zone_name": name,
        "species": species,
        "biome": biome,
        "bpm": bpm,
        "key": " ".join(key_list),
        "contrast": contrast,
        "alien_word": alien,
        "tokens": tokens,
    }
    return prompt, meta
