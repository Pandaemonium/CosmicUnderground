# promptgen.py
# Funk-forward prompt builder for Stable Audio Small (ToeJam & Earl vibes)
# Structure: Format → (alien tag) → Genre → Sub-genre → Instruments → Moods → Styles → BPM → Key → Extras
# Guardrails: clamps bars to <= ~11s, emits rich meta. Heavily biases toward funk/groove.
#
# Usage:
#   prompt, meta = promptgen.build(zone_spec, bars=zone_spec.bars, rng=random.Random(), intensity=0.55, **context)
#
# Optional context:
#   time_of_day, weather, heat, debt_pressure, festival, cosmic_event, neighbor_contrast_prev
#
# Notes:
# - Keeps one of ["alien", "weird", "extraterrestrial", "cosmic", ...] near the start of the prompt.
# - Funk bias is strong by default; you can tweak FUNK_BIAS if needed.

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import random

SCHEMA_VERSION = "promptgen/v3_funk"

# ---------------------------------------------------------------------------

ALIEN_WORDS = [
    "alien", "weird", "extraterrestrial", "cosmic", "otherworldly", "offworld", "xeno"
]

# Global funk dial (0..1). Higher = more instruments/styles/genres skew to funk/groove.
FUNK_BIAS = 0.9

# Funk-leaning contrast pairs; we’ll still flip for neighbor variety when asked.
CONTRAST_PAIRS: List[Tuple[str, str]] = [
    ("funky", "classic"),
    ("retro", "futuristic"),
    ("brassy", "woodwinds"),
    ("layered", "simple"),
    ("loud", "soft"),
    ("fast", "slow"),
]

# Genre pools: always funk-centric, species/biome add flavor on top.
FUNK_BASES = [
    "Funk", "P-Funk", "Boogie", "G-Funk", "Jazz-funk", "Electro-funk",
    "Hip-hop breaks", "UKG funk", "Disco-funk", "Neo-funk"
]
FUNK_SUBS = [
    "Acid funk", "Chrome boogie", "Velvet P-Funk", "Talkbox boogie",
    "Slime-funk", "Glitch-funk breaks", "Industrial funk parade",
    "Space-funk", "Retro arcade funk", "Analog slap-funk"
]

# Species→flavor nudges (still funk at heart)
GENRE_BY_SPECIES = {
    "Chillaxians": ("Funk / Downtempo", ["Chill-funk", "Ice boogie", "Glasshouse funk"]),
    "Glorpals": ("Slime-funk", ["Liquid boogie", "Wet funk breaks", "Rubber-bass funk"]),
    "Bzaris": ("Glitch-funk", ["IDM-funk", "Electro-funk breaks", "Neuro-boogie"]),
    "Shagdeliacs": ("P-Funk / Boogie", ["Jazz-funk", "Parade boogie", "Clav-funk"]),
    "Rockheads": ("Industrial-funk", ["Foundry funk", "Metallic boogie", "Gong-funk"]),
}

GENRE_BY_BIOME = {
    "Crystal Canyons": ("Glass-funk", ["Glacial boogie", "Crystal P-Funk"]),
    "Slime Caves": ("Slime-funk", ["Bio-boogie", "Acidic funk"]),
    "Centerspire": ("Arcade-funk", ["Electro-boogie", "Holo-funk breaks"]),
    "Groove Pits": ("House-funk", ["Jackin' boogie", "Bass-funk"]),
    "Polar Ice": ("Chill-funk", ["Arctic boogie"]),
    "Mag-Lev Yards": ("Rail-yard funk", ["Industrial boogie"]),
    "Electro Marsh": ("Garage-funk", ["2-step boogie"]),
}

# Core funk instrument lexicon (we always try to include at least 2–3 of these)
FUNK_CORE_INSTR = [
    "clavinet", "wah guitar", "slap bass", "rubber bass", "talkbox",
    "vocoder chops", "horn section", "congas", "rimshot kit",
    "707 drum machine", "808 subs", "909 hats", "cowbell",
    "analog brass", "phase Rhodes", "tape delay stabs"
]

EXTRA_INSTR = [
    "gel bells", "ice chimes", "bitcrush hats", "FM lead", "granular stutters",
    "kalimba", "woodblocks", "dub chords", "hand drums", "resonant zap",
    "deep toms", "metal gongs", "anvil hits", "space choir"
]

# Production/style terms with a groove bias.
PRODUCTION_TERMS = [
    "funky", "groove-forward", "swung", "syncopated", "in the pocket",
    "punchy", "tape-saturated", "compressed", "lofi", "gritty",
    "wet", "dry", "sproingy", "glassy", "mechanical", "magnetic",
    "slurpy", "fuzzy", "crystalline", "retro", "futuristic",
]

TONICS = ["A","B","C","D","E","F","G","A♭","B♭","C♯","D♯","F♯","G♯"]
MODES  = ["minor","Dorian","Mixolydian","major","Phrygian","Lydian","Aeolian"]  # minor/Dorian/Mixo first (funky)

# ---------------------------------------------------------------------------

def _get(obj: Any, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)

def _uniq_keep_order(xs: Iterable[str]) -> List[str]:
    seen = set(); out = []
    for x in xs:
        if not x or x in seen: continue
        seen.add(x); out.append(x)
    return out

def _beats_per_bar(timesig: Tuple[int,int]) -> int:
    return max(1, int(timesig[0] if isinstance(timesig, (tuple, list)) and len(timesig)>=1 else 4))

def _fit_bars_to_small(bpm: int, bars: int, timesig: Tuple[int,int], max_seconds: float = 11.0) -> Tuple[int, float]:
    beats = bars * _beats_per_bar(timesig)
    sec = beats * (60.0 / max(1, bpm))
    if sec <= max_seconds:
        return bars, sec
    per_bar = (60.0 / max(1, bpm)) * _beats_per_bar(timesig)
    max_bars = max(2, int(max_seconds // per_bar)) if per_bar > 0 else bars
    max_bars = max(2, min(bars, max_bars))
    beats = max_bars * _beats_per_bar(timesig)
    sec = beats * (60.0 / max(1, bpm))
    return max_bars, sec

def _choose_contrast(rng: random.Random, prefer_opposite_of: Optional[str]) -> str:
    if prefer_opposite_of:
        for a, b in CONTRAST_PAIRS:
            if prefer_opposite_of == a: return b
            if prefer_opposite_of == b: return a
    a, b = rng.choice(CONTRAST_PAIRS)
    return rng.choice([a, b])

def _derive_genre(species: Optional[str], biome: Optional[str], rng: random.Random) -> Tuple[str, str]:
    # Start from pure funk, let species/biome tint it.
    base = rng.choice(FUNK_BASES)
    sub  = rng.choice(FUNK_SUBS)
    sp_base, sp_subs = GENRE_BY_SPECIES.get(species or "", (None, None))
    if sp_base:
        # Blend by funk bias: mostly species flavor but never leave funk.
        if rng.random() < (0.65 * FUNK_BIAS):
            base = sp_base
            sub  = rng.choice(sp_subs)
    bio_base, bio_subs = GENRE_BY_BIOME.get(biome or "", (None, None))
    if bio_base and rng.random() < (0.45 * FUNK_BIAS):
        base = bio_base
        sub  = rng.choice(bio_subs)
    return base, sub

def _ensure_funk_core(pool: List[str], rng: random.Random) -> List[str]:
    # Always ensure 2–3 funk staples present.
    staples = rng.sample(FUNK_CORE_INSTR, k=3)
    merged = _uniq_keep_order(staples + pool)
    return merged

def _pick_instruments(spec_instruments: List[str], species: Optional[str], intensity: float, rng: random.Random) -> List[str]:
    # Species hints (still funkified)
    hints = {
        "Chillaxians": ["warm pads","ice chimes","soft bass","brush kit","chorus keys"],
        "Glorpals": ["squelch synth","gel bells","drip fx","rubber bass","wet claps"],
        "Bzaris": ["bitcrush blips","granular stutters","noisy hats","FM lead","resonant zap"],
        "Shagdeliacs": ["wah guitar","clavinet","horn section","rimshot kit","upright-ish synth bass"],
        "Rockheads": ["metal gongs","anvil hits","deep toms","clang perc","industrial pad"],
    }.get(species or "", [])

    # Build pool: user → species → extras; force funk staples to front.
    pool = _uniq_keep_order(list(spec_instruments or []) + hints + EXTRA_INSTR)
    pool = _ensure_funk_core(pool, rng)

    # More intensity = more layers
    k = max(3, min(6, int(3 + 2 * (intensity + FUNK_BIAS*0.5))))
    picks = rng.sample(pool, k=min(k, len(pool))) if len(pool) >= k else pool

    # High probability to include rhythmic section + bass + funk color
    def _maybe_add(x):
        if x not in picks and rng.random() < 0.7: picks.insert(0, x)

    _maybe_add(rng.choice(["707 drum machine","rimshot kit","909 hats"]))
    _maybe_add(rng.choice(["slap bass","rubber bass","808 subs"]))
    _maybe_add(rng.choice(["clavinet","wah guitar","talkbox","vocoder chops"]))

    # Trim to 4–6 instruments for clarity
    picks = _uniq_keep_order(picks)[:max(4, min(6, len(picks)))]
    return picks

def _pick_moods(mood: Optional[str], tags: List[str], descriptors: List[str], rng: random.Random, ctx: Dict) -> List[str]:
    out = []
    if mood: out.append(mood.strip().lower())
    # Funky defaults to keep the vibe recognizable
    base_funk = ["funky","playful","groovy","energetic","swagger"]
    out += rng.sample(base_funk, k=min(2, len(base_funk)))

    # Context seasoning
    tod = (ctx.get("time_of_day") or "").lower()
    weather = (ctx.get("weather") or "").lower()
    if tod in ("night","dusk"): out.append("nocturnal")
    if "storm" in weather: out.append("stormy")
    if "snow" in weather: out.append("icy")
    if ctx.get("festival"): out.append("festive")
    if ctx.get("cosmic_event"): out.append("cosmic")

    # Keep 3–4
    out = _uniq_keep_order(out)
    if len(out) > 4: out = rng.sample(out, k=4)
    return out

def _pick_styles(tags: List[str], descriptors: List[str], contrast: str, rng: random.Random) -> List[str]:
    groove_terms = ["swung","syncopated","in the pocket","ghost notes","brassy stabs","wah accents"]
    base = _uniq_keep_order(groove_terms + tags + descriptors + PRODUCTION_TERMS)
    picks = rng.sample(base, k=min(3, len(base))) if base else []
    if contrast not in picks: picks = ([contrast] + picks)[:4]
    # Guarantee at least one of the core groove terms
    if not any(t in picks for t in ["swung","syncopated","in the pocket"]):
        picks = ([rng.choice(["swung","syncopated","in the pocket"])] + picks)[:4]
    return _uniq_keep_order(picks)[:4]

def _coerce_key(key_mode: Optional[str], rng: random.Random) -> str:
    if key_mode and isinstance(key_mode, str) and key_mode.strip():
        return key_mode
    # Funk leans minor/Dorian/Mixo; we biased MODES order already.
    return f"{rng.choice(TONICS)} {rng.choice(MODES)}"

def _swing_amount(rng: random.Random) -> int:
    # Typical funk/boogie shuffle ~ 56–62%
    return rng.randint(56, 62)

def _format_field(name: str, value: str) -> str:
    return f"{name}: {value}"

# ---------------------------------------------------------------------------

def build(spec: Any, bars: int = 4, rng: Optional[random.Random] = None, intensity: float = 0.6, **context) -> Tuple[str, Dict]:
    """
    Build a funk-forward Stable Audio Small prompt + meta.
    """
    rng = rng or random.Random()

    species     = _get(spec, "species", "Unknown")
    biome       = _get(spec, "biome", "Unknown biome")
    mood        = _get(spec, "mood", "funky")
    bpm         = int(_get(spec, "bpm", 112) or 112)  # 100–124 is a sweet funk pocket; keep caller’s bpm though.
    key_mode    = _coerce_key(_get(spec, "key_mode", None), rng)
    timesig     = _get(spec, "timesig", (4,4))
    descs       = list(_get(spec, "descriptors", []) or [])
    instr_src   = list(_get(spec, "instruments", []) or [])
    tags        = list(_get(spec, "tags", []) or [])
    name        = _get(spec, "name", "Unknown Zone")

    # Contrast alternation for neighbor variety
    contrast = _choose_contrast(rng, context.get("neighbor_contrast_prev"))

    # Instruments / moods / styles
    instruments = _pick_instruments(instr_src, species, intensity, rng)
    moods       = _pick_moods(mood, tags, descs, rng, context)
    styles      = _pick_styles(tags, descs, contrast, rng)

    # Funk genre mapping
    genre, subgenre = _derive_genre(species, biome, rng)

    # Format: prefer Band/Ensemble for funk layers
    fmt = "Ensemble" if len(instruments) >= 5 else "Band"

    # Bars fit for Small
    bars_fitted, loop_seconds = _fit_bars_to_small(bpm=bpm, bars=bars, timesig=timesig, max_seconds=11.0)

    # Near-start alien word
    alien_word = rng.choice(ALIEN_WORDS)

    # Swing %
    swing_pct = _swing_amount(rng)

    # Assemble (order matters)
    fields = [
        _format_field("Format", fmt),
        alien_word,
        _format_field("Genre", genre),
        _format_field("Sub-genre", subgenre),
        _format_field("Instruments", ", ".join(instruments)),
        _format_field("Moods", ", ".join(moods)),
        _format_field("Styles", ", ".join(styles)),
        _format_field("BPM", str(bpm)),
        _format_field("Key", key_mode),
        _format_field(
            "Extras",
            # Loop hygiene + groove directives
            f"loopable {bars_fitted} bars, seamless loop, clean downbeat, minimal silence at edges, "
            f"no vocals, tight low end, swung {swing_pct}%, syncopated bassline, brassy stabs, wah accents"
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
        "swing_percent": swing_pct,
        "funk_bias": FUNK_BIAS,
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
