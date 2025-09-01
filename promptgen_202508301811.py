# promptgen.py
# Funk-forward prompt builder for Stable Audio Small with extra BASS focus
# Structure: Format → (alien tag) → Genre → Sub-genre → Instruments → Moods → Styles → BPM → Key → Extras
# Bars are clamped to ~11s (Stable Audio Small). Heavily biases toward groove + strong bassline.
#
# Usage:
#   prompt, meta = promptgen.build(zone_spec, bars=zone_spec.bars, rng=random.Random(), intensity=0.6, **context)
# Optional context:
#   time_of_day, weather, heat, debt_pressure, festival, cosmic_event, neighbor_contrast_prev
#
# New knobs:
#   - bass_focus (0..1): how aggressively to push bass instruments & directives (default 0.9)
#   - bass_profile: None | "slap" | "rubber" | "synth" | "talkbox" | "sub"
#       If None, chosen procedurally from species/biome + RNG.

from typing import Any, Dict, Iterable, List, Optional, Tuple
import random
import math

SCHEMA_VERSION = "promptgen/v4_funk_bass"

# ---------------------------------------------------------------------------

ALIEN_WORDS = ["alien", "weird", "extraterrestrial", "cosmic", "otherworldly", "offworld", "xeno"]

FUNK_BIAS = 0.9  # overall funk lean

CONTRAST_PAIRS: List[Tuple[str, str]] = [
    ("funky", "classic"),
    ("retro", "futuristic"),
    ("brassy", "woodwinds"),
    ("layered", "simple"),
    ("loud", "soft"),
    ("fast", "slow"),
]

FUNK_BASES = [
    "Funk", "P-Funk", "Boogie", "G-Funk", "Jazz-funk", "Electro-funk",
    "Hip-hop breaks", "UKG funk", "Disco-funk", "Neo-funk", "Dance", "Party"
]
FUNK_SUBS = [
    "Acid funk", "Chrome boogie", "Velvet P-Funk", "Talkbox boogie",
    "Slime-funk", "Glitch-funk breaks", "Industrial funk parade",
    "Space-funk", "Retro arcade funk", "Analog slap-funk"
]

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

# Core funk instruments
FUNK_CORE_INSTR = [
    "clavinet", "wah guitar", "horn section", "congas", "rimshot kit",
    "707 drum machine", "808 subs", "909 hats", "cowbell",
    "analog brass", "phase Rhodes", "tape delay stabs", "vocoder chops"
]

EXTRA_INSTR = [
    "gel bells", "ice chimes", "bitcrush hats", "FM lead", "granular stutters",
    "kalimba", "woodblocks", "dub chords", "hand drums", "resonant zap",
    "deep toms", "metal gongs", "anvil hits", "space choir", "talkbox"
]

# Bass-centric instrument pools (we’ll insert these prominently)
BASS_INSTR_PROFILES = {
    "slap":  ["slap bass", "octave pops", "muted string plucks"],
    "rubber":["rubber bass", "envelope filter bass", "pluck bass"],
    "synth": ["analog synth bass", "moog bass", "FM bass"],
    "talkbox": ["talkbox bass", "vocoder bass"],
    "sub":   ["808 subs", "sub-bass", "low sine bass"],
}

# Added style terms with bass focus
GROOVE_TERMS = ["swung", "syncopated", "in the pocket", "ghost notes", "octave jumps", "percussive mutes", "slides"]

PRODUCTION_TERMS = [
    "funky", "groove-forward", "punchy", "tape-saturated", "compressed", "lofi", "gritty",
    "wet", "dry", "sproingy", "glassy", "mechanical", "magnetic", "slurpy", "fuzzy", "crystalline",
    "retro", "futuristic"
]

TONICS = ["A","B","C","D","E","F","G","A♭","B♭","C♯","D♯","F♯","G♯"]
MODES  = ["minor","Dorian","Mixolydian","major","Phrygian","Lydian","Aeolian"]  # funk-first ordering

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
    base = rng.choice(FUNK_BASES); sub  = rng.choice(FUNK_SUBS)
    sp_base, sp_subs = GENRE_BY_SPECIES.get(species or "", (None, None))
    if sp_base and rng.random() < (0.65 * FUNK_BIAS):
        base = sp_base; sub = rng.choice(sp_subs)
    bio_base, bio_subs = GENRE_BY_BIOME.get(biome or "", (None, None))
    if bio_base and rng.random() < (0.45 * FUNK_BIAS):
        base = bio_base; sub = rng.choice(bio_subs)
    return base, sub

def _pick_bass_profile(species: Optional[str], biome: Optional[str], rng: random.Random, user_profile: Optional[str]) -> str:
    if user_profile in BASS_INSTR_PROFILES: return user_profile
    # Gentle mapping: Rockheads → sub/synth, Shagdeliacs → slap/talkbox, Glorpals → rubber/sub, Bzaris → synth, Chillaxians → rubber/synth
    candidates = {
        "Rockheads": ["sub","synth"],
        "Shagdeliacs": ["slap","talkbox"],
        "Glorpals": ["rubber","sub"],
        "Bzaris": ["synth","talkbox"],
        "Chillaxians": ["rubber","synth"],
    }.get(species or "", ["slap","rubber","synth","talkbox","sub"])
    return rng.choice(candidates)

def _ensure_funk_core(pool: List[str], rng: random.Random) -> List[str]:
    staples = rng.sample(FUNK_CORE_INSTR, k=3)
    return _uniq_keep_order(staples + pool)

def _pick_instruments(spec_instruments: List[str], species: Optional[str], intensity: float,
                      rng: random.Random, bass_profile: str, bass_focus: float) -> List[str]:
    hints = {
        "Chillaxians": ["warm pads","ice chimes","soft bass","brush kit","chorus keys"],
        "Glorpals":    ["squelch synth","gel bells","drip fx","wet claps"],
        "Bzaris":      ["bitcrush blips","granular stutters","noisy hats","FM lead"],
        "Shagdeliacs": ["wah guitar","clavinet","horn section","rimshot kit"],
        "Rockheads":   ["metal gongs","anvil hits","deep toms","clang perc","industrial pad"],
    }.get(species or "", [])

    pool = _uniq_keep_order(list(spec_instruments or []) + hints + EXTRA_INSTR)
    pool = _ensure_funk_core(pool, rng)

    # Strongly inject bass profile instruments to the front
    bass_pack = BASS_INSTR_PROFILES.get(bass_profile, ["rubber bass"])
    # Merge with priority, avoid dups
    pool = _uniq_keep_order(bass_pack + pool)

    # More intensity = more layers
    k = max(4, min(6, int(4 + 2 * (intensity + FUNK_BIAS*0.5))))
    picks = rng.sample(pool, k=min(k, len(pool))) if len(pool) >= k else pool

    # Guarantee rhythm section & bass core
    def _prioritize(x, p=0.8):
        if x not in picks and rng.random() < p: picks.insert(0, x)

    _prioritize(rng.choice(["707 drum machine","rimshot kit","909 hats"]), 0.9)
    # Bass presence scaled by bass_focus
    if rng.random() < (0.6 + 0.4 * bass_focus): _prioritize(rng.choice(bass_pack), 1.0)
    if rng.random() < (0.5 + 0.4 * bass_focus): _prioritize(rng.choice(["slap bass","rubber bass","analog synth bass","moog bass","808 subs"]), 1.0)

    return _uniq_keep_order(picks)[:6]

def _pick_moods(mood: Optional[str], tags: List[str], descriptors: List[str],
                rng: random.Random, ctx: Dict) -> List[str]:
    out = []
    if mood: out.append(mood.strip().lower())
    base_funk = ["funky","playful","groovy","energetic","swagger"]
    out += rng.sample(base_funk, k=min(2, len(base_funk)))
    tod = (ctx.get("time_of_day") or "").lower()
    weather = (ctx.get("weather") or "").lower()
    if tod in ("night","dusk"): out.append("nocturnal")
    if "storm" in weather: out.append("stormy")
    if "snow" in weather: out.append("icy")
    if ctx.get("festival"): out.append("festive")
    if ctx.get("cosmic_event"): out.append("cosmic")
    out = _uniq_keep_order(out)
    if len(out) > 4: out = rng.sample(out, k=4)
    return out

def _pick_styles(tags: List[str], descriptors: List[str], contrast: str,
                 rng: random.Random, bass_focus: float) -> List[str]:
    base = _uniq_keep_order(GROOVE_TERMS + tags + descriptors + PRODUCTION_TERMS)
    picks = rng.sample(base, k=min(3, len(base))) if base else []
    if contrast not in picks: picks = ([contrast] + picks)[:4]
    # Ensure at least one bass behavior term
    bass_terms = ["ghost notes","octave jumps","percussive mutes","slides"]
    if rng.random() < (0.6 + 0.35 * bass_focus) and not any(t in picks for t in bass_terms):
        picks = ([rng.choice(bass_terms)] + picks)[:4]
    # Ensure a core groove term
    if not any(t in picks for t in ["swung","syncopated","in the pocket"]):
        picks = ([rng.choice(["swung","syncopated","in the pocket"])] + picks)[:4]
    return _uniq_keep_order(picks)[:4]

def _coerce_key(key_mode: Optional[str], rng: random.Random) -> str:
    if key_mode and isinstance(key_mode, str) and key_mode.strip():
        return key_mode
    return f"{rng.choice(TONICS)} {rng.choice(MODES)}"

def _swing_amount(rng: random.Random) -> int:
    return rng.randint(56, 62)  # classic funky shuffle range

def _bassline_directives(profile: str, swing_pct: int, bass_focus: float, rng: random.Random) -> str:
    # Tailored instructions the model tends to respect
    common = [
        f"prominent bassline, swung {swing_pct}%",
        "syncopated groove",
        "tight note lengths",
        "call-and-response with drums",
    ]
    finesse = ["ghost notes", "octave pops", "slides", "percussive muting"]
    if bass_focus > 0.7:
        common += rng.sample(finesse, k=2)
    else:
        common += rng.sample(finesse, k=1)

    if profile == "slap":
        flavor = ["slap articulation", "thumb pops", "muted percussive plucks"]
    elif profile == "rubber":
        flavor = ["rubbery envelope plucks", "auto-wah movement", "light filter sweep"]
    elif profile == "synth":
        flavor = ["analog filter plucks", "short decay, no smear", "sub support without boom"]
    elif profile == "talkbox":
        flavor = ["talkbox vowels on bass notes", "tight phrasing", "no long tail"]
    elif profile == "sub":
        flavor = ["sub-focused pattern", "sidechain feel with kick", "avoid mud above 120Hz"]
    else:
        flavor = ["groovy bass phrasing"]

    return ", ".join(_uniq_keep_order(common + flavor))

def _format_field(name: str, value: str) -> str:
    return f"{name}: {value}"

# ---------------------------------------------------------------------------

def build(spec: Any, bars: int = 4, rng: Optional[random.Random] = None, intensity: float = 0.6,
          bass_focus: float = 0.9, bass_profile: Optional[str] = None, **context) -> Tuple[str, Dict]:
    """
    Build a funk-forward prompt with extra emphasis on basslines.
    """
    rng = rng or random.Random()

    species     = _get(spec, "species", "Unknown")
    biome       = _get(spec, "biome", "Unknown biome")
    mood        = _get(spec, "mood", "funky")
    bpm         = int(_get(spec, "bpm", 112) or 112)
    key_mode    = _coerce_key(_get(spec, "key_mode", None), rng)
    timesig     = _get(spec, "timesig", (4,4))
    descs       = list(_get(spec, "descriptors", []) or [])
    instr_src   = list(_get(spec, "instruments", []) or [])
    tags        = list(_get(spec, "tags", []) or [])
    name        = _get(spec, "name", "Unknown Zone")

    contrast = _choose_contrast(rng, context.get("neighbor_contrast_prev"))

    chosen_profile = _pick_bass_profile(species, biome, rng, bass_profile)
    instruments = _pick_instruments(instr_src, species, intensity, rng, chosen_profile, bass_focus)
    moods       = _pick_moods(mood, tags, descs, rng, context)
    styles      = _pick_styles(tags, descs, contrast, rng, bass_focus)

    genre, subgenre = _derive_genre(species, biome, rng)
    fmt = "Ensemble" if len(instruments) >= 5 else "Band"

    bars_fitted, loop_seconds = _fit_bars_to_small(bpm=bpm, bars=bars, timesig=timesig, max_seconds=11.0)
    alien_word = rng.choice(ALIEN_WORDS)
    swing_pct = _swing_amount(rng)

    bass_directives = _bassline_directives(chosen_profile, swing_pct, bass_focus, rng)

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
            f"loopable {bars_fitted} bars, seamless loop, clean downbeat, minimal silence at edges, "
            f"no vocals, tight low end, {bass_directives}"
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
        "bass": {
            "focus": round(bass_focus, 2),
            "profile": chosen_profile,
            "directives": bass_directives,
        },
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
