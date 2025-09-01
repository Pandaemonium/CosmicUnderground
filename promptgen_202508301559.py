"""
promptgen.py — Alien DJ prompt engine
-------------------------------------
Pure, reloadable prompt builder for Stable Audio Small.
- Accepts either a dict or any object with attributes (duck-typed "spec")
- Handles NPCs, Objects, Locations, and Event/Stingers
- Applies climate/species/biome flavor, mood/time/weather/heat modifiers
- Ensures compact, loop-safe prompts with hygiene constraints

Usage (in-game):
    import promptgen, importlib, random
    # Hot reload on F5:
    importlib.reload(promptgen)
    prompt = promptgen.build_prompt_from_spec(zone, bars=zone.bars, rng=random.Random(), intensity=0.55)

Optional richer call:
    prompt, meta = promptgen.build(zone, bars=8, rng=random.Random(123), intensity=0.6)

Copyright:
    You can copy/paste/modify freely inside your project.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
import random

SCHEMA_VERSION = "pg/2.0.0"

# ----------------- Core catalogs -----------------

# Contrast pairs (we will pick exactly ONE side, never both)
CONTRAST_PAIRS: List[Tuple[str, str]] = [
    ("fast", "slow"),
    ("funky", "classic"),
    ("brassy", "woodwinds"),
    ("loud", "soft"),
    ("layered", "simple"),
    ("retro", "futuristic"),
]

# Species → instrument/textures + typical BPM lanes
SPECIES_POOLS: Dict[str, Dict[str, Any]] = {
    "Chillaxians": dict(
        bpm=(84, 102),
        instruments=["warm pads", "ice chimes", "soft bass", "brush kit", "chorus keys"],
        textures=["calm", "retro", "soft", "crystalline", "airy"],
    ),
    "Glorpals": dict(
        bpm=(96, 122),
        instruments=["squelch synth", "gel bells", "drip fx", "rubber bass", "wet claps"],
        textures=["wet", "slurpy", "layered", "sproingy"],
    ),
    "Bzaris": dict(
        bpm=(120, 140),
        instruments=["bitcrush blips", "granular stutters", "noisy hats", "FM lead", "resonant zap"],
        textures=["fast", "glitchy", "futuristic", "electric"],
    ),
    "Shagdeliacs": dict(
        bpm=(96, 120),
        instruments=["wah guitar", "clavinet", "horn section", "rimshot kit", "upright-ish synth bass"],
        textures=["funky", "brassy", "jazzy", "retro", "hairy", "furry"],
    ),
    "Rockheads": dict(
        bpm=(100, 126),
        instruments=["metal gongs", "anvil hits", "deep toms", "clang perc", "industrial pad"],
        textures=["mechanical", "metallic", "loud", "magnetic"],
    ),
}

# Biome → climate guardrails (mode palettes, tempo lanes, ambience)
CLIMATE: Dict[str, Dict[str, Any]] = {
    "Crystal Canyons": dict(bpm=(118, 132), mode=["Lydian", "Mixolydian", "major"], ambience=["crystalline", "airy", "shimmering"]),
    "Slime Caves":     dict(bpm=(110, 124), mode=["Dorian", "Aeolian", "minor"],   ambience=["wet", "slurpy", "dubby"]),
    "Centerspire":     dict(bpm=(120, 134), mode=["Aeolian", "Dorian", "Mixolydian"], ambience=["futuristic", "glossy", "sleek"]),
    "Groove Pits":     dict(bpm=(100, 120), mode=["Dorian", "Mixolydian", "minor"], ambience=["funky", "brassy", "classic"]),
    "Polar Ice":       dict(bpm=(84, 100),  mode=["Lydian", "major", "Mixolydian"], ambience=["calm", "soft", "retro"]),
    "Mag-Lev Yards":   dict(bpm=(112, 128), mode=["Phrygian", "Aeolian", "Dorian"], ambience=["mechanical", "magnetic", "industrial"]),
    "Electro Marsh":   dict(bpm=(116, 126), mode=["Dorian", "Mixolydian", "minor"], ambience=["wet", "electric", "layered"]),
}

# Entity base templates: default bars, density, loop notes, & role presets
ENTITY_BASE: Dict[str, Dict[str, Any]] = {
    "npc": dict(
        bars=8,
        roles={
            # role → (instruments extra, textures extra, tag bias)
            "busker":     (["clavinet", "wah guitar", "rimshot kit"], ["funky", "playful"], "funky"),
            "merchant":   (["Rhodes", "congas", "upright-ish synth bass"], ["mellow", "retro"], "classic"),
            "enforcer":   (["deep toms", "anvil hits", "metal gongs"], ["loud", "mechanical"], "fast"),
            "elder":      (["choir pads", "glass mallets"], ["soft", "crystalline"], "slow"),
            "swarm":      (["bitcrush blips", "granular stutters"], ["glitchy", "electric"], "fast"),
        },
        loop_notes="tight downbeat; short tails",
    ),
    "object": dict(
        bars=4,
        kinds={
            "crystal":   (["bell arps", "glass mallets"], ["crystalline", "airy"],   "futuristic"),
            "slime":     (["gel bells", "rubber bass", "drip perc"], ["wet", "slurpy"], "layered"),
            "pylon":     (["pulse train", "tape echo ticks"], ["mechanical", "magnetic"], "retro"),
            "gong_gate": (["metal gongs", "deep toms"], ["loud", "metallic"], "brassy"),
            "arcade":    (["8-step arps", "ping blips"], ["retro", "bright"], "retro"),
            "fungal":    (["kalimba", "hand drums", "woodblocks"], ["organic", "fuzzy"], "classic"),
        },
        loop_notes="percussion-forward; minimal reverb tail; no vocal words",
    ),
    "location": dict(
        bars=8,
        kinds={
            "market": (["gamelan plucks", "elastic bass", "clicky hats"], ["glossy", "layered"], "futuristic"),
            "cavern": (["dubby chords", "sub swells"], ["wet", "spacious"], "simple"),
            "parade": (["horn stabs", "clav comp"], ["brassy", "energetic"], "funky"),
            "tundra": (["warm pads", "ice chimes"], ["calm", "soft"], "slow"),
        },
        loop_notes="ambient bed; unobtrusive; mono sub under 120Hz",
    ),
    "event": dict(
        bars=4,
        kinds={
            "stinger": (["tom fill", "resonant zap"], ["loud", "electric"], "fast"),
            "discovery": (["bell flourish", "choir swell"], ["crystalline", "triumphant"], "classic"),
            "alert": (["metal hits", "pulse train"], ["mechanical", "urgent"], "fast"),
        },
        loop_notes="one clean fill then settle; short tails",
    ),
}

# Extras the generator may sprinkle in
EXTRA_INSTR = [
    "kalimba", "woodblocks", "808 subs", "tape echo stabs", "analog brass",
    "vocoder chops", "hand drums", "dub chords", "arpeggiator", "space choir",
]
EXTRA_TEXTS = [
    "crystalline", "laser", "electric", "mechanical", "nuclear",
    "magnetic", "chemical", "fuzzy", "hair-clad", "hirsute", "sproingy",
]

TONICS = ["A","B","C","D","E","F","G","A♭","B♭","C♯","D♯","F♯","G♯"]
MODES  = ["major","minor","Dorian","Phrygian","Mixolydian","Lydian","Aeolian"]

# ----------------- Helpers -----------------

def _get(spec: Any, key: str, default=None):
    """Fetch attr or dict key."""
    if hasattr(spec, key):
        return getattr(spec, key)
    if isinstance(spec, dict) and key in spec:
        return spec[key]
    return default

def _uniq(seq: Iterable[str]) -> List[str]:
    """Keep order, remove dupes, ignore falsy."""
    seen, out = set(), []
    for s in seq:
        if not s: continue
        if s in seen: continue
        seen.add(s); out.append(s)
    return out

def _rand(rng: random.Random, seq: List[str]) -> str:
    return rng.choice(seq) if seq else ""

def _pick_contrast(rng: random.Random, prefer: Optional[str] = None) -> str:
    """
    Pick exactly one side from the contrast pairs.
    If prefer is given (e.g., 'retro'), bias toward that side when present.
    """
    if prefer:
        for a, b in CONTRAST_PAIRS:
            if prefer == a or prefer == b:
                # choose preferred with 70% prob
                return prefer if rng.random() < 0.7 else (b if prefer == a else a)
    a, b = rng.choice(CONTRAST_PAIRS)
    return a if rng.random() < 0.5 else b

def _apply_jitter(rng: random.Random, base: int, lo: int, hi: int, step: int = 2) -> int:
    """Clamp to [lo,hi], add small jitter."""
    j = rng.choice([-4, -2, 0, 2, 4])
    x = int(max(lo, min(hi, base + j)))
    # snap to step grid
    if step > 1:
        x -= (x - lo) % step
    return x

def _mode_from_palette(rng: random.Random, palette: List[str]) -> str:
    return _rand(rng, palette or MODES)

def _merge_texts(*groups: Iterable[str]) -> List[str]:
    return _uniq([*groups[0], *groups[1], *groups[2]]) if len(groups) >= 3 else _uniq([s for g in groups for s in g])

def _cap_words(words: List[str], max_n: int) -> List[str]:
    return words[:max_n] if len(words) > max_n else words

def _shorten_chars(s: str, max_chars: int = 400) -> str:
    return s if len(s) <= max_chars else (s[:max_chars-1] + "…")

# ----------------- Layer logic -----------------

def _climate_for_biome(biome: str) -> Dict[str, Any]:
    return CLIMATE.get(biome, dict(bpm=(112, 124), mode=["Dorian", "minor", "Mixolydian"], ambience=[]))

def _entity_defaults(entity_type: str, role_or_kind: Optional[str]) -> Dict[str, Any]:
    base = ENTITY_BASE.get(entity_type, ENTITY_BASE["location"])
    out = dict(bars=base["bars"], loop_notes=base.get("loop_notes", ""))
    table = base.get("roles" if entity_type == "npc" else "kinds", {})
    if role_or_kind and role_or_kind in table:
        inst, texts, bias = table[role_or_kind]
        out.update(role_instruments=inst, role_textures=texts, role_bias=bias)
    else:
        out.update(role_instruments=[], role_textures=[], role_bias=None)
    return out

def _species_block(species: str) -> Dict[str, Any]:
    return SPECIES_POOLS.get(species, dict(bpm=(112,124), instruments=[], textures=[]))

def _mood_block(mood: str) -> Dict[str, Any]:
    mood = (mood or "").lower()
    # mode palette + texture hints + dynamics tag bias
    if mood in ("angry", "hostile", "aggressive"):
        return dict(mode_bias=["Aeolian", "Phrygian", "minor"], textures=["loud","mechanical"], bias="fast")
    if mood in ("energetic", "excited"):
        return dict(mode_bias=["Mixolydian","Dorian","major"], textures=["bouncy","crisp"], bias="fast")
    if mood in ("calm", "serene", "chill"):
        return dict(mode_bias=["Lydian","major","Mixolydian"], textures=["soft","airy"], bias="slow")
    if mood in ("melancholy", "sad"):
        return dict(mode_bias=["Aeolian","minor","Dorian"], textures=["hazy","retro"], bias="simple")
    if mood in ("triumphant", "heroic"):
        return dict(mode_bias=["Mixolydian","major","Lydian"], textures=["brassy","bright"], bias="loud")
    if mood in ("playful"):
        return dict(mode_bias=["Dorian","Mixolydian","major"], textures=["funky","quirky"], bias="funky")
    # default
    return dict(mode_bias=[], textures=[], bias=None)

def _time_weather_block(time_of_day: Optional[str], weather: Optional[str]) -> Dict[str, Any]:
    t = (time_of_day or "").lower()
    w = (weather or "").lower()
    texts: List[str] = []
    bpm_delta = 0
    decay = "short tails"

    if t in ("dawn", "sunrise"):
        texts += ["warm", "gentle", "soft"]
    elif t in ("dusk", "sunset"):
        texts += ["tape haze", "mellow highs"]
    elif t in ("night",):
        texts += ["wider stereo", "chorus"]

    if w in ("snow",):
        texts += ["frosty shimmer", "softened transients"]
    elif w in ("rain", "light rain", "drizzle"):
        texts += ["raindrop perc", "filtered hats"]
    elif w in ("storm", "magnetic storm", "solar storm"):
        texts += ["geomagnetic howl", "sub swells"]; bpm_delta += 2

    return dict(textures=texts, bpm_delta=bpm_delta, decay=decay)

def _narrative_block(heat: float = 0.0, debt: float = 0.0, festival: bool = False, cosmic: bool = False) -> Dict[str, Any]:
    texts: List[str] = []
    bias: Optional[str] = None
    if heat > 0.5:
        texts += ["crunchy", "urgent"]
        bias = "fast"
    if debt > 0.5:
        texts += ["mechanical", "metallic"]
    if festival:
        texts += ["parade", "brassy"]
        bias = bias or "funky"
    if cosmic:
        texts += ["electric", "magnetic"]
    return dict(textures=texts, bias=bias)

# ----------------- Public API -----------------

def build(spec: Any,
          *,
          entity_type: Optional[str] = None,
          role_or_kind: Optional[str] = None,
          bars: Optional[int] = None,
          rng: Optional[random.Random] = None,
          intensity: float = 0.55,
          neighbor_contrast_prev: Optional[str] = None,
          time_of_day: Optional[str] = None,
          weather: Optional[str] = None,
          heat: float = 0.0,
          debt_pressure: float = 0.0,
          festival: bool = False,
          cosmic_event: bool = False) -> Tuple[str, Dict[str, Any]]:
    """
    Build a prompt and return (prompt_text, meta_dict).

    Inputs:
      spec: dict or object with fields:
        name, species, biome, mood, bpm (opt), key_mode (opt),
        descriptors (list), instruments (list), tags (list)
      entity_type: 'npc' | 'object' | 'location' | 'event' (default inferred: npc if actor-like, else location)
      role_or_kind: e.g., 'busker' (npc) or 'crystal' (object) or 'market' (location) or 'stinger' (event)
      bars: override base bars (default from entity type)
      intensity: 0..1 controls instrument & texture count
      neighbor_contrast_prev: previous contrast tag (e.g., 'retro') to bias variety
      time_of_day, weather, heat, debt_pressure, festival, cosmic_event: modifiers

    Returns:
      prompt text + meta (schema/version + all choices).
    """
    rng = rng or random.Random()

    # Read base fields
    name      = _get(spec, "name", "Unknown Zone")
    species   = _get(spec, "species", "")
    biome     = _get(spec, "biome", "Unknown")
    mood      = _get(spec, "mood", "energetic")
    bpm_in    = _get(spec, "bpm", None)
    key_in    = _get(spec, "key_mode", None)
    given_desc= _get(spec, "descriptors", []) or []
    given_inst= _get(spec, "instruments", []) or []
    given_tags= _get(spec, "tags", []) or []

    # Figure entity type default
    et = (entity_type or _get(spec, "entity_type", None) or "location").lower()
    if et not in ENTITY_BASE: et = "location"
    base = _entity_defaults(et, role_or_kind)
    bars_final = bars or base["bars"]

    # Climate + species guardrails
    climate = _climate_for_biome(biome)
    sp      = _species_block(species)

    # Mood and time/weather layers
    mood_blk = _mood_block(mood)
    tw_blk   = _time_weather_block(time_of_day, weather)
    nar_blk  = _narrative_block(heat, debt_pressure, festival, cosmic_event)

    # BPM decision
    climate_lo, climate_hi = climate["bpm"]
    sp_lo, sp_hi = sp["bpm"]
    # Start from: explicit bpm > species lane > climate lane > 120
    base_bpm = bpm_in or int((sp_lo + sp_hi) / 2) if species else int((climate_lo + climate_hi) / 2)
    # Apply deltas & jitter, clamp within intersection-ish (soft)
    base_bpm += tw_blk["bpm_delta"]
    lo, hi = min(climate_lo, sp_lo), max(climate_hi, sp_hi)
    bpm = _apply_jitter(rng, base_bpm, lo, hi, step=2)
    bpm = max(84, min(140, bpm))

    # Key/mode decision
    if key_in:
        key_mode = key_in
    else:
        palette = climate["mode"]
        if mood_blk["mode_bias"]:
            # Blend mood bias with climate
            palette = _uniq([*mood_blk["mode_bias"], *palette])
        tonic = _rand(rng, TONICS)
        mode  = _mode_from_palette(rng, palette)
        key_mode = f"{tonic} {mode}"

    # Contrast choice (bias away from the previous tile's side)
    prefer_side = None
    if neighbor_contrast_prev:
        # Pick the opposite side with 70% probability for variety
        for a, b in CONTRAST_PAIRS:
            if neighbor_contrast_prev == a: prefer_side = b
            if neighbor_contrast_prev == b: prefer_side = a
    # Blend with role/mood/narrative biases
    prefer_side = prefer_side or base.get("role_bias") or mood_blk.get("bias") or nar_blk.get("bias")
    contrast_side = _pick_contrast(rng, prefer=prefer_side)

    # Assemble instruments & textures
    role_inst = base.get("role_instruments", [])
    sp_inst   = sp.get("instruments", [])
    inst_pool = _uniq([*sp_inst, *role_inst, *given_inst, *EXTRA_INSTR])
    k_instr   = max(2, min(4, 2 + int(2 * intensity)))
    instruments = rng.sample(inst_pool, k=min(k_instr, len(inst_pool))) if inst_pool else []

    role_tex = base.get("role_textures", [])
    sp_tex   = sp.get("textures", [])
    climate_tex = climate.get("ambience", [])
    mood_tex = mood_blk.get("textures", [])
    tw_tex   = tw_blk.get("textures", [])
    nar_tex  = nar_blk.get("textures", [])
    tex_pool = _uniq([*sp_tex, *climate_tex, *role_tex, *mood_tex, *tw_tex, *nar_tex, *given_desc, *EXTRA_TEXTS])
    k_tex    = max(2, min(3, 2 + int(1 * intensity)))
    textures = rng.sample(tex_pool, k=min(k_tex, len(tex_pool))) if tex_pool else []

    # Style tags: exactly one contrast side + one extra texture
    style_tags = _uniq([contrast_side, textures[0] if textures else ""])
    # Ensure no contradictions (we never add both sides by design)

    # Entity context sentence (kept short)
    if et == "npc":
        entity_ctx = f"{species or 'local'} {('(' + role_or_kind + ')') if role_or_kind else ''}".strip()
        context = f"in the {biome}, {entity_ctx}".replace("  ", " ")
    elif et == "object":
        context = f"in the {biome}, {role_or_kind or 'resonant object'}"
    elif et == "event":
        context = f"over {biome}, {role_or_kind or 'event'}"
    else:  # location
        context = f"in the {biome}"

    # Compose lines (compact)
    texture_line = ", ".join(textures[:3])
    instr_line   = ", ".join(instruments[:4])
    style_line   = ", ".join(style_tags[:2])

    # Loop hygiene
    loop_contract = (
        f"Loopable {bars_final} bars, {bpm} BPM, {key_mode}. "
        f"Clean downbeat; seamless loop; minimal silence at edges; short reverb tails; mono sub under 120Hz."
    )
    # Add entity-specific loop notes
    notes = ENTITY_BASE[et].get("loop_notes", "")
    if notes:
        loop_contract = f"{loop_contract} {notes}."

    # Final prompt (2–3 short sentences + contract)
    body = (
        f"{mood.capitalize()} vibes {context}. "
        f"{texture_line}. " + (f"Instruments: {instr_line}. " if instr_line else "") +
        (f"Style tags: {style_line}. " if style_line else "")
    )
    prompt = _shorten_chars(body + loop_contract, max_chars=420)

    meta = dict(
        schema=SCHEMA_VERSION,
        entity_type=et,
        role_or_kind=role_or_kind,
        name=name,
        species=species,
        biome=biome,
        mood=mood,
        bpm=bpm,
        key_mode=key_mode,
        bars=bars_final,
        contrast=contrast_side,
        instruments=instruments,
        textures=textures,
        style_tags=style_tags,
        time_of_day=time_of_day,
        weather=weather,
        heat=heat,
        debt_pressure=debt_pressure,
        festival=festival,
        cosmic_event=cosmic_event,
    )
    return prompt, meta

def build_prompt_from_spec(spec: Any,
                           bars: Optional[int] = 8,
                           rng: Optional[random.Random] = None,
                           intensity: float = 0.55,
                           **kwargs) -> str:
    """
    Back-compat simple builder. Delegates to build(...), returns only the prompt text.
    kwargs can include: entity_type, role_or_kind, time_of_day, weather,
                        neighbor_contrast_prev, heat, debt_pressure, festival, cosmic_event
    """
    prompt, _ = build(spec, bars=bars, rng=rng, intensity=intensity, **kwargs)
    return prompt

# --------------- Tiny self-test (optional) ---------------
if __name__ == "__main__":
    from types import SimpleNamespace
    rng = random.Random(123)
    cases = [
        SimpleNamespace(name="NPC Busker", entity_type="npc", species="Shagdeliacs", biome="Groove Pits",
                        mood="funky", descriptors=["hairy","brassy"], instruments=["clavinet"]),
        SimpleNamespace(name="Crystal Turbine", entity_type="object", species="Rockheads", biome="Crystal Canyons",
                        mood="calm"),
        SimpleNamespace(name="Centerspire Market", entity_type="location", species="Bzaris", biome="Centerspire",
                        mood="energetic"),
        SimpleNamespace(name="Alert", entity_type="event", species="Rockheads", biome="Mag-Lev Yards",
                        mood="angry"),
    ]
    for spec in cases:
        p, meta = build(spec, role_or_kind=("busker" if spec.name=="NPC Busker" else None),
                        time_of_day="dusk", weather="light rain", rng=rng, intensity=0.6)
        print("----", spec.name, "----")
        print(p)
        print(meta)
