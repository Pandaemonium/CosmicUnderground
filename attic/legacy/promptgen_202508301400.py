# promptgen.py
from __future__ import annotations
import random

SCHEMA_VERSION = "pg/1.0.0"

CONTRAST_PAIRS = [
    ("fast","slow"), ("funky","classic"), ("brassy","woodwinds"),
    ("loud","soft"), ("layered","simple"), ("retro","futuristic"),
]

SPECIES_POOLS = {
    "Chillaxians": dict(
        bpm=(84,102),
        instruments=["warm pads","ice chimes","soft bass","brush kit","chorus keys"],
        textures=["calm","retro","soft","crystalline","airy"],
    ),
    "Glorpals": dict(
        bpm=(96,122),
        instruments=["squelch synth","gel bells","drip fx","rubber bass","wet claps"],
        textures=["wet","slurpy","layered","sproingy"],
    ),
    "Bzaris": dict(
        bpm=(120,140),
        instruments=["bitcrush blips","granular stutters","noisy hats","FM lead","resonant zap"],
        textures=["fast","glitchy","futuristic","electric"],
    ),
    "Shagdeliacs": dict(
        bpm=(96,120),
        instruments=["wah guitar","clavinet","horn section","rimshot kit","upright-ish synth bass"],
        textures=["funky","brassy","jazzy","retro","hairy","furry"],
    ),
    "Rockheads": dict(
        bpm=(100,126),
        instruments=["metal gongs","anvil hits","deep toms","clang perc","industrial pad"],
        textures=["mechanical","metallic","loud","magnetic"],
    ),
}

EXTRA_INSTR = ["kalimba","woodblocks","808 subs","tape echo stabs","analog brass",
               "vocoder chops","hand drums","dub chords","arpeggiator"]
EXTRA_TEXTS = ["crystalline","laser","electric","mechanical","nuclear","magnetic",
               "chemical","fuzzy","hair-clad","hirsute"]
MODES  = ["major","minor","Dorian","Phrygian","Mixolydian","Lydian","Aeolian"]
TONICS = ["A","B","C","D","E","F","G","A♭","B♭","C♯","D♯","F♯","G♯"]

def build_prompt_from_spec(spec, bars=8, rng=None, intensity=0.55) -> str:
    """
    spec: an object/dict with attributes/keys:
      species, biome, mood, bpm (optional), key_mode (optional),
      descriptors (list), instruments (list), name
    Returns: Stable Audio prompt string.
    """
    rng = rng or random.Random()

    # --- BPM ---
    bpm = getattr(spec, "bpm", None) if hasattr(spec, "bpm") else spec.get("bpm")
    species = getattr(spec, "species", None) if hasattr(spec, "species") else spec.get("species")
    species_opts = SPECIES_POOLS.get(species or "", None)
    if species_opts and bpm is None:
        lo, hi = species_opts["bpm"]
        bpm = rng.randrange(lo, hi+1, 2)
    bpm = int(max(84, min(140, (bpm or 120) + rng.choice([-4,-2,0,2,4]))))

    # --- Key/mode ---
    key_mode = getattr(spec, "key_mode", None) if hasattr(spec, "key_mode") else spec.get("key_mode")
    if not key_mode:
        key_mode = f"{rng.choice(TONICS)} {rng.choice(MODES)}"

    # --- Instruments ---
    given_instruments = getattr(spec, "instruments", None) if hasattr(spec, "instruments") else spec.get("instruments", [])
    base_instr = (species_opts["instruments"] if species_opts else []) + list(given_instruments or [])
    base_instr = list(dict.fromkeys(base_instr + EXTRA_INSTR))  # unique, ordered
    k_instr = max(2, min(4, 2 + int(2*intensity)))
    instr = rng.sample(base_instr, k=min(k_instr, len(base_instr))) if base_instr else []

    # --- Textures ---
    given_desc = getattr(spec, "descriptors", None) if hasattr(spec, "descriptors") else spec.get("descriptors", [])
    base_tex = (species_opts["textures"] if species_opts else []) + list(given_desc or [])
    base_tex = list(dict.fromkeys(base_tex + EXTRA_TEXTS))
    k_tex = max(2, min(3, 2 + int(1*intensity)))
    tex = rng.sample(base_tex, k=min(k_tex, len(base_tex))) if base_tex else []

    # --- Contrast (one side) ---
    side = rng.choice(CONTRAST_PAIRS)
    tags = [rng.choice(side)]
    if tex: tags.append(rng.choice(tex))

    # --- Compose ---
    mood   = getattr(spec, "mood", None)  if hasattr(spec, "mood")  else spec.get("mood", "energetic")
    biome  = getattr(spec, "biome", None) if hasattr(spec, "biome") else spec.get("biome", "unknown biome")
    spname = species or "unknown species"

    desc = ", ".join(tex[:3])
    inst = ", ".join(instr[:4])
    tagline = ", ".join(tags[:2])

    return (
        f"{mood} vibes in the {biome}, home of the {spname}. "
        f"{desc}. " + (f"Instruments: {inst}. " if inst else "") +
        (f"Style tags: {tagline}. " if tagline else "") +
        f"Loopable {bars} bars, {bpm} BPM, {key_mode}. "
        f"Clean downbeat; seamless loop; minimal silence at edges; tight low end."
    )
