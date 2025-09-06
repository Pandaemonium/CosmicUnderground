from dataclasses import dataclass

# --- Input mapping (pygame key constants are looked up at runtime in engine) ---
LEFT_LANES  = ("A", "S", "D", "F")           # melodic lanes
RIGHT_LANES = ("LEFT", "DOWN", "UP", "RIGHT")# percussive lanes
SPACE_LANE  = "SPACE"                         # kick

# --- Visuals ---
TRACK_BG   = (18, 16, 24)
LANE_BG    = (28, 26, 38)
LANE_BORDER= (90, 90, 120)
NOTE_COLOR = (210, 230, 255)
NOTE_STRONG= (255, 250, 160)   # space / kick
RECEPTOR   = (220, 220, 255)
TEXT       = (240, 240, 255)
ACC_BAR_BG = (40, 40, 60)
ACC_BAR_OK = (110, 220, 160)
ACC_BAR_BAD= (220, 120, 120)

# --- Timing & scoring ---
JUDGEMENTS = [
    ("PERFECT", 300,  1000),
    ("GREAT",   500,   700),
    ("GOOD",   1000,   400),
    ("MISS",  9999,     0),
]

CALIBRATION_MS = 0
SPAWN_LEAD_MS = 4000           # how far notes spawn above
COUNT_IN_BEATS = 8             # clicks before start

SCROLL_BEATS_VISIBLE = 8.0        # how many beats between spawn and receptor
HITLINE_OFFSET_PX    = 80         # distance from bottom edge for receptors
LANE_WIDTH           = 90
LANE_GAP             = 18
NOTE_H               = 18          # note height (px)
SPAWN_MARGIN_MS      = 1200        # notes appear when <= this ahead (safety cap)

NOTE_GLYPH = (16, 16, 24)

# Width: either set absolute px, or just use the multiplier
SPACE_LANE_WIDTH        = None            # e.g. 140
SPACE_LANE_WIDTH_MULT   = 1.5             # × LANE_WIDTH if SPACE_LANE_WIDTH is None

# Distinct colors for left-hand lanes
LEFT_LANE_COLORS = {
    "A": (255, 160, 160),
    "S": (255, 200, 130),
    "D": (160, 220, 255),
    "F": (190, 170, 255),
}

# --- Session ---
@dataclass
class SessionTuning:
    loops_to_play: int = 1           # play for N loops of the current audio
    speed_px_per_beat: int = 60      # vertical speed (pixels/beat)
    show_bar_lines: bool = True
