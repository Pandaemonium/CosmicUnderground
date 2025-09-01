import pygame

# Window settings
WIDTH = 1600
HEIGHT = 1024
BG_COLOR = (12, 17, 38)

# Debug settings
DEBUG_MODE = True  # Set to False to disable verbose logging

# Colors
PLAYER_COLOR = (201, 159, 255)
ZONE_COLOR = (45, 212, 191)       # Mint Lagoon
LISTEN_COLOR = (163, 230, 53)     # Listening Station
TEXT_COLOR = (232, 240, 255)
INFO_COLOR = (170, 190, 220)
UI_OUTLINE = (255, 255, 255)
HILITE = (80, 120, 200)
ERR = (239, 68, 68)
GRID = (60, 80, 120)
CLIP_FILL = (70, 100, 150)
CLIP_BORDER = (200, 230, 255)
PLAYHEAD = (255, 190, 0)  # gold audition playhead
TL_PLAYHEAD = (255, 90, 90)

# DAW Track Colors
TRACK_COLORS = [
    (70, 130, 180),   # Steel Blue
    (220, 20, 60),    # Crimson
    (34, 139, 34),    # Forest Green
    (255, 140, 0),    # Dark Orange
    (138, 43, 226),   # Blue Violet
    (255, 215, 0),    # Gold
    (255, 20, 147),   # Deep Pink
    (0, 191, 255),    # Deep Sky Blue
]

# DAW UI Colors
PLAYHEAD_COLOR = (255, 190, 0)  # Gold playhead
GRID_COLOR = (60, 80, 120)      # Grid lines
TIMELINE_COLOR = (30, 40, 60)   # Timeline background
TRACK_LABEL_COLOR = (200, 220, 255)  # Track label text
HELP_COLOR = (100, 200, 255)    # Help text color

# Fonts (init after pygame.init())
def fonts():
    return (
        pygame.font.SysFont(None, 22),
        pygame.font.SysFont(None, 18),
        pygame.font.SysFont(pygame.font.get_default_font(), 16),
    )

# World scale
TILE_SIZE = 48
PLAYER_SCALE = 0.9  # sprite scales to ~90% of tile

# Sprite path (PNG with alpha recommended)
SPRITE_PATH = r"C:\Users\Owner\Documents\Games\Cosmic Underground\sprites\character1.png"

# CRATE SYSTEM - SONG GROUPS AND INDIVIDUAL TRACKS
# =================================================
# 
# The crate system now supports both individual tracks and song groups:
# 
# 1. INDIVIDUAL TRACKS: Specify the full path to a .wav/.mp3 file
#    Example: r"C:\Users\Owner\Downloads\MySong.wav"
# 
# 2. SONG GROUPS: Specify a folder path to include ALL audio files from that folder
#    Example: r"C:\Users\Owner\Music\AI music\Cosmic Underground songs\Funky Space Groove"
# 
# When a crate contains a song group folder, ALL audio files from that folder will be
# automatically included in the crate contents. This is perfect for complete song
# collections where you want all the individual instrument tracks.
# 
# SUPPORTED AUDIO FORMATS: .wav, .mp3, .flac, .ogg, .m4a, .aiff
# 
# BEHAVIOR SETTINGS (in SONG_GROUP_SETTINGS below):
# - GIVE_ALL_TRACKS_FROM_GROUPS: When True, gives ALL tracks from song groups
# - MAX_INDIVIDUAL_TRACKS: Maximum individual tracks to give per crate
# - MIN_TRACKS_PER_CRATE: Minimum total tracks to give per crate
# 
# Add placeholder tracks (fill in with your actual .wav paths)
# You can now specify either individual files or folder paths for song groups
CRATE_TRACKS = {
    "Crate 1": [
        # Song group - this will include ALL audio files from the Martian Blues folder
        r"C:\Users\Owner\Music\AI music\Cosmic Underground songs\Martian Blues",
        # Individual files can also be mixed with song groups
        r"C:\Users\Downloads\Bonus Track.wav"
    ],
    "Crate 2": [
        # Song group - this will include ALL audio files from the Funky Space Groove folder
        r"C:\Users\Owner\Music\AI music\Cosmic Underground songs\Funky Space Groove",
        # Individual files
        r"C:\Users\Downloads\Drippy.wav",
        r"C:\Users\Downloads\Cosmic Clockwork.wav"
    ],
    "Crate 3": [
        # Multiple song groups in one crate
        r"C:\Users\Owner\Music\AI music\Cosmic Underground songs\Cosmic Serenade",
        r"C:\Users\Music\AI music\Cosmic Underground songs\Neon Dreams",
        # Plus some individual tracks
        r"C:\Users\Downloads\Galactic Groove.wav"
    ]
}

# Song group configuration - define which folders should be treated as song groups
# When a crate contains a folder path, all audio files from that folder will be included
SONG_GROUP_FOLDERS = [
    r"C:\Users\Owner\Music\AI music\Cosmic Underground songs\Funky Space Groove",
    r"C:\Users\Owner\Music\AI music\Cosmic Underground songs\Martian Blues",
    r"C:\Users\Owner\Music\AI music\Cosmic Underground songs\Cosmic Serenade",
    # Add more song group folders here as needed
]

# Song group behavior settings
SONG_GROUP_SETTINGS = {
    # When a song group is found, give ALL tracks from that group instead of random selection
    "GIVE_ALL_TRACKS_FROM_GROUPS": True,
    
    # Maximum number of individual tracks to give when no song groups are found
    "MAX_INDIVIDUAL_TRACKS": 4,
    
    # Minimum number of tracks to give from a crate
    "MIN_TRACKS_PER_CRATE": 2,
}

# Supported audio file extensions for song groups
SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff']
    

ZONES = {
    "Mint Lagoon": (100, 100, 120, 80),
    "Listening Station": (400, 100, 140, 100),
    "Exec HQ": (250, 300, 120, 100),
    "Crimson Bazaar": (600, 200, 120, 80),
    "Azure Caves": (150, 400, 120, 80),
    "Neon District": (700, 350, 100, 90),
    "Frost Peak": (50, 250, 100, 70),
    "Crate 1": (700, 200, 80, 80),
    "Crate 2": (300, 400, 80, 80),
    "Crate 3": (500, 500, 80, 80),
}


# Add a color map for zones (fallback colors still used if not present)
ZONE_COLORS = {
    "Mint Lagoon": (45, 212, 191),        # teal
    "Listening Station": (163, 230, 53),  # green
    "Crimson Bazaar": (220, 70, 90),      # crimson
    "Azure Caves": (80, 156, 255),        # azure
    "Neon District": (255, 20, 147),      # hot pink
    "Frost Peak": (173, 216, 230),        # light blue
    "Crate 1": (255, 165, 0),             # orange
    "Crate 2": (255, 215, 0),             # gold
    "Crate 3": (138, 43, 226),            # blue violet
}
ZONE_OUTLINE = (250, 250, 255)  # white outline to make them pop

# Pickup file lists, 1..4 keys per zone
MINT_LOCAL_FILES = [
    r"C:\Users\Owner\Music\AI music\Cosmic Underground songs\Cosmic Serenade.wav",
    r"C:\Users\Owner\Music\AI music\Cosmic Underground songs\Aliens_1.wav",
    r"C:\Users\Owner\Music\AI music\Cosmic Underground songs\Aliens_2.wav",
    r"C:\Users\Owner\Music\AI music\Cosmic Underground songs\Aliens_3.wav",
]

# Placeholders for your two new destinations — fill these in with real .wav paths
CRIMSON_LOCAL_FILES = [
    r"C:\Users\Owner\Downloads\Drippy.wav",  # press 1
    r"C:\Users\Owner\Downloads\Drippy (2).wav",  # press 2
    r"C:\Users\Owner\Downloads\Nauli Oona-12.wav",  # press 3
    r"C:\Users\Owner\Downloads\Cosmic Clockwork.wav",  # press 4
]

AZURE_LOCAL_FILES = [
    r"C:\Users\Owner\Downloads\Cosmic Chimes.wav",    # press 1
    r"C:\Users\Owner\Downloads\Cosmic Chime Choir.wav",    # press 2
    r"C:\Users\Owner\Downloads\Alien Serenade.wav",    # press 3
    r"C:\Users\Owner\Downloads\Galactic Groove.wav",    # press 4
]

# New location track files
NEON_DISTRICT_FILES = [
    r"C:\Users\Owner\Music\AI music\Cosmic Serenade\Cosmic Serenade (Guitar).mp3",        # press 1
    r"C:\Users\Owner\Music\AI music\Cosmic Serenade\Cosmic Serenade (Synth).mp3",        # press 2
    r"C:\Users\Owner\Music\AI music\Cosmic Serenade\Cosmic Serenade (Drums).mp3",    # press 3
    r"C:\Users\Owner\Music\AI music\Cosmic Serenade\Cosmic Serenade (Bass).mp3",  # press 4
]

FROST_PEAK_FILES = [
    r"C:\Users\Owner\Downloads\Digital Fractures.wav",        # press 1
    r"C:\Users\Owner\Downloads\Digital Daydreams.wav",     # press 2
    r"C:\Users\Owner\Downloads\Cosmic Chaos Symphony.wav",     # press 3
    r"C:\Users\Owner\Downloads\Galactic Drift.wav",    # press 4
]

# A single lookup so screens don’t hardcode zone file lists
ZONE_PICKUPS = {
    "Mint Lagoon": MINT_LOCAL_FILES,
    "Crimson Bazaar": CRIMSON_LOCAL_FILES,
    "Azure Caves": AZURE_LOCAL_FILES,
    "Neon District": NEON_DISTRICT_FILES,
    "Frost Peak": FROST_PEAK_FILES,
    # Note: "Listening Station" has no pickups
}


# Audio constants
SR = 44100
MIN_CLIP_SEC = 0.05
MIN_ZOOM_NORM = 0.02

# Game mechanics constants
BOOTLEG_ATTENTION_RATE = 10.0  # Attention gained per second while recording
SNEAK_ATTENTION_MULTIPLIER = 0.3  # Attention multiplier when sneaking
MAX_ATTENTION = 100.0  # Maximum attention before getting caught
ATTENTION_DECAY_RATE = 5.0  # Attention lost per second when not recording

# Alien species and their preferences
ALIEN_SPECIES = {
    "Chillaxians": {
        "preferences": ["chill", "ambient", "peaceful", "snow", "ice", "polar", "relaxing", "calm"]
    },
    "Glorpals": {
        "preferences": ["wet", "slimy", "fluid", "organic", "natural", "flowing", "smooth"]
    },
    "Bzaris": {
        "preferences": ["glitchy", "high-energy", "fast", "electronic", "energetic", "intense", "buzzing"]
    },
    "Shagdeliacs": {
        "preferences": ["jazzy", "funky", "groovy", "rhythmic", "swing", "soul", "blues"]
    },
    "Rockheads": {
        "preferences": ["metal", "gong", "percussive", "underground", "industrial", "heavy", "powerful"]
    }
}

# Executive preferences for music descriptors
EXEC_PREFERENCES = {
    "Executive Alpha": ["chill", "ambient", "peaceful"],
    "Executive Beta": ["energetic", "intense", "powerful"],
    "Executive Gamma": ["funky", "groovy", "rhythmic"],
    "Executive Delta": ["ethereal", "mystical", "otherworldly"],
    "Executive Epsilon": ["mechanical", "industrial", "robotic"]
}

# Import DAW-specific configuration
from .daw_config import AUDIO_CONFIG, UI_CONFIG, TIMELINE_CONFIG, UNDO_CONFIG, VALIDATION_CONFIG, DEBUG_CONFIG

