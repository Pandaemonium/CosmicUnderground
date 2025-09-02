# Display / timing
DEFAULT_FULLSCREEN = True
FPS = 60
ENGINE_SR = 44100

# Map / tiles
SCREEN_W, SCREEN_H = 1200, 700  # default; your controller still resizes the window
TILE_W, TILE_H = 100, 120
MAP_W, MAP_H = 96, 96

# Zones
ZONE_MIN, ZONE_MAX = 10, 40
AVG_ZONE = 20
POIS_NPC_RANGE = (0, 2)
POIS_OBJ_RANGE = (0, 1)

# Starting
START_TILE = (MAP_W // 2, MAP_H // 2)
START_ZONE_NAME = "Scrapyard Funk"
START_THEME_WAV  = r"C:\Games\CosmicUnderground\inventory\rec_1756545018_Scrapyard Funk_d5ae11.wav"

# Sprites
PLAYER_SPRITE = r"C:\Games\CosmicUnderground\sprites\character1.png"
PLAYER_SPRITE_COMPLETE = r"C:\Games\CosmicUnderground\sprites\laser_bunny.png"

# Generation / cache
MAX_ACTIVE_LOOPS = 120
GEN_WORKERS = 1
