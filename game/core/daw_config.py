"""
DAW Configuration Settings
Centralized configuration for the Cosmic DAW system
"""

# Audio Processing Settings
AUDIO_CONFIG = {
    'sample_rate': 44100,
    'buffer_size': 1024,
    'min_clip_duration': 0.1,  # Minimum clip duration in seconds
    'max_clip_duration': 600.0,  # Maximum clip duration in seconds (10 minutes)
    'max_gap_for_merge': 0.1,   # Maximum gap between clips to allow merging
    'audio_channels': 2,         # Default number of audio channels
}

# UI Settings
UI_CONFIG = {
    'refresh_rate': 60,          # UI refresh rate in FPS
    'clip_min_width': 40,       # Minimum clip width in pixels to show text
    'clip_min_height': 20,      # Minimum clip height in pixels
    'selection_border_width': 3, # Width of selection border
    'corner_accent_size': 20,   # Size of corner accent decorations
}

# DAW Timeline Settings
TIMELINE_CONFIG = {
    'default_zoom': 100,        # Default pixels per second
    'min_zoom': 10,             # Minimum zoom level
    'max_zoom': 1000,           # Maximum zoom level
    'snap_threshold': 5,        # Snap threshold in pixels
    'grid_snap': True,          # Enable grid snapping
    'grid_divisions': 16,       # Grid divisions per beat
}

# Undo/Redo Settings
UNDO_CONFIG = {
    'max_undo_steps': 50,       # Maximum number of undo steps to remember
    'auto_save_interval': 30,   # Auto-save interval in seconds
}

# Validation Settings
VALIDATION_CONFIG = {
    'max_clips_per_track': 100, # Maximum clips allowed per track
    'max_tracks': 16,           # Maximum number of tracks
    'max_clip_name_length': 50, # Maximum length of clip names
}

# Debug Settings
DEBUG_CONFIG = {
    'log_audio_operations': True,    # Log audio slicing/merging operations
    'log_ui_operations': False,     # Log UI rendering operations
    'log_performance': True,        # Log performance metrics
    'audio_validation': True,       # Validate audio data integrity
}

