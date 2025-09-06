"""
DAW Input Validation Module
Centralized validation logic for DAW operations
"""

from typing import List, Tuple, Optional, Any
from .debug import debug_info, debug_warning, debug_error
from .daw_config import VALIDATION_CONFIG, AUDIO_CONFIG

class DAWValidationError(Exception):
    """Custom exception for DAW validation errors"""
    def __init__(self, message: str, operation: str = "Unknown"):
        self.message = message
        self.operation = operation
        super().__init__(f"[{operation}] {message}")

def validate_clip_list(clips: List[Any], operation: str = "Unknown") -> Tuple[bool, str]:
    """
    Validate a list of clips for basic operations
    
    Args:
        clips: List of clips to validate
        operation: Name of the operation being performed
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not clips:
        return False, "No clips provided"
    
    if not isinstance(clips, list):
        return False, "Clips must be provided as a list"
    
    if len(clips) < 1:
        return False, "At least one clip is required"
    
    if len(clips) > VALIDATION_CONFIG['max_clips_per_track']:
        return False, f"Too many clips ({len(clips)} > {VALIDATION_CONFIG['max_clips_per_track']})"
    
    return True, "Validation passed"

def validate_merge_operation(clips: List[Any], operation: str = "Merge") -> Tuple[bool, str]:
    """
    Validate that clips can be merged
    
    Args:
        clips: List of clips to merge
        operation: Name of the operation
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Basic validation
    is_valid, error_msg = validate_clip_list(clips, operation)
    if not is_valid:
        return False, error_msg
    
    if len(clips) < 2:
        return False, "Need at least 2 clips to merge"
    
    # Check if all clips have required attributes
    for i, clip in enumerate(clips):
        if not hasattr(clip, 'track_index'):
            return False, f"Clip {i} missing track_index attribute"
        if not hasattr(clip, 'start_time'):
            return False, f"Clip {i} missing start_time attribute"
        if not hasattr(clip, 'duration'):
            return False, f"Clip {i} missing duration attribute"
        if not hasattr(clip, 'alien_track'):
            return False, f"Clip {i} missing alien_track attribute"
    
    # Check if all clips are on the same track
    track_index = clips[0].track_index
    if not all(clip.track_index == track_index for clip in clips):
        return False, "All clips must be on the same track to merge"
    
    # Check if clips are adjacent (no large gaps)
    sorted_clips = sorted(clips, key=lambda c: c.start_time)
    for i in range(len(sorted_clips) - 1):
        current_clip = sorted_clips[i]
        next_clip = sorted_clips[i + 1]
        
        gap = next_clip.start_time - (current_clip.start_time + current_clip.duration)
        if gap > AUDIO_CONFIG['max_gap_for_merge']:
            return False, f"Clips must be adjacent to merge. Gap between {current_clip.name} and {next_clip.name}: {gap:.2f}s"
    
    return True, "Validation passed"

def validate_slice_operation(clip: Any, slice_time: float, operation: str = "Slice") -> Tuple[bool, str]:
    """
    Validate that a clip can be sliced at the specified time
    
    Args:
        clip: Clip to slice
        slice_time: Time to slice at
        operation: Name of the operation
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not clip:
        return False, "No clip provided"
    
    if not hasattr(clip, 'start_time'):
        return False, "Clip missing start_time attribute"
    
    if not hasattr(clip, 'duration'):
        return False, "Clip missing duration attribute"
    
    if not hasattr(clip, 'alien_track'):
        return False, "Clip missing alien_track attribute"
    
    # Check if slice time is within clip bounds
    clip_start = clip.start_time
    clip_end = clip_start + clip.duration
    
    if not (clip_start < slice_time < clip_end):
        return False, f"Slice time {slice_time:.2f}s must be within clip bounds ({clip_start:.2f}s to {clip_end:.2f}s)"
    
    # Check if resulting clips would be too short
    first_duration = slice_time - clip_start
    second_duration = clip.duration - first_duration
    
    if first_duration < AUDIO_CONFIG['min_clip_duration']:
        return False, f"First slice would be too short: {first_duration:.2f}s < {AUDIO_CONFIG['min_clip_duration']}s"
    
    if second_duration < AUDIO_CONFIG['min_clip_duration']:
        return False, f"Second slice would be too short: {second_duration:.2f}s < {AUDIO_CONFIG['min_clip_duration']}s"
    
    return True, "Validation passed"

def validate_audio_data(clip: Any, operation: str = "Audio") -> Tuple[bool, str]:
    """
    Validate that a clip has valid audio data
    
    Args:
        clip: Clip to validate
        operation: Name of the operation
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not clip:
        return False, "No clip provided"
    
    if not hasattr(clip, 'alien_track'):
        return False, "Clip missing alien_track attribute"
    
    alien_track = clip.alien_track
    if not hasattr(alien_track, 'array'):
        return False, "AlienTrack missing audio array"
    
    if alien_track.array is None:
        return False, "AlienTrack audio array is None"
    
    if not hasattr(alien_track.array, 'shape'):
        return False, "Audio array missing shape attribute"
    
    if len(alien_track.array.shape) == 0:
        return False, "Audio array has no dimensions"
    
    if alien_track.array.shape[0] == 0:
        return False, "Audio array has no samples"
    
    return True, "Validation passed"

def validate_track_index(track_index: int, max_tracks: int, operation: str = "Track") -> Tuple[bool, str]:
    """
    Validate a track index
    
    Args:
        track_index: Track index to validate
        max_tracks: Maximum number of tracks
        operation: Name of the operation
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(track_index, int):
        return False, "Track index must be an integer"
    
    if track_index < 0:
        return False, "Track index cannot be negative"
    
    if track_index >= max_tracks:
        return False, f"Track index {track_index} exceeds maximum tracks ({max_tracks})"
    
    return True, "Validation passed"

def validate_duration(duration: float, operation: str = "Duration") -> Tuple[bool, str]:
    """
    Validate a duration value
    
    Args:
        duration: Duration to validate
        operation: Name of the operation
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(duration, (int, float)):
        return False, "Duration must be a number"
    
    if duration < AUDIO_CONFIG['min_clip_duration']:
        return False, f"Duration {duration:.2f}s is below minimum {AUDIO_CONFIG['min_clip_duration']}s"
    
    if duration > AUDIO_CONFIG['max_clip_duration']:
        return False, f"Duration {duration:.2f}s exceeds maximum {AUDIO_CONFIG['max_clip_duration']}s"
    
    return True, "Validation passed"

def validate_start_time(start_time: float, operation: str = "StartTime") -> Tuple[bool, str]:
    """
    Validate a start time value
    
    Args:
        start_time: Start time to validate
        operation: Name of the operation
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(start_time, (int, float)):
        return False, "Start time must be a number"
    
    if start_time < 0:
        return False, "Start time cannot be negative"
    
    return True, "Validation passed"

def safe_validate(func, *args, **kwargs) -> Tuple[bool, str]:
    """
    Safely execute a validation function, catching any exceptions
    
    Args:
        func: Validation function to call
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        debug_error(f"Validation function {func.__name__} failed: {e}")
        return False, f"Validation error: {str(e)}"

