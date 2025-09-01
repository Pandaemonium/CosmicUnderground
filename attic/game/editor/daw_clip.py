"""
DAWClip class - represents a clip in the DAW timeline
Extracted from cosmic_daw.py for better modularity
"""

import pygame
from typing import Optional
from ..core.debug import debug_info, debug_error


class DAWClip:
    """Represents a clip in the DAW timeline"""
    
    def __init__(self, alien_track, start_time: float, track_index: int, daw=None, duration: float = None, name: str = None):
        self.alien_track = alien_track
        self.start_time = start_time
        self.track_index = track_index
        self.daw = daw
        self.duration = duration if duration is not None else (alien_track.duration if alien_track else 0.0)
        self.name = name if name is not None else (alien_track.name if alien_track else "Unnamed Clip")
        
        # Selection state
        self.is_selected = False
        self.is_playing = False
        
        # Audio playback
        self.current_channel = None
        self.current_sliced_sound = None
    
    def render(self, surface, timeline_x: int, timeline_y: int, pixels_per_second: float, track_height: int, track_spacing: int):
        """Render the clip on the timeline"""
        from ..core.config import CLIP_FILL, CLIP_BORDER
        
        # Calculate clip position and size
        x = timeline_x + (self.start_time * pixels_per_second)
        y = timeline_y + (self.track_index * (track_height + track_spacing))
        width = self.duration * pixels_per_second  # Allow full scaling with zoom
        height = track_height
        
        # Debug: Print clip rendering coordinates (commented out for performance)
        # debug_info(f"🎵 Rendering clip '{self.name}' at x={x}, y={y}, width={width}, height={height}")
        # debug_info(f"🎵 Timeline: x={timeline_x}, y={timeline_y}, track_index={self.track_index}")
        # debug_info(f"🎵 Clip duration: {self.duration}s, pixels_per_second: {pixels_per_second}")
        
        # Draw clip background
        clip_color = CLIP_FILL if not self.is_selected else (255, 255, 100)  # Yellow when selected
        pygame.draw.rect(surface, clip_color, (x, y, width, height))
        pygame.draw.rect(surface, CLIP_BORDER, (x, y, width, height), 2)
        
        # Draw clip name
        try:
            font = pygame.font.Font(None, 20)
            text = font.render(self.name, True, (0, 0, 0))
            text_rect = text.get_rect(center=(x + width//2, y + height//2))
            surface.blit(text, text_rect)
        except Exception as e:
            debug_info(f"Could not render clip text: {e}")
        
        # Draw waveform if clip is wide enough and has audio data
        if width > 20 and hasattr(self, 'alien_track') and self.alien_track and hasattr(self.alien_track, 'array'):
            # Ensure width is an integer for waveform generation
            waveform_width = max(1, int(width))
            self._render_waveform(surface, x, y, waveform_width, height)
    
    def _render_waveform(self, surface, x: int, y: int, width: int, height: int):
        """Render a simple waveform visualization"""
        try:
            audio_data = self.alien_track.array
            
            # Skip if no audio data
            if audio_data is None or len(audio_data) == 0:
                return
            
            # Generate waveform points
            waveform_points = self._generate_waveform_points(audio_data, width)
            
            if not waveform_points:
                return
            
            # Draw waveform
            waveform_color = (255, 255, 255)  # Bright white for visibility
            center_y = y + height // 2
            
            # Draw waveform lines
            for i in range(len(waveform_points) - 1):
                if i + 1 < len(waveform_points):
                    # Scale the x position to match the actual clip width
                    x1 = x + int(i * width / len(waveform_points))
                    y1 = center_y + int(waveform_points[i] * (height // 2) * 0.8)
                    x2 = x + int((i + 1) * width / len(waveform_points))
                    y2 = center_y + int(waveform_points[i + 1] * (height // 2) * 0.8)
                    
                    # Ensure y coordinates are within clip bounds
                    y1 = max(y + 2, min(y + height - 2, y1))
                    y2 = max(y + 2, min(y + height - 2, y2))
                    
                    pygame.draw.line(surface, waveform_color, (x1, y1), (x2, y2), 2)  # Thicker lines
                    
                    # Draw mirrored waveform line above center for better visibility
                    y1_mirror = center_y - int(waveform_points[i] * (height // 2) * 0.8)
                    y2_mirror = center_y - int(waveform_points[i + 1] * (height // 2) * 0.8)
                    
                    # Ensure mirrored y coordinates are within clip bounds
                    y1_mirror = max(y + 2, min(y + height - 2, y1_mirror))
                    y2_mirror = max(y + 2, min(y + height - 2, y2_mirror))
                    
                    pygame.draw.line(surface, waveform_color, (x1, y1_mirror), (x2, y2_mirror), 2)
                    
        except Exception as e:
            pass  # Silently handle waveform rendering errors
    
    def _generate_waveform_points(self, audio_data, target_width: int):
        """Generate waveform points from audio data"""
        try:
            if len(audio_data) == 0:
                return []
            
            # Handle stereo audio data - flatten to mono for waveform
            if len(audio_data.shape) > 1 and audio_data.shape[1] == 2:
                # Stereo audio - convert to mono by averaging channels
                audio_data = audio_data.mean(axis=1)
            elif len(audio_data.shape) > 1:
                # Multi-channel audio - take first channel
                audio_data = audio_data[:, 0]
            
            # Sample audio data to match target width
            step = max(1, len(audio_data) // target_width)
            samples = audio_data[::step]
            
            # Create a simple but visible waveform
            waveform_points = []
            
            # Use a simpler approach: sample evenly across the audio data
            for i in range(target_width):
                # Calculate the sample index for this pixel
                sample_index = int(i * len(audio_data) / target_width)
                if sample_index < len(audio_data):
                    # Get a small chunk around this sample for averaging
                    start_idx = max(0, sample_index - 100)
                    end_idx = min(len(audio_data), sample_index + 100)
                    chunk = audio_data[start_idx:end_idx]
                    
                    # Calculate RMS value for this chunk
                    if len(chunk) > 0:
                        rms_val = (chunk ** 2).mean() ** 0.5
                        # Normalize to 0-1 range
                        normalized = min(1.0, rms_val / 2000.0)
                        waveform_points.append(normalized)
                    else:
                        waveform_points.append(0.0)
                else:
                    waveform_points.append(0.0)
            
            return waveform_points[:target_width]
            
        except Exception as e:
            print(f"DEBUG: Error in _generate_waveform_points: {e}")
            return []
    
    def duplicate(self):
        """Create a duplicate of this clip"""
        try:
            duplicated_clip = DAWClip(
                alien_track=self.alien_track,
                start_time=self.start_time + self.duration + 0.1,  # Place after current clip
                track_index=self.track_index,
                daw=self.daw,
                duration=self.duration,
                name=f"{self.name}_copy"
            )
            return duplicated_clip
        except Exception as e:
            debug_error(f"Failed to duplicate clip: {e}")
            return None
    
    def delete(self):
        """
        Delete this clip from its track and clean up resources
        
        This method removes the clip from its parent track and stops any
        associated audio playback. It should be called before removing
        the clip from the track's clips list.
        """
        try:
            # Stop any audio that might be playing from this clip
            if hasattr(self, 'current_channel') and self.current_channel:
                try:
                    self.current_channel.stop()
                    self.current_channel = None
                except Exception as e:
                    from ..core.debug import debug_warning
                    debug_warning(f"Could not stop channel for {self.name}: {e}")
            
            if hasattr(self, 'current_sliced_sound') and self.current_sliced_sound:
                try:
                    self.current_sliced_sound.stop()
                    self.current_sliced_sound = None
                except Exception as e:
                    from ..core.debug import debug_warning
                    debug_warning(f"Could not stop sliced sound for {self.name}: {e}")
            
            # Remove from parent track if it exists
            if hasattr(self, 'daw') and self.daw and hasattr(self.daw, 'tracks'):
                if (0 <= self.track_index < len(self.daw.tracks) and 
                    hasattr(self.daw.tracks[self.track_index], 'clips')):
                    track = self.daw.tracks[self.track_index]
                    if hasattr(track, 'remove_clip') and self in track.clips:
                        track.remove_clip(self)
                        debug_info(f"Removed clip {self.name} from track {track.name}")
            
            debug_info(f"Successfully deleted clip: {self.name}")
            
        except Exception as e:
            debug_error(f"Error deleting clip {self.name}: {e}")
            import traceback
            traceback.print_exc()
