"""
DAW Audio playback and processing methods
Extracted from cosmic_daw.py for better modularity
"""

import pygame
import numpy as np
from typing import List, Optional, Tuple
from ..core.debug import debug_info, debug_error, debug_warning
from .daw_track import DAWTrack
from .daw_clip import DAWClip


class DAWAudio:
    """Handles audio playback and processing for the DAW"""
    
    def __init__(self, daw):
        self.daw = daw
        
        # Audio state
        self.is_playing = False
        self.is_recording = False
        self.is_looping = False
        
        # Playback state
        self.playhead_position = 0.0
        self.loop_start = 0.0
        self.loop_end = 1200.0
        
        # Audio settings
        self.master_volume = 1.0
        self.sample_rate = 44100
        self.buffer_size = 1024
        
        # Effects and mixing
        self.reverb_enabled = False
        self.reverb_level = 0.3
        self.delay_enabled = False
        self.delay_time = 0.5
        self.delay_feedback = 0.3
    
    def start_playback(self):
        """Start playback of all tracks"""
        if self.is_playing:
            debug_info("Playback already running")
            return
        
        self.is_playing = True
        debug_info("🎵 DAW: Starting playback")
        
        # Start playback for each track
        for track in self.daw.tracks:
            if hasattr(track, 'start_playback'):
                track.start_playback(self.playhead_position)
            else:
                debug_warning(f"Track {track.name if hasattr(track, 'name') else 'Unknown'} has no start_playback method")
    
    def stop_playback(self):
        """Stop playback of all tracks"""
        if not self.is_playing:
            debug_info("Playback already stopped")
            return
        
        self.is_playing = False
        debug_info("🎵 DAW: Stopping all playback and resetting playhead")
        
        # Stop playback for each track
        for track in self.daw.tracks:
            if hasattr(track, 'stop_playback'):
                track.stop_playback()
            else:
                debug_warning(f"Track {track.name if hasattr(track, 'name') else 'Unknown'} has no stop_playback method")
        
        # Stop all pygame mixer audio
        pygame.mixer.stop()
        debug_info("🎵 Stopped all pygame mixer audio")
        
        # Reset playhead
        self.playhead_position = 0.0
        debug_info("🎵 DAW: All playback stopped and playhead reset to 0.0s")
    
    def pause_playback(self):
        """Pause playback without stopping"""
        if not self.is_playing:
            debug_info("Playback not running")
            return
        
        self.is_playing = False
        debug_info("🎵 DAW: Playback paused")
        
        # Pause all tracks
        for track in self.daw.tracks:
            if hasattr(track, 'stop_playback'):
                track.stop_playback()
    
    def toggle_playback(self):
        """Toggle between play and pause"""
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()
    
    def seek_to_position(self, position: float):
        """Seek to a specific time position"""
        if position < 0:
            position = 0.0
        elif position > self.daw.max_duration:
            position = self.daw.max_duration
        
        debug_info(f"🎵 DAW: Seeking to {position:.2f}s")
        
        # Update playhead position
        self.playhead_position = position
        
        # Update all tracks
        for track in self.daw.tracks:
            if hasattr(track, 'set_playhead_position'):
                track.set_playhead_position(position)
            else:
                debug_warning(f"Track {track.name if hasattr(track, 'name') else 'Unknown'} has no set_playhead_position method")
    
    def update_playback(self, dt: float):
        """Update playback state for all tracks"""
        if not self.is_playing:
            return
        
        # Update playhead position
        self.playhead_position += dt
        
        # Check for loop end
        if self.is_looping and self.playhead_position >= self.loop_end:
            self.playhead_position = self.loop_start
            debug_info(f"🎵 DAW: Loop reached end, jumping to {self.loop_start:.2f}s")
        
        # Check for end of timeline
        if self.playhead_position >= self.daw.max_duration:
            if self.is_looping:
                self.playhead_position = self.loop_start
                debug_info(f"🎵 DAW: Timeline end reached, looping to {self.loop_start:.2f}s")
            else:
                self.stop_playback()
                debug_info("🎵 DAW: Timeline end reached, stopping playback")
                return
        
        # Update all tracks
        for track in self.daw.tracks:
            if hasattr(track, 'update_playback'):
                track.update_playback(dt, self.playhead_position)
            else:
                debug_warning(f"Track {track.name if hasattr(track, 'name') else 'Unknown'} has no update_playback method")
    
    def start_recording(self):
        """Start recording on all tracks marked for recording"""
        if self.is_recording:
            debug_info("Recording already active")
            return
        
        self.is_recording = True
        debug_info("🎵 DAW: Starting recording")
        
        # Start recording on tracks marked for recording
        for track in self.daw.tracks:
            if hasattr(track, 'is_recording') and track.is_recording:
                if hasattr(track, 'start_recording'):
                    track.start_recording()
                else:
                    debug_warning(f"Track {track.name if hasattr(track, 'name') else 'Unknown'} has no start_recording method")
    
    def stop_recording(self):
        """Stop recording on all tracks"""
        if not self.is_recording:
            debug_info("Recording not active")
            return
        
        self.is_recording = False
        debug_info("🎵 DAW: Stopping recording")
        
        # Stop recording on all tracks
        for track in self.daw.tracks:
            if hasattr(track, 'stop_recording'):
                track.stop_recording()
            else:
                debug_warning(f"Track {track.name if hasattr(track, 'name') else 'Unknown'} has no stop_recording method")
    
    def toggle_recording(self):
        """Toggle recording state"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def set_loop_points(self, start: float, end: float):
        """Set loop start and end points"""
        if start < 0:
            start = 0.0
        if end > self.daw.max_duration:
            end = self.daw.max_duration
        if start >= end:
            debug_warning("Loop start must be before loop end")
            return
        
        self.loop_start = start
        self.loop_end = end
        debug_info(f"🎵 DAW: Loop points set to {start:.2f}s - {end:.2f}s")
    
    def toggle_loop(self):
        """Toggle loop mode on/off"""
        self.is_looping = not self.is_looping
        debug_info(f"🎵 DAW: Loop mode {'enabled' if self.is_looping else 'disabled'}")
    
    def apply_effects(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply effects to audio data"""
        if not (self.reverb_enabled or self.delay_enabled):
            return audio_data
        
        processed_audio = audio_data.copy()
        
        # Apply reverb
        if self.reverb_enabled:
            processed_audio = self._apply_reverb(processed_audio)
        
        # Apply delay
        if self.delay_enabled:
            processed_audio = self._apply_delay(processed_audio)
        
        return processed_audio
    
    def _apply_reverb(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply simple reverb effect"""
        try:
            # Simple reverb: add delayed, attenuated copies of the signal
            reverb_samples = int(self.sample_rate * 0.1)  # 100ms delay
            reverb_audio = np.zeros_like(audio_data)
            
            # Add delayed signal with attenuation
            if len(audio_data) > reverb_samples:
                reverb_audio[reverb_samples:] = audio_data[:-reverb_samples] * self.reverb_level
            
            # Mix original and reverb
            result = audio_data + reverb_audio
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result = result / max_val
            
            return result
        except Exception as e:
            debug_error(f"Error applying reverb: {e}")
            return audio_data
    
    def _apply_delay(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply delay effect"""
        try:
            # Calculate delay in samples
            delay_samples = int(self.delay_time * self.sample_rate)
            
            if delay_samples >= len(audio_data):
                return audio_data
            
            # Create delayed signal
            delayed_audio = np.zeros_like(audio_data)
            delayed_audio[delay_samples:] = audio_data[:-delay_samples] * self.delay_feedback
            
            # Mix original and delayed signal
            result = audio_data + delayed_audio
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result = result / max_val
            
            return result
        except Exception as e:
            debug_error(f"Error applying delay: {e}")
            return audio_data
    
    def get_master_volume(self) -> float:
        """Get the master volume level"""
        return self.master_volume
    
    def set_master_volume(self, volume: float):
        """Set the master volume level"""
        if volume < 0.0:
            volume = 0.0
        elif volume > 1.0:
            volume = 1.0
        
        self.master_volume = volume
        debug_info(f"🎵 DAW: Master volume set to {volume:.2f}")
        
        # Update pygame mixer volume
        pygame.mixer.music.set_volume(volume)
    
    def get_playback_status(self) -> dict:
        """Get current playback status"""
        return {
            'is_playing': self.is_playing,
            'is_recording': self.is_recording,
            'is_looping': self.is_looping,
            'playhead_position': self.playhead_position,
            'loop_start': self.loop_start,
            'loop_end': self.loop_end,
            'master_volume': self.master_volume,
            'reverb_enabled': self.reverb_enabled,
            'delay_enabled': self.delay_enabled
        }
    
    def cleanup(self):
        """Clean up audio resources"""
        try:
            # Stop all playback
            self.stop_playback()
            
            # Stop pygame mixer
            pygame.mixer.quit()
            
            debug_info("🎵 DAW: Audio system cleaned up")
        except Exception as e:
            debug_error(f"Error cleaning up audio: {e}")
