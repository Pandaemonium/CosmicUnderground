"""
DAWTrack class - represents a track in the DAW
Extracted from cosmic_daw.py for better modularity
"""

import pygame
from typing import List, Optional
from ..core.debug import debug_info, debug_error
from .daw_clip import DAWClip


class DAWTrack:
    """Represents a track in the DAW"""
    
    def __init__(self, name: str, color: tuple, track_index: int, daw=None):
        self.name = name
        self.color = color
        self.track_index = track_index
        self.daw = daw
        
        # Track state
        self.is_muted = False
        self.is_soloed = False
        self.is_recording = False
        
        # Audio properties
        self.volume = 1.0
        self.pan = 0.0
        
        # Clips
        self.clips: List[DAWClip] = []
        
        # Playback state
        self.is_playing = False
        self.current_clip: Optional[DAWClip] = None
        self.playback_position = 0.0
        self._audio_started = False
        self.current_channel = None
        self.current_sliced_sound = None
    
    def add_clip(self, clip: DAWClip):
        """Add a clip to this track"""
        if clip not in self.clips:
            self.clips.append(clip)
            clip.track_index = self.track_index
            debug_info(f"Added clip {clip.name} to track {self.name}")
    
    def remove_clip(self, clip: DAWClip):
        """Remove a clip from this track"""
        if clip in self.clips:
            self.clips.remove(clip)
            debug_info(f"Removed clip {clip.name} from track {self.name}")
    
    def has_content(self) -> bool:
        """Check if this track has any clips"""
        return len(self.clips) > 0
    
    def toggle_mute(self):
        """Toggle mute state"""
        self.is_muted = not self.is_muted
        
        # Immediately apply mute state to current audio
        if self.is_muted:
            # Stop current audio if muted
            if self.current_channel and self.current_channel.get_busy():
                self.current_channel.stop()
                debug_info(f"Track {self.name}: Muted, stopped current audio")
        else:
            # Unmuted - restart audio if we should be playing
            if self.current_clip and self.is_playing:
                current_time = self.daw.playhead_position if hasattr(self.daw, 'playhead_position') else 0.0
                if (self.current_clip.start_time <= current_time <= 
                    self.current_clip.start_time + self.current_clip.duration):
                    # Check if we can actually play (no solo conflicts)
                    if not self.daw._has_soloed_tracks() or self.is_soloed:
                        self._start_audio_playback(self.current_clip, current_time)
                        debug_info(f"Track {self.name}: Unmuted, restarted audio")
        
        debug_info(f"Track {self.name} mute toggled: {'Muted' if self.is_muted else 'Unmuted'}")
    
    def toggle_solo(self):
        """Toggle solo state"""
        self.is_soloed = not self.is_soloed
        
        # Immediately apply solo state to current audio
        if self.daw._has_soloed_tracks():
            # Some tracks are soloed - check if this track should play
            if self.is_soloed:
                # This track is now soloed - start audio if we should be playing
                if self.current_clip and self.is_playing and not self.is_muted:
                    current_time = self.daw.playhead_position if hasattr(self.daw, 'playhead_position') else 0.0
                    if (self.current_clip.start_time <= current_time <= 
                        self.current_clip.start_time + self.current_clip.duration):
                        self._start_audio_playback(self.current_clip, current_time)
                        debug_info(f"Track {self.name}: Soloed, started audio")
            else:
                # This track is no longer soloed - stop audio
                if self.current_channel and self.current_channel.get_busy():
                    self.current_channel.stop()
                    debug_info(f"Track {self.name}: No longer soloed, stopped audio")
        else:
            # No tracks are soloed - restore normal playback for unmuted tracks
            if not self.is_muted and self.current_clip and self.is_playing:
                current_time = self.daw.playhead_position if hasattr(self.daw, 'playhead_position') else 0.0
                if (self.current_clip.start_time <= current_time <= 
                    self.current_clip.start_time + self.current_clip.duration):
                    self._start_audio_playback(self.current_clip, current_time)
                    debug_info(f"Track {self.name}: No solo conflicts, restarted audio")
        
        debug_info(f"Track {self.name} solo toggled: {'Soloed' if self.is_soloed else 'Unsoloed'}")
        
        # Handle solo conflicts for all tracks
        if self.daw:
            for track in self.daw.tracks:
                if track != self:  # Don't call on self
                    track._handle_solo_conflicts()
    
    def _handle_solo_conflicts(self):
        """Handle solo conflicts by stopping audio if this track shouldn't be playing"""
        if not self.daw:
            return
        
        # Check if any tracks are soloed
        if self.daw._has_soloed_tracks():
            # Some tracks are soloed - this track should only play if it's also soloed
            if not self.is_soloed and self.current_channel and self.current_channel.get_busy():
                self.current_channel.stop()
                debug_info(f"Track {self.name}: Stopped due to solo conflict")
        else:
            # No tracks are soloed - this track should play if not muted and has content at current time
            if not self.is_muted and self.is_playing:
                current_time = self.daw.playhead_position if hasattr(self.daw, 'playhead_position') else 0.0
                
                # Find if we have a clip that should be playing at this time
                should_play_clip = None
                for clip in self.clips:
                    if (clip.start_time <= current_time <= 
                        clip.start_time + clip.duration):
                        should_play_clip = clip
                        break
                
                if should_play_clip:
                    # Only restart if not already playing
                    if not self.current_channel or not self.current_channel.get_busy():
                        self._start_audio_playback(should_play_clip, current_time)
                        debug_info(f"Track {self.name}: Restarted after solo conflict resolved")
                else:
                    # No clip should be playing at this time, stop if playing
                    if self.current_channel and self.current_channel.get_busy():
                        self.current_channel.stop()
                        debug_info(f"Track {self.name}: Stopped - no clip at current time")
    
    def toggle_record(self):
        """Toggle record state"""
        self.is_recording = not self.is_recording
        debug_info(f"Track {self.name} record toggled: {'Recording' if self.is_recording else 'Not recording'}")
    
    def start_playback(self, playhead_position: float):
        """Start playback for this track"""
        self.is_playing = True
        self.playback_position = playhead_position
        self._find_current_clip(playhead_position)
        
        if self.current_clip:
            self._start_audio_playback(self.current_clip, playhead_position)
    
    def stop_playback(self):
        """Stop playback for this track"""
        self.is_playing = False
        self.current_clip = None
        self.playback_position = 0.0
        self._audio_started = False
        
        # Stop any audio
        self._stop_audio_playback()
    
    def set_playhead_position(self, position: float):
        """Set the playhead to a new position and update audio accordingly"""
        debug_info(f"Track {self.name}: Setting playhead to {position:.2f}s")
        
        if self.is_playing:
            # Stop current audio
            if self.current_clip:
                self._stop_audio_playback()
                self._audio_started = False
            
            # Update position and find new clip
            self.playback_position = position
            self._find_current_clip(position)
            
            # Start new audio if we have a clip
            if self.current_clip:
                self._start_audio_playback(self.current_clip, position)
        else:
            # Just update position for when we start playing
            self.playback_position = position
            self._find_current_clip(position)
    
    def update_playback(self, dt: float, playhead_position: float):
        """Update playback state for this track"""
        if not self.is_playing:
            return
        
        # Update playback position
        self.playback_position = playhead_position
        
        # Find current clip at this position
        self._find_current_clip(playhead_position)
        
        # Update audio playback
        if self.current_clip:
            should_be_playing = (self.current_clip.start_time <= playhead_position <= 
                               self.current_clip.start_time + self.current_clip.duration)
            
            if should_be_playing and not self.current_clip.is_playing:
                debug_info(f"Track {self.name}: Starting clip {self.current_clip.name}")
                self._start_audio_playback(self.current_clip, playhead_position)
            elif not should_be_playing and self.current_clip.is_playing:
                debug_info(f"Track {self.name}: Stopping clip {self.current_clip.name}")
                self._stop_audio_playback()
    
    def _start_audio_playback(self, clip: DAWClip, playhead_position: float):
        """Start playing audio for a specific clip"""
        try:
            # Check if track is muted - if so, don't play audio
            if self.is_muted:
                debug_info(f"Track {self.name}: Muted, not playing {clip.name}")
                return
            
            # Check if any other track is soloed - if so, only play if this track is also soloed
            if self.daw and self.daw._has_soloed_tracks():
                if not self.is_soloed:
                    debug_info(f"Track {self.name}: Not soloed, not playing {clip.name} (other tracks are soloed)")
                    return
            
            # Calculate offset within the clip
            clip_offset = playhead_position - clip.start_time
            
            # Create a sliced sound from the clip's alien track
            if hasattr(clip.alien_track, 'make_slice_sound'):
                sliced_sound = clip.alien_track.make_slice_sound(clip_offset, clip.duration)
                if sliced_sound:
                    # Find an available channel
                    self.current_channel = pygame.mixer.find_channel(True)
                    if self.current_channel:
                        # Apply volume to the channel
                        self.current_channel.set_volume(self.volume)
                        self.current_channel.play(sliced_sound)
                        self.current_sliced_sound = sliced_sound
                        clip.is_playing = True
                        debug_info(f"Track {self.name}: Starting playback of {clip.name} at {clip_offset:.1f}s")
                    else:
                        debug_info(f"Track {self.name}: No available channels for {clip.name}")
                else:
                    debug_info(f"Track {self.name}: Could not create sliced sound for {clip.name}")
            else:
                debug_info(f"Track {self.name}: {clip.name} has no make_slice_sound method")
        except Exception as e:
            debug_error(f"Track {self.name}: Error starting playback of {clip.name}: {e}")
    
    def _stop_audio_playback(self, clip: DAWClip = None):
        """Stop playing audio for a specific clip or all clips"""
        try:
            if self.current_channel:
                self.current_channel.stop()
                self.current_channel = None
            if self.current_sliced_sound:
                self.current_sliced_sound = None
            
            if clip:
                clip.is_playing = False
                debug_info(f"Track {self.name}: Stopping playback of {clip.name}")
            else:
                # Stop all clips
                for clip in self.clips:
                    clip.is_playing = False
                debug_info(f"Track {self.name}: Stopping all audio")
        except Exception as e:
            debug_error(f"Track {self.name}: Error stopping playback: {e}")
    
    def _find_current_clip(self, playhead_position: float):
        """Find which clip should be playing at the current playhead position"""
        self.current_clip = None
        
        for clip in self.clips:
            if (clip.start_time <= playhead_position <= 
                clip.start_time + clip.duration):
                self.current_clip = clip
                break
