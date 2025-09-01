import pygame
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from ..core.config import (
    WIDTH, HEIGHT, TEXT_COLOR, INFO_COLOR, UI_OUTLINE, HILITE, ERR,
    BG_COLOR, GRID, CLIP_FILL, CLIP_BORDER, PLAYHEAD
)
from ..core.debug import debug_enter, debug_exit, debug_info, debug_warning, debug_error
from ..audio.alien_track import AlienTrack
from ..audio.mixer import stop_all_audio
import time

class DAWClip:
    """Represents a clip in the DAW timeline"""
    
    def __init__(self, alien_track: AlienTrack, start_time: float, track_index: int, daw=None, duration: float = None, name: str = None):
        self.alien_track = alien_track
        self.start_time = start_time
        self.track_index = track_index
        self.daw = daw
        self.duration = duration if duration is not None else alien_track.duration
        self.name = name if name is not None else alien_track.name
        
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
        width = self.duration * pixels_per_second
        height = track_height
        
        # Draw clip background
        clip_color = CLIP_FILL if not self.is_selected else (255, 255, 100)  # Yellow when selected
        pygame.draw.rect(surface, clip_color, (x, y, width, height))
        pygame.draw.rect(surface, CLIP_BORDER, (x, y, width, height), 2)
        
        # Draw clip name
        from ..core.debug import debug_info
        try:
            font = pygame.font.Font(None, 20)
            text = font.render(self.name, True, (0, 0, 0))
            text_rect = text.get_rect(center=(x + width//2, y + height//2))
            surface.blit(text, text_rect)
        except Exception as e:
            debug_info(f"Could not render clip text: {e}")
    
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
            from ..core.debug import debug_error
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
            if self.daw and 0 <= self.track_index < len(self.daw.tracks):
                track = self.daw.tracks[self.track_index]
                if self in track.clips:
                    track.remove_clip(self)
                    from ..core.debug import debug_info
                    debug_info(f"Removed clip {self.name} from track {track.name}")
            
            # Clear references
            self.daw = None
            self.alien_track = None
            
        except Exception as e:
            from ..core.debug import debug_error
            debug_error(f"Error during clip deletion: {e}")
            import traceback
            traceback.print_exc()


class DAWTrack:
    """Represents a track in the DAW"""
    
    def __init__(self, name: str, color: tuple, track_index: int, daw=None):
        self.name = name
        self.color = color
        self.track_index = track_index
        self.daw = daw
        self.clips: List[DAWClip] = []
        
        # Track state
        self.is_muted = False
        self.is_soloed = False
        self.is_recording = False
        self.is_playing = False
        self.volume = 1.0
        self.pan = 0.0
        
        # Audio playback
        self.current_channel = None
        self.current_sliced_sound = None
    
    def add_clip(self, clip: DAWClip):
        """Add a clip to this track"""
        clip.track_index = self.track_index
        self.clips.append(clip)
    
    def remove_clip(self, clip: DAWClip):
        """Remove a clip from this track"""
        if clip in self.clips:
            self.clips.remove(clip)
    
    def has_content(self):
        """Check if this track has any clips"""
        return len(self.clips) > 0
    
    def toggle_mute(self):
        """Toggle mute state"""
        self.is_muted = not self.is_muted
    
    def toggle_solo(self):
        """Toggle solo state"""
        self.is_soloed = not self.is_soloed
    
    def toggle_record(self):
        """Toggle record state"""
        self.is_recording = not self.is_recording
    
    def start_playback(self, playhead_position: float):
        """Start playing this track from the given playhead position"""
        self.is_playing = True
        self.set_playhead_position(playhead_position)
    
    def stop_playback(self):
        """Stop playing this track"""
        self.is_playing = False
        if self.current_channel:
            try:
                self.current_channel.stop()
            except Exception:
                pass
            self.current_channel = None
        if self.current_sliced_sound:
            self.current_sliced_sound = None
    
    def set_playhead_position(self, position: float):
        """Set the playhead position for this track"""
        # Implementation for setting playhead position
        pass
    
    def update_playback(self, dt: float, playhead_position: float):
        """Update playback state for this track"""
        from ..core.debug import debug_info
        
        # Check if any clips should start playing
        for clip in self.clips:
            clip_start = clip.start_time
            clip_end = clip.start_time + clip.duration
            should_be_playing = clip_start <= playhead_position < clip_end
            
            debug_info(f"Track {self.name}: Clip {clip.name} - pos={playhead_position:.2f}, start={clip_start:.2f}, end={clip_end:.2f}, should_play={should_be_playing}, is_playing={clip.is_playing}")
            
            if should_be_playing and not clip.is_playing:
                debug_info(f"Track {self.name}: Starting clip {clip.name}")
                self._start_audio_playback(clip, playhead_position)
            elif not should_be_playing and clip.is_playing:
                debug_info(f"Track {self.name}: Stopping clip {clip.name}")
                self._stop_audio_playback(clip)
    
    def _start_audio_playback(self, clip: DAWClip, playhead_position: float):
        """Start playing audio for a specific clip"""
        try:
            # Check if track is muted - if so, don't play audio
            if self.is_muted:
                from ..core.debug import debug_info
                debug_info(f"Track {self.name}: Muted, not playing {clip.name}")
                return
            
            # Check if any other track is soloed - if so, only play if this track is also soloed
            if self.daw and self.daw._has_soloed_tracks():
                if not self.is_soloed:
                    from ..core.debug import debug_info
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
                        from ..core.debug import debug_info
                        debug_info(f"Track {self.name}: Starting playback of {clip.name} at {clip_offset:.1f}s")
                    else:
                        from ..core.debug import debug_info
                        debug_info(f"Track {self.name}: No available channels for {clip.name}")
                else:
                    from ..core.debug import debug_info
                    debug_info(f"Track {self.name}: Could not create sliced sound for {clip.name}")
            else:
                from ..core.debug import debug_info
                debug_info(f"Track {self.name}: {clip.name} has no make_slice_sound method")
        except Exception as e:
            from ..core.debug import debug_error
            debug_error(f"Track {self.name}: Error starting playback of {clip.name}: {e}")
    
    def _stop_audio_playback(self, clip: DAWClip):
        """Stop playing audio for a specific clip"""
        try:
            if self.current_channel:
                self.current_channel.stop()
                self.current_channel = None
            if self.current_sliced_sound:
                self.current_sliced_sound = None
            clip.is_playing = False
            from ..core.debug import debug_info
            debug_info(f"Track {self.name}: Stopping playback of {clip.name}")
        except Exception as e:
            from ..core.debug import debug_error
            debug_error(f"Track {self.name}: Error stopping playback of {clip.name}: {e}")


class CosmicDAW:
    """
    Cosmic Underground Digital Audio Workstation
    A comprehensive DAW designed for alien DJ mixing excellence
    """
    
    def __init__(self, fonts, game_state):
        self.FONT, self.SMALL, self.MONO = fonts
        self.game_state = game_state
        
        # Mix management system
        from .mix_manager import MixManager
        self.mix_manager = MixManager(game_state)
        
        # DAW dimensions and layout - now full screen
        self.daw_width = WIDTH
        self.daw_height = HEIGHT
        self.daw_x = 0
        self.daw_y = 0
        
        # Track system
        self.tracks: List[DAWTrack] = []
        self.max_tracks = 8
        self.track_height = 80
        self.track_spacing = 5
        
        # Waveform caching for performance
        self.waveform_cache: Dict[str, List[float]] = {}
        self.waveform_cache_size = 100  # Maximum number of cached waveforms
        
        # Timeline system - adjusted for larger screen
        self.timeline_width = self.daw_width - 250  # More space for track controls
        self.timeline_height = self.daw_height - 150  # More vertical space
        self.timeline_x = 250  # More space for track controls
        self.timeline_y = 150  # More space for header
        self.pixels_per_second = 50  # Timeline zoom
        self.max_duration = 1200.0  # Increased to 20 minutes
        
        # Playback system
        self.is_playing = False
        self.playhead_position = 0.0  # Current time in seconds
        self.playback_speed = 1.0
        self.loop_start = 0.0
        self.loop_end = 1200.0
        self.is_looping = False
        
        # Panning system
        self.timeline_offset = 0.0  # How much we've panned left/right
        self.min_timeline_offset = 0.0  # Can't pan past the beginning
        self.max_timeline_offset = 0.0  # Will be calculated based on content
        
        # Selection system
        self.selected_clips: List[DAWClip] = []
        self.selected_tracks: List[int] = []
        
        # Clip editing system
        self.clip_edit_mode = False  # True when editing clips
        self.clip_being_edited = None  # Currently edited clip
        self.edit_start_time = 0.0  # Start time for edit operations
        self.edit_end_time = 0.0  # End time for edit operations
        
        # Drag and drop system
        self.dragging_clip = None  # Clip being dragged
        self.potential_drag_clip = None  # Clip ready for dragging (after mouse movement)
        self.drag_start_pos = (0, 0)  # Starting position of drag
        self.drag_start_time = 0.0  # Starting time of drag
        self.drag_start_track = 0  # Starting track of drag
        
        # Clipboard system
        self.clipboard_clips: List[DAWClip] = []
        
        # Effects system
        self.available_effects = {
            "Reverb": {"wet": 0.3, "room_size": 0.5, "damping": 0.5},
            "Delay": {"time": 0.3, "feedback": 0.4, "mix": 0.5},
            "Distortion": {"amount": 0.2, "oversample": 2},
            "Filter": {"cutoff": 1000, "resonance": 0.5, "type": "lowpass"},
            "Compressor": {"threshold": -20, "ratio": 4, "attack": 0.003, "release": 0.25},
            "EQ": {"low": 0, "mid": 0, "high": 0}
        }
        
        # Mixing system
        self.master_volume = 1.0
        self.master_pan = 0.0
        self.master_effects = {}
        
        # UI state
        self.show_effects_panel = False
        self.show_mixer_panel = False
        self.show_timeline_panel = True
        self.show_transport_panel = True
        
        # Snap settings
        self.snap_to_grid = True
        self.grid_division = 0.25  # Quarter note snap
        
        # Undo/Redo system
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_steps = 50
        
        # Auto-save
        self.auto_save_interval = 30  # seconds
        self.last_auto_save = 0
        
        # Exit flag
        self.should_close = False
        
        # Initialize default tracks
        self._initialize_default_tracks()
        
        # Track selection menu
        self.show_track_menu = False
        self.selected_inventory_index = 0
        self.selected_daw_track_index = 0
        self.show_daw_track_selection = False
        
        # Seeking feedback system
        self.seeking_feedback = False
        self.seek_feedback_time = 0.0
        self.seek_feedback_position = 0.0
        
        # Initialize default tracks
        self._initialize_default_tracks()
    
    def reset_state(self):
        """Reset DAW state when closing"""
        debug_enter("reset_state", "cosmic_daw.py")
        print("🎵 DAW: Resetting state and stopping all audio")
        
        # Stop playback
        self.is_playing = False
        self.playhead_position = 0.0
        
        # Clear selections
        self.selected_clips.clear()
        self.selected_tracks.clear()
        self.should_close = False
        
        # Stop all tracks
        for track in self.tracks:
            track.stop_playback()
            track.set_playhead_position(0.0)
        
        # Force stop all audio globally
        try:
            pygame.mixer.stop()
            print("🎵 Stopped all pygame mixer audio on DAW close")
        except Exception as e:
            print(f"⚠️ Warning: Could not stop pygame mixer on close: {e}")
        
        print("🎵 DAW: State reset complete, all audio stopped")
        debug_exit("reset_state", "cosmic_daw.py")
    
    def _initialize_default_tracks(self):
        """Initialize default DAW tracks"""
        debug_enter("_initialize_default_tracks", "cosmic_daw.py")
        track_names = [f"Track {i + 1}" for i in range(self.max_tracks)]
        track_colors = [
            (255, 100, 100),  # Master - Red
            (255, 150, 50),   # Drums - Orange
            (100, 255, 100),  # Bass - Green
            (100, 100, 255),  # Lead - Blue
            (255, 100, 255),  # Pad - Magenta
            (100, 255, 255),  # FX - Cyan
            (255, 255, 100),  # Vocals - Yellow
            (150, 150, 150)   # Ambient - Gray
        ]
        
        for i in range(self.max_tracks):
            track = DAWTrack(
                name=track_names[i],
                color=track_colors[i],
                track_index=i,
                daw=self
            )
            self.tracks.append(track)
        debug_exit("_initialize_default_tracks", "cosmic_daw.py")
    
    def add_track_from_inventory(self, alien_track: AlienTrack, track_index: int = None):
        """Add a track from the player's inventory to the DAW"""
        debug_enter("add_track_from_inventory", "cosmic_daw.py", 
                   track_name=alien_track.name, track_index=track_index)
        
        if track_index is None:
            # Find first available track
            for i, track in enumerate(self.tracks):
                if not track.has_content():
                    track_index = i
                    break
        
        if track_index is not None and track_index < len(self.tracks):
            # Create a DAW clip from the alien track
            clip = DAWClip(
                alien_track=alien_track,
                start_time=0.0,
                track_index=track_index,
                daw=self
            )
            
            # Add clip to track
            self.tracks[track_index].add_clip(clip)
            
            # Auto-arrange clips to avoid overlap
            self._auto_arrange_clips()
            
            print(f"🎵 Added {alien_track.name} to {self.tracks[track_index].name} track!")
            print(f"   Duration: {alien_track.duration:.1f}s")
            print(f"   Best match: {alien_track.get_best_species_match()[0]}")
            
            debug_exit("add_track_from_inventory", "cosmic_daw.py", 
                      f"Added {alien_track.name} to track {track_index}")
            return True
        
        debug_exit("add_track_from_inventory", "cosmic_daw.py", "No available tracks")
        return False
    
    def _auto_arrange_clips(self):
        """Automatically arrange clips to avoid overlap"""
        for track in self.tracks:
            if len(track.clips) > 1:
                # Sort clips by start time
                track.clips.sort(key=lambda clip: clip.start_time)
                
                # Adjust start times to avoid overlap
                current_end = 0.0
                for clip in track.clips:
                    if clip.start_time < current_end:
                        clip.start_time = current_end
                    current_end = clip.start_time + clip.duration
    
    def _has_soloed_tracks(self):
        """Check if any tracks are soloed"""
        return any(track.is_soloed for track in self.tracks)
    
    def _handle_mute_solo_changes(self):
        """Handle changes in mute/solo states by updating audio playback"""
        for track in self.tracks:
            # If track is muted or affected by solo, stop its audio
            if track.is_muted or (self._has_soloed_tracks() and not track.is_soloed):
                if track.current_channel:
                    try:
                        track.current_channel.stop()
                        track.current_channel = None
                    except Exception as e:
                        from ..core.debug import debug_warning
                        debug_warning(f"Could not stop channel for muted/soloed track {track.name}: {e}")
                
                if track.current_sliced_sound:
                    track.current_sliced_sound = None
                
                # Mark clips as not playing
                for clip in track.clips:
                    clip.is_playing = False
    
    def _toggle_mute_selected_track(self):
        """Toggle mute for the track that has the most selected clips"""
        if not self.selected_clips:
            from ..core.debug import debug_warning
            debug_warning("No clips selected to determine track for mute toggle")
            return
        
        # Find the track with the most selected clips
        track_counts = {}
        for clip in self.selected_clips:
            track_counts[clip.track_index] = track_counts.get(clip.track_index, 0) + 1
        
        if track_counts:
            most_selected_track_index = max(track_counts, key=track_counts.get)
            track = self.tracks[most_selected_track_index]
            track.toggle_mute()
            from ..core.debug import debug_info
            debug_info(f"Track {track.name} mute toggled via keyboard: {'Muted' if track.is_muted else 'Unmuted'}")
            self._handle_mute_solo_changes()
    
    def _toggle_solo_selected_track(self):
        """Toggle solo for the track that has the most selected clips"""
        if not self.selected_clips:
            from ..core.debug import debug_warning
            debug_warning("No clips selected to determine track for solo toggle")
            return
        
        # Find the track with the most selected clips
        track_counts = {}
        for clip in self.selected_clips:
            track_counts[clip.track_index] = track_counts.get(clip.track_index, 0) + 1
        
        if track_counts:
            most_selected_track_index = max(track_counts, key=track_counts.get)
            track = self.tracks[most_selected_track_index]
            track.toggle_solo()
            from ..core.debug import debug_info
            debug_info(f"Track {track.name} solo toggled via keyboard: {'Soloed' if track.is_soloed else 'Unsoloed'}")
            self._handle_mute_solo_changes()
    
    def _test_mute_solo_functionality(self):
        """Test method to verify mute/solo functionality is working"""
        from ..core.debug import debug_info
        
        debug_info("=== Testing Mute/Solo Functionality ===")
        
        if not self.tracks:
            debug_info("No tracks available for testing")
            return
        
        # Test the first track
        track = self.tracks[0]
        debug_info(f"Testing track: {track.name}")
        debug_info(f"Initial state - Muted: {track.is_muted}, Soloed: {track.is_soloed}")
        
        # Test mute toggle
        track.toggle_mute()
        debug_info(f"After mute toggle - Muted: {track.is_muted}")
        
        # Test solo toggle
        track.toggle_solo()
        debug_info(f"After solo toggle - Soloed: {track.is_soloed}")
        
        # Test mute toggle again
        track.toggle_mute()
        debug_info(f"After second mute toggle - Muted: {track.is_muted}")
        
        # Test solo toggle again
        track.toggle_solo()
        debug_info(f"After second solo toggle - Soloed: {track.is_soloed}")
        
        # Test the mute/solo change handler
        debug_info("Calling _handle_mute_solo_changes()...")
        self._handle_mute_solo_changes()
        
        debug_info("=== Mute/Solo Test Complete ===")
    
    def _test_waveform_system(self):
        """Test method to verify waveform system is working"""
        from ..core.debug import debug_info
        
        debug_info("=== Testing Waveform System ===")
        
        if not self.tracks:
            debug_info("No tracks available for testing")
            return
        
        total_clips = sum(len(track.clips) for track in self.tracks)
        debug_info(f"Total clips in DAW: {total_clips}")
        
        for i, track in enumerate(self.tracks):
            debug_info(f"Track {i}: {track.name} - {len(track.clips)} clips")
            
            for j, clip in enumerate(track.clips):
                debug_info(f"  Clip {j}: {clip.name}")
                debug_info(f"    Duration: {clip.duration:.2f}s")
                debug_info(f"    Start time: {clip.start_time:.2f}s")
                debug_info(f"    Track index: {clip.track_index}")
                
                # Check alien_track
                if hasattr(clip, 'alien_track'):
                    debug_info(f"    Has alien_track: Yes")
                    if hasattr(clip.alien_track, 'array'):
                        debug_info(f"    Has array: Yes")
                        if clip.alien_track.array is not None:
                            debug_info(f"    Array shape: {clip.alien_track.array.shape}")
                            debug_info(f"    Array type: {clip.alien_track.array.dtype}")
                        else:
                            debug_info(f"    Array is None")
                    else:
                        debug_info(f"    Has array: No")
                else:
                    debug_info(f"    Has alien_track: No")
        
        debug_info("=== Waveform System Test Complete ===")
    
    def _create_test_clip_for_waveforms(self):
        """Create a test clip to verify waveform rendering works"""
        from ..core.debug import debug_info
        
        debug_info("=== Creating Test Clip for Waveform Testing ===")
        
        if not self.tracks:
            debug_info("No tracks available for testing")
            return
        
        # Create a simple test clip on the first track
        track = self.tracks[0]
        
        # Create a mock alien track with test audio data first
        import numpy as np
        
        # Generate 1 second of test audio (sine wave)
        sample_rate = 44100
        duration = 1.0
        samples = int(sample_rate * duration)
        
        # Create a simple sine wave
        t = np.linspace(0, duration, samples)
        frequency = 440  # A4 note
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.5
        
        # Convert to int16 format
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create a mock alien track
        class MockAlienTrack:
            def __init__(self, name, array):
                self.name = name
                self.array = array
                self.duration = len(array) / sample_rate
        
        mock_track = MockAlienTrack("TEST_WAVEFORM", audio_data)
        
        # Now create the DAWClip with the mock track
        test_clip = DAWClip(
            alien_track=mock_track,  # Pass the mock track directly
            start_time=2.0,
            track_index=0,
            daw=self
        )
        
        # Set the name explicitly
        test_clip.name = "TEST_WAVEFORM"
        
        # Add the clip to the track
        track.clips.append(test_clip)
        
        debug_info(f"Created test clip: {test_clip.name} on track {track.name}")
        debug_info(f"Clip duration: {test_clip.duration}s, start time: {test_clip.start_time}s")
        debug_info(f"Audio data shape: {test_clip.alien_track.array.shape}")
        debug_info(f"Track now has {len(track.clips)} clips")
        
        debug_info("=== Test Clip Created - Check for Waveforms ===")
    
    def _seek_audio_to_playhead(self, new_position: float):
        """Seek all playing audio to the new playhead position"""
        from ..core.debug import debug_info
        
        debug_info(f"Seeking audio to new playhead position: {new_position:.2f}s")
        
        # Store current playback state
        was_playing = self.is_playing
        
        # Stop all current audio playback
        for track in self.tracks:
            track.stop_playback()
        
        # Small delay to ensure audio stops completely
        import time
        time.sleep(0.01)
        
        # Restart playback from the new position if it was playing before
        if was_playing:
            debug_info("Restarting playback from new position")
            # Add visual feedback for seeking
            self._show_seek_feedback(new_position)
            self._start_playback()
        else:
            debug_info("Playback was not running - only visual playhead updated")
    
    def _show_seek_feedback(self, position: float):
        """Show visual feedback when seeking to a new position"""
        from ..core.debug import debug_info
        
        # Set a flag to show seeking feedback
        self.seeking_feedback = True
        self.seek_feedback_time = time.time()
        self.seek_feedback_position = position
        
        debug_info(f"Showing seek feedback at position {position:.2f}s")
        
        # The feedback will be rendered in the render method
    
    def _update_seek_feedback(self):
        """Update seeking feedback state - clear after timeout"""
        if self.seeking_feedback:
            current_time = time.time()
            if current_time - self.seek_feedback_time > 1.0:  # Show for 1 second
                self.seeking_feedback = False
    
    def _toggle_waveform_scaling_mode(self):
        """Toggle between different waveform scaling modes for testing"""
        from ..core.debug import debug_info
        
        if not hasattr(self, 'waveform_scaling_mode'):
            self.waveform_scaling_mode = 0
        
        self.waveform_scaling_mode = (self.waveform_scaling_mode + 1) % 5
        
        modes = [
            "Standard (70% log + 30% std)",
            "Logarithmic only",
            "Standard deviation only", 
            "Enhanced compression (60% log + 40% std + gamma 0.6)",
            "Quiet audio optimized (aggressive quiet section scaling)"
        ]
        
        debug_info(f"Waveform scaling mode: {modes[self.waveform_scaling_mode]}")
        
        # Clear waveform cache to force regeneration with new scaling
        self.clear_waveform_cache()
    
    def handle_event(self, ev, inventory: List[AlienTrack]):
        """Handle all DAW events"""
        # Store inventory for use in other methods
        self.current_inventory = inventory
        
        # Only log non-mouse events to reduce spam
        if ev.type != pygame.MOUSEMOTION:
            debug_enter("handle_event", "cosmic_daw.py", event_type=ev.type, event_key=getattr(ev, 'key', None))
        
        if ev.type == pygame.MOUSEBUTTONDOWN:
            self._handle_mouse_down(ev)
        elif ev.type == pygame.MOUSEBUTTONUP:
            self._handle_mouse_up(ev)
        elif ev.type == pygame.MOUSEMOTION:
            self._handle_mouse_motion(ev)
        elif ev.type == pygame.KEYDOWN:
            print(f"🎹 DAW received key: {ev.key} ({pygame.key.name(ev.key)})")
            self._handle_key_down(ev, inventory)
        elif ev.type == pygame.MOUSEWHEEL:
            self._handle_mouse_wheel(ev)
        
        # Check for scroll bar interaction
        if ev.type in [pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION]:
            self._handle_scroll_bar_interaction(ev)
        
        if ev.type != pygame.MOUSEMOTION:
            debug_exit("handle_event", "cosmic_daw.py")
    
    def _handle_mouse_down(self, ev):
        """Handle mouse button down events"""
        x, y = ev.pos
        
        from ..core.debug import debug_info
        debug_info(f"Mouse click at ({x}, {y}) in DAW")
        
        # Check if click is within DAW bounds (full screen)
        if not (0 <= x <= self.daw_width and 0 <= y <= self.daw_height):
            debug_info(f"Click outside DAW bounds: ({x}, {y}) not in (0, 0, {self.daw_width}, {self.daw_height})")
            return
        
        debug_info(f"Click within DAW bounds, processing...")
        
        # Handle timeline clicks
        if (self.timeline_x <= x <= self.timeline_x + self.timeline_width and
            self.timeline_y <= y <= self.timeline_y + self.timeline_height):
            debug_info(f"Timeline click at ({x}, {y})")
            if ev.button == 1:  # Left click
                self._handle_timeline_click(x, y, ev.button)
                # Check if we clicked on a clip to start dragging
                self._check_clip_drag_start(x, y)
            elif ev.button == 3:  # Right click
                self._handle_timeline_click(x, y, ev.button)
        
        # Handle track control clicks
        if 20 <= x <= 240:  # Track controls area (x=20 to x=240, matches _render_track_controls)
            debug_info(f"Track control click detected at ({x}, {y})")
            self._handle_track_control_click(x, y, ev.button)
        else:
            debug_info(f"Click at ({x}, {y}) not in track control area (20-240)")
    
    def _check_clip_drag_start(self, x: int, y: int):
        """Check if we should start dragging a clip"""
        # Convert screen coordinates to timeline coordinates
        rel_x = x - self.timeline_x
        rel_y = y - self.timeline_y
        
        # Calculate time position and track index - MUST account for timeline offset (panning)
        time_position = (rel_x / self.pixels_per_second) + self.timeline_offset
        track_index = int(rel_y / (self.track_height + self.track_spacing))
        
        if 0 <= track_index < len(self.tracks):
            track = self.tracks[track_index]
            for clip in track.clips:
                if (clip.start_time <= time_position <= 
                    clip.start_time + clip.duration):
                    # Store potential drag target but don't start dragging yet
                    # Dragging will only start if mouse moves significantly
                    self.drag_start_pos = (x, y)
                    self.drag_start_time = clip.start_time
                    self.drag_start_track = track_index
                    self.potential_drag_clip = clip  # New attribute to store potential drag target
                    print(f"🎵 Clicked on clip: {clip.name} (ready for drag if moved)")
                    break
    
    def _handle_mouse_up(self, ev):
        """Handle mouse button up events"""
        if self.dragging_clip:
            # Finish drag operation
            self._finish_clip_drag(ev.pos)
    
    def _finish_clip_drag(self, end_pos: tuple):
        """Finish dragging a clip to its new position"""
        if not self.dragging_clip:
            return
        
        x, y = end_pos
        
        # Check if we're still within DAW bounds
        if (self.daw_x <= x <= self.daw_x + self.daw_width and
            self.daw_y <= y <= self.daw_y + self.daw_height):
            
            # Convert to timeline coordinates
            rel_x = x - self.timeline_x
            rel_y = y - self.timeline_y
            
            # Calculate new time position and track index - MUST account for timeline offset (panning)
            new_time = (rel_x / self.pixels_per_second) + self.timeline_offset
            new_track_index = int(rel_y / (self.track_height + self.track_spacing))
            
            if 0 <= new_track_index < len(self.tracks):
                # Remove from old track
                old_track = self.tracks[self.drag_start_track]
                old_track.remove_clip(self.dragging_clip)
                
                # Add to new track
                new_track = self.tracks[new_track_index]
                self.dragging_clip.track_index = new_track_index
                self.dragging_clip.start_time = new_time
                new_track.add_clip(self.dragging_clip)
                
                # Auto-arrange to avoid overlap
                self._auto_arrange_clips()
                
                # Record action for undo AFTER the move and auto-arrange
                action = MoveClipAction(self, self.dragging_clip, 
                                      self.drag_start_track, self.drag_start_time,
                                      new_track_index, new_time)
                
                # Capture the final state after auto-arrange for proper undo
                action.capture_final_state()
                
                self.undo_stack.append(action)
                # Clear redo stack when new action is performed
                self.redo_stack.clear()
                
                print(f"🎵 Moved {self.dragging_clip.name} to {new_track.name} at {new_time:.2f}s")
            else:
                # Invalid track, restore original position
                old_track = self.tracks[self.drag_start_track]
                self.dragging_clip.start_time = self.drag_start_time
                old_track.add_clip(self.dragging_clip)
                print(f"🎵 Restored {self.dragging_clip.name} to original position")
        
        # Reset drag state
        self.dragging_clip = None
        self.potential_drag_clip = None
        self.drag_start_pos = (0, 0)
        self.drag_start_time = 0.0
        self.drag_start_track = 0
    
    def _handle_mouse_motion(self, ev):
        """Handle mouse motion events"""
        # Check if we should start dragging (mouse moved significantly from click AND button is pressed)
        if (hasattr(self, 'potential_drag_clip') and self.potential_drag_clip and 
            not self.dragging_clip and pygame.mouse.get_pressed()[0]):  # Left button pressed
            x, y = ev.pos
            start_x, start_y = self.drag_start_pos
            
            # Calculate distance moved
            distance = ((x - start_x) ** 2 + (y - start_y) ** 2) ** 0.5
            
            # Start dragging if moved more than 5 pixels
            if distance > 5:
                self.dragging_clip = self.potential_drag_clip
                self.potential_drag_clip = None
                print(f"🎵 Started dragging clip: {self.dragging_clip.name}")
        
        if self.dragging_clip:
            # Only update clip position if mouse button is still pressed
            if pygame.mouse.get_pressed()[0]:  # Left button still pressed
                # Update clip position during drag (visual feedback)
                x, y = ev.pos
                if (self.daw_x <= x <= self.daw_x + self.daw_width and
                    self.daw_y <= y <= self.daw_y + self.daw_height):
                    
                    # Convert to timeline coordinates
                    rel_x = x - self.timeline_x
                    rel_y = y - self.timeline_y
                    
                    # Calculate new time position - MUST account for timeline offset (panning)
                    new_time = (rel_x / self.pixels_per_second) + self.timeline_offset
                    
                    # Update clip position for visual feedback
                    self.dragging_clip.start_time = new_time
            else:
                # Mouse button released, stop dragging
                self.dragging_clip = None
                self.potential_drag_clip = None
                print(f"🎵 Stopped dragging (button released)")
        elif hasattr(self, 'potential_drag_clip') and self.potential_drag_clip:
            # Not dragging but have a potential drag clip - check if button is released
            if not pygame.mouse.get_pressed()[0]:  # Left button not pressed
                # Clear potential drag state if button is released
                self.potential_drag_clip = None
                print(f"🎵 Cleared potential drag state (button released)")
    
    def _handle_timeline_click(self, x, y, button):
        """Handle clicks on the timeline"""
        # Convert screen coordinates to timeline time
        rel_x = x - self.timeline_x
        rel_y = y - self.timeline_y
        
        # Calculate time position - MUST account for timeline offset (panning)
        # The timeline is panned by timeline_offset, so we need to add that to get the actual time
        time_position = (rel_x / self.pixels_per_second) + self.timeline_offset
        
        # Calculate track index
        track_index = int(rel_y / (self.track_height + self.track_spacing))
        
        if 0 <= track_index < len(self.tracks):
            if button == 1:  # Left click
                # Set playhead position
                self.playhead_position = time_position
                
                # Update the specific track's playhead position
                self.tracks[track_index].set_playhead_position(time_position)
                
                # CRITICAL: If playback is running, seek audio to new position
                if self.is_playing:
                    self._seek_audio_to_playhead(time_position)
                
                # Check for clip selection
                self._select_clip_at_position(track_index, time_position)
            elif button == 3:  # Right click
                # Context menu or delete clip
                self._show_context_menu(track_index, time_position)
    
    def _handle_track_control_click(self, x, y, button):
        """Handle clicks on track controls"""
        from ..core.debug import debug_info
        debug_info(f"Handling track control click at ({x}, {y})")
        
        # Calculate track index based on y position relative to track controls area
        track_y_start = 160  # This should match the y position in _render_track_controls
        track_index = int((y - track_y_start) / (self.track_height + self.track_spacing))
        
        debug_info(f"Calculated track index: {track_index} (y={y}, track_y_start={track_y_start})")
        
        if 0 <= track_index < len(self.tracks):
            track = self.tracks[track_index]
            debug_info(f"Processing click for track: {track.name}")
            
            # Track mute button (position 25, control_y, 20x20)
            if 25 <= x <= 45:
                debug_info(f"Mute button clicked for track {track.name}")
                track.toggle_mute()
                debug_info(f"Track {track.name} mute toggled: {'Muted' if track.is_muted else 'Unmuted'}")
                # Handle audio changes due to mute
                self._handle_mute_solo_changes()
            # Track solo button (position 50, control_y, 20x20)
            elif 50 <= x <= 70:
                debug_info(f"Solo button clicked for track {track.name}")
                track.toggle_solo()
                debug_info(f"Track {track.name} solo toggled: {'Soloed' if track.is_soloed else 'Unsoloed'}")
                # Handle audio changes due to solo
                self._handle_mute_solo_changes()
            # Track record button (position 75, control_y, 20x20)
            elif 75 <= x <= 95:
                debug_info(f"Record button clicked for track {track.name}")
                track.toggle_record()
            # Track effects button (position 95, control_y, 20x20)
            elif 95 <= x <= 115:
                debug_info(f"Effects button clicked for track {track.name}")
                self.show_effects_panel = not self.show_effects_panel
        else:
            debug_info(f"Track index {track_index} out of range (0-{len(self.tracks)-1})")
    
    def _handle_key_down(self, ev, inventory: List[AlienTrack]):
        """Handle keyboard events"""
        debug_enter("_handle_key_down", "cosmic_daw.py", key=ev.key, key_name=pygame.key.name(ev.key))
        
        # Debug: Show what modifiers are active
        mods = pygame.key.get_mods()
        debug_info(f"Key pressed: {pygame.key.name(ev.key)} | Modifiers: CTRL={bool(mods & pygame.KMOD_CTRL)}, SHIFT={bool(mods & pygame.KMOD_SHIFT)}, ALT={bool(mods & pygame.KMOD_ALT)}")
        
        # Check CTRL+SHIFT combinations FIRST (before CTRL only) to avoid conflicts
        if pygame.key.get_mods() & pygame.KMOD_CTRL and pygame.key.get_mods() & pygame.KMOD_SHIFT:
            if ev.key == pygame.K_s:
                debug_info("CTRL+SHIFT+S - saving mix")
                self.save_current_mix()
            elif ev.key == pygame.K_e:
                debug_info("CTRL+SHIFT+E - exporting mix to WAV")
                self.export_current_mix()
            elif ev.key == pygame.K_l:
                debug_info("CTRL+SHIFT+L - loading mix")
                self.load_saved_mix()
            else:
                debug_info(f"Unhandled CTRL+SHIFT+ key: {pygame.key.name(ev.key)}")
        # Check CTRL combinations (but not CTRL+SHIFT)
        elif pygame.key.get_mods() & pygame.KMOD_CTRL and not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            if ev.key == pygame.K_z:
                debug_info("CTRL+Z - undo")
                self.undo()
            elif ev.key == pygame.K_y:
                debug_info("CTRL+Y - redo")
                self.redo()
            elif ev.key == pygame.K_c:
                debug_info("CTRL+C - copying clips")
                self.copy_selected_clips()
            elif ev.key == pygame.K_v:
                debug_info("CTRL+V - pasting clips")
                self.paste_clips()
            elif ev.key == pygame.K_x:
                debug_info("CTRL+X - cutting clips")
                self.cut_selected_clips()
            elif ev.key == pygame.K_s:
                debug_info("CTRL+S - slicing clips at playhead")
                self.slice_clips_at_playhead()
            elif ev.key == pygame.K_r:
                debug_info("CTRL+R - removing time range from selected clips")
                if self.selected_clips:
                    # For now, remove the entire selected clips
                    # In the future, this could be enhanced to remove time ranges
                    self.remove_selected_clips()
                else:
                    debug_warning("No clips selected for removal")
            elif ev.key == pygame.K_i:
                debug_info("CTRL+I - inserting clip at playhead")
                self.insert_clip_at_playhead(inventory)
            elif ev.key == pygame.K_d:
                debug_info("CTRL+D - duplicating selected clips")
                self.duplicate_selected_clips()
            else:
                debug_info(f"Unhandled CTRL+ key: {pygame.key.name(ev.key)}")
        
        # Regular key handling (no CTRL modifier)
        else:
            if ev.key == pygame.K_SPACE:
                debug_info("SPACE key - toggling playback")
                self.toggle_playback()
            elif ev.key == pygame.K_r:
                debug_info("R key - recording")
                self.record()
            elif ev.key == pygame.K_s:
                debug_info("S key - stopping")
                self.stop()
            elif ev.key == pygame.K_m:
                debug_info("M key - merging selected clips")
                self.merge_selected_clips()
            elif ev.key == pygame.K_l:
                debug_info("L key - toggling loop")
                self.toggle_loop()
            elif ev.key == pygame.K_j:
                debug_info("J key - seeking to playhead position")
                if self.is_playing:
                    self._seek_audio_to_playhead(self.playhead_position)
                else:
                    debug_info("Playback not running - just updating visual playhead")
            elif ev.key == pygame.K_LEFT:
                debug_info("LEFT arrow key - panning left")
                self.pan_left_by_visible_amount()
            elif ev.key == pygame.K_RIGHT:
                debug_info("RIGHT arrow key - panning right")
                self.pan_right_by_visible_amount()
            elif ev.key == pygame.K_F3:
                debug_info("F3 key - toggle mute for selected track")
                self._toggle_mute_selected_track()
            elif ev.key == pygame.K_F4:
                debug_info("F4 key - toggle solo for selected track")
                self._toggle_solo_selected_track()
            elif ev.key == pygame.K_F5:
                debug_info("F5 key - test mute/solo functionality")
                self._test_mute_solo_functionality()
            elif ev.key == pygame.K_F6:
                debug_info("F6 key - clear waveform cache")
                self.clear_waveform_cache()
            elif ev.key == pygame.K_F7:
                debug_info("F7 key - test waveform system")
                self._test_waveform_system()
            elif ev.key == pygame.K_F8:
                debug_info("F8 key - create test clip for waveform testing")
                self._create_test_clip_for_waveforms()
            elif ev.key == pygame.K_F9:
                debug_info("F9 key - toggle waveform scaling mode")
                self._toggle_waveform_scaling_mode()
            elif ev.key == pygame.K_DELETE:
                debug_info("DELETE key - deleting selected clips")
                self.delete_selected_clips()
            elif ev.key == pygame.K_F1:
                debug_info("F1 key - Testing slice functionality")
                self.slice_clips_at_playhead()
            elif ev.key == pygame.K_F2:
                debug_info("F2 key - Testing slice functionality (alternative)")
                self.slice_clips_at_playhead()
            elif ev.key == pygame.K_EQUALS:
                debug_info("+ key - zooming in")
                self.zoom_in()
            elif ev.key == pygame.K_MINUS:
                debug_info("- key - zooming out")
                self.zoom_out()
            elif ev.key == pygame.K_i:
                debug_info("I key - showing track selection menu")
                # Show track selection menu
                self.show_track_menu = True
                self.selected_inventory_index = 0
                if not inventory:
                    debug_warning("No tracks in inventory! Collect some music first.")
                    self.show_track_menu = False
            # Number keys for quick track selection from inventory
            elif ev.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                if not self.show_track_menu and not self.show_daw_track_selection:
                    # Quick add track using number key
                    track_index = ev.key - pygame.K_1  # 0, 1, 2, 3, 4, 5, 6, 7, 8
                    if track_index < len(inventory):
                        selected_track = inventory[track_index]
                        debug_info(f"Quick add track {track_index + 1}: {selected_track.name}")
                        # Auto-select first available DAW track
                        for i, daw_track in enumerate(self.tracks):
                            if not daw_track.has_content():
                                self.add_track_from_inventory(selected_track, i)
                                debug_info(f"Added to DAW track {i + 1}")
                                break
                        else:
                            # If all tracks have content, add to first track
                            self.add_track_from_inventory(selected_track, 0)
                            debug_info("Added to DAW track 1 (all tracks full)")
                elif self.show_track_menu:
                    # Track menu is open - select inventory track
                    track_index = ev.key - pygame.K_1
                    if track_index < len(inventory):
                        self.selected_inventory_index = track_index
                        debug_info(f"Selected inventory track {track_index + 1}: {inventory[track_index].name}")
                        # Auto-select first available DAW track
                        for i, daw_track in enumerate(self.tracks):
                            if not daw_track.has_content():
                                self.selected_daw_track_index = i
                                break
                        else:
                            self.selected_daw_track_index = 0
                        debug_info(f"Auto-selected DAW track {self.selected_daw_track_index + 1}")
                        # Auto-add the track
                        selected_track = inventory[self.selected_inventory_index]
                        success = self.add_track_from_inventory(selected_track, self.selected_daw_track_index)
                        if success:
                            debug_info(f"Successfully added {selected_track.name} to DAW track {self.selected_daw_track_index + 1}!")
                        else:
                            debug_warning("Failed to add track to DAW!")
                        self.show_track_menu = False
            elif ev.key == pygame.K_ESCAPE:
                debug_info("ESC key - exiting DAW")
                # Exit DAW - signal to close
                self.should_close = True
            elif ev.key == pygame.K_UP and self.show_track_menu:
                # Navigate up in track menu
                if inventory:
                    self.selected_inventory_index = (self.selected_inventory_index - 1) % len(inventory)
                    debug_info(f"Selected track: {inventory[self.selected_inventory_index].name}")
            elif ev.key == pygame.K_DOWN and self.show_track_menu:
                # Navigate down in track menu
                if inventory:
                    self.selected_inventory_index = (self.selected_inventory_index + 1) % len(inventory)
                    debug_info(f"Selected track: {inventory[self.selected_inventory_index].name}")
            elif ev.key == pygame.K_RETURN and self.show_track_menu:
                # Select track and show DAW track selection
                if inventory:
                    selected_track = inventory[self.selected_inventory_index]
                    debug_info(f"Selected {selected_track.name} from inventory")
                    debug_info("Now choose which DAW track to add it to...")
                    self.show_track_menu = False
                    self.show_daw_track_selection = True
                    self.selected_daw_track_index = 0
            elif ev.key == pygame.K_ESCAPE and self.show_track_menu:
                # Close track menu
                debug_info("Closing track selection menu")
                self.show_track_menu = False
            elif ev.key == pygame.K_UP and self.show_daw_track_selection:
                # Navigate up in DAW track selection
                self.selected_daw_track_index = (self.selected_daw_track_index - 1) % len(self.tracks)
                debug_info(f"Selected DAW track: {self.tracks[self.selected_daw_track_index].name}")
            elif ev.key == pygame.K_DOWN and self.show_daw_track_selection:
                # Navigate down in DAW track selection
                self.selected_daw_track_index = (self.selected_daw_track_index + 1) % len(self.tracks)
                debug_info(f"Selected track: {self.tracks[self.selected_daw_track_index].name}")
            elif ev.key == pygame.K_RETURN and self.show_daw_track_selection:
                # Add track to selected DAW track
                if inventory:
                    selected_track = inventory[self.selected_inventory_index]
                    selected_daw_track = self.tracks[self.selected_daw_track_index]
                    debug_info(f"Adding {selected_track.name} to {selected_daw_track.name} track...")
                    success = self.add_track_from_inventory(selected_track, self.selected_daw_track_index)
                    if success:
                        debug_info(f"Successfully added {selected_track.name} to {selected_daw_track.name} track!")
                    else:
                        debug_warning("Failed to add track to DAW - track is full!")
                    self.show_daw_track_selection = False
            elif ev.key == pygame.K_ESCAPE and self.show_daw_track_selection:
                # Cancel DAW track selection
                debug_info("Cancelled DAW track selection")
                self.show_daw_track_selection = False
            else:
                debug_info(f"Unhandled key: {pygame.key.name(ev.key)}")
        
        debug_exit("_handle_key_down", "cosmic_daw.py")
    
    def _handle_mouse_wheel(self, ev):
        """Handle mouse wheel events for zooming and panning"""
        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
            # SHIFT + wheel = horizontal panning
            pan_amount = 5.0  # seconds to pan per wheel click
            if ev.y > 0:
                # Pan left (earlier in time)
                self.pan_left(pan_amount)
            else:
                # Pan right (later in time)
                self.pan_right(pan_amount)
        else:
            # Regular wheel = zooming
            if ev.y > 0:
                self.zoom_in()
            else:
                self.zoom_out()
    
    def toggle_playback(self):
        """Toggle playback on/off"""
        debug_enter("toggle_playback", "cosmic_daw.py")
        debug_info(f"Playback toggled: is_playing={self.is_playing} -> {not self.is_playing}")
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            debug_info("Starting DAW playback")
            self._start_playback()
        else:
            debug_info("Stopping DAW playback")
            self._stop_playback()
        
        debug_exit("toggle_playback", "cosmic_daw.py")
    
    def _start_playback(self):
        """Start DAW playback"""
        debug_enter("_start_playback", "cosmic_daw.py")
        
        # Reset all clips to not playing state to ensure clean restart
        for track in self.tracks:
            for clip in track.clips:
                clip.is_playing = False
        
        # Start playing all tracks
        for track in self.tracks:
            track.start_playback(self.playhead_position)
        
        # Reset playhead if at end
        if self.playhead_position >= self.max_duration:
            self.playhead_position = 0.0
            
        debug_exit("_start_playback", "cosmic_daw.py")
    
    def _stop_playback(self):
        """Stop DAW playback"""
        debug_enter("_stop_playback", "cosmic_daw.py")
        debug_info("Stopping all DAW tracks")
        
        # Stop all tracks and reset their playback state
        for track in self.tracks:
            track.stop_playback()
            # Reset all clips to not playing state
            for clip in track.clips:
                clip.is_playing = False
        
        # Force stop all pygame mixer audio globally to ensure nothing continues
        try:
            pygame.mixer.stop()
            debug_info("Global pygame mixer stop completed")
            
            # Small delay to let pygame process the stop command
            time.sleep(0.01)
            
            # Stop again to catch any delayed audio
            pygame.mixer.stop()
            debug_info("Second global pygame mixer stop completed")
            
        except Exception as e:
            debug_warning(f"Could not stop global pygame mixer: {e}")
        
        debug_exit("_stop_playback", "cosmic_daw.py")
    
    def record(self):
        """Start recording"""
        # Implementation for recording functionality
        pass
    
    def stop(self):
        """Stop playback and recording"""
        print("🎵 DAW: Stopping all playback and resetting playhead")
        self.is_playing = False
        self.playhead_position = 0.0  # Reset playhead to start
        
        # Stop all tracks and reset their playback state
        for track in self.tracks:
            track.stop_playback()
            track.set_playhead_position(0.0)  # Reset track playhead
        
        # Stop all audio globally to ensure no background audio continues
        try:
            pygame.mixer.stop()
            print("🎵 Stopped all pygame mixer audio")
        except Exception as e:
            print(f"⚠️ Warning: Could not stop pygame mixer: {e}")
        
        print("🎵 DAW: All playback stopped and playhead reset to 0.0s")
    
    def toggle_loop(self):
        """Toggle loop mode"""
        self.is_looping = not self.is_looping
    
    def undo(self):
        """Undo last action"""
        if self.undo_stack:
            action = self.undo_stack.pop()
            self.redo_stack.append(action)
            action.undo()
    
    def redo(self):
        """Redo last undone action"""
        if self.redo_stack:
            action = self.redo_stack.pop()
            self.undo_stack.append(action)
            action.redo()
    
    def delete_selected_clips(self):
        """Delete selected clips"""
        print(f"🎵 DAW: Deleting {len(self.selected_clips)} selected clips")
        
        # Record action for undo
        if self.selected_clips:
            action = DeleteClipsAction(self, self.selected_clips.copy())
            self.undo_stack.append(action)
            # Clear redo stack when new action is performed
            self.redo_stack.clear()
        
        # Stop any audio that might be playing from these clips
        for clip in self.selected_clips:
            # If this clip is currently playing, stop its audio
            if clip.alien_track.sound:
                try:
                    clip.alien_track.sound.stop()
                    print(f"🎵 Stopped audio for deleted clip: {clip.name}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not stop audio for {clip.name}: {e}")
        
        # Delete the clips
        for clip in self.selected_clips:
            clip.delete()
        
        # Clear selection
        self.selected_clips.clear()
        
        # Force stop all audio to ensure no orphaned audio continues
        try:
            pygame.mixer.stop()
            print("🎵 Stopped all audio after clip deletion")
        except Exception as e:
            print(f"⚠️ Warning: Could not stop all audio: {e}")
        
        print(f"✅ Deleted {len(self.selected_clips)} clips and stopped all audio")
    
    def cut_selected_clips(self):
        """Cut selected clips to clipboard and remove from timeline"""
        print(f"🎵 DAW: Cutting {len(self.selected_clips)} clips to clipboard")
        
        # Record action for undo
        if self.selected_clips:
            action = CutClipsAction(self, self.selected_clips.copy())
            self.undo_stack.append(action)
            # Clear redo stack when new action is performed
            self.redo_stack.clear()
        
        # Stop any audio that might be playing from these clips
        for clip in self.selected_clips:
            # If this clip is currently playing, stop its audio
            if clip.alien_track.sound:
                try:
                    clip.alien_track.sound.stop()
                    print(f"🎵 Stopped audio for cut clip: {clip.name}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not stop audio for {clip.name}: {e}")
        
        self.clipboard_clips.clear()
        for clip in self.selected_clips:
            # Copy to clipboard
            self.clipboard_clips.append(clip)
            # Remove from timeline
            clip.delete()
        
        # Clear selection
        self.selected_clips.clear()
        
        # Force stop all audio to ensure no orphaned audio continues
        try:
            pygame.mixer.stop()
            print("🎵 Stopped all audio after cutting clips")
        except Exception as e:
            print(f"⚠️ Warning: Could not stop all audio: {e}")
        
        print(f"🎵 Cut {len(self.clipboard_clips)} clips to clipboard and stopped all audio")
    
    def copy_selected_clips(self):
        """Copy selected clips to clipboard"""
        self.clipboard_clips.clear()
        for clip in self.selected_clips:
            # Create a copy for clipboard
            copied_clip = clip.duplicate()
            copied_clip.start_time = 0.0  # Reset start time for pasting
            self.clipboard_clips.append(copied_clip)
        print(f"🎵 Copied {len(self.clipboard_clips)} clips to clipboard")
    
    def paste_clips(self):
        """Paste clips from clipboard at current playhead position"""
        if not self.clipboard_clips:
            print("❌ No clips in clipboard!")
            return
        
        # Record action for undo
        action = PasteClipsAction(self, self.clipboard_clips.copy(), self.playhead_position)
        self.undo_stack.append(action)
        # Clear redo stack when new action is performed
        self.redo_stack.clear()
        
        # Paste each clip at the current playhead position
        for i, clip in enumerate(self.clipboard_clips):
            # Create new clip instance
            new_clip = DAWClip(
                alien_track=clip.alien_track,
                start_time=self.playhead_position + (i * 0.1),  # Offset each clip slightly
                track_index=clip.track_index,
                daw=self
            )
            
            # Add to the appropriate track
            if 0 <= new_clip.track_index < len(self.tracks):
                self.tracks[new_clip.track_index].add_clip(new_clip)
                print(f"🎵 Pasted {new_clip.name} at {new_clip.start_time:.2f}s")
        
        # Auto-arrange to avoid overlap
        self._auto_arrange_clips()
        print(f"✅ Pasted {len(self.clipboard_clips)} clips from clipboard")
    
    def slice_clips_at_playhead(self):
        """
        Slice all clips at the current playhead position
        
        Returns:
            bool: True if any clips were successfully sliced, False otherwise
        """
        debug_info(f"🎵 Attempting to slice clips at playhead position: {self.playhead_position:.2f}s")
        debug_info(f"🎵 Selected clips: {len(self.selected_clips)}")
        
        if not self.selected_clips:
            debug_info("No clips selected for slicing")
            return False
        
        # Import validation functions
        from ..core.daw_validation import validate_clip_list, safe_validate
        
        # Validate that we have clips to work with
        is_valid, error_msg = safe_validate(validate_clip_list, self.selected_clips, "Slice")
        if not is_valid:
            debug_warning(f"Slice validation failed: {error_msg}")
            return False
        
        # Record action for undo
        action = SliceClipsAction(self, self.selected_clips.copy(), self.playhead_position)
        
        debug_info(f"Slicing {len(self.selected_clips)} clips at {self.playhead_position:.2f}s")
        
        sliced_count = 0
        for clip in self.selected_clips:
            debug_info(f"Attempting to slice clip: {clip.name}")
            
            if self._slice_clip_at_time(clip, self.playhead_position):
                sliced_count += 1
                debug_info(f"Successfully sliced {clip.name}")
            else:
                debug_warning(f"Failed to slice {clip.name}")
        
        if sliced_count > 0:
            # Capture the created slices for undo
            action.capture_created_slices()
            self.undo_stack.append(action)
            # Clear redo stack when new action is performed
            self.redo_stack.clear()
            
            # Clear waveform cache since clips have changed
            self.clear_waveform_cache()
            
            debug_info(f"Sliced {sliced_count} clips at {self.playhead_position:.2f}s")
            return True
        else:
            debug_warning("No clips could be sliced at current position")
            return False
    
    def merge_selected_clips(self):
        """
        Merge selected clips into a single clip
        
        Returns:
            bool: True if merge was successful, False otherwise
            
        Raises:
            DAWValidationError: If validation fails
            RuntimeError: If audio processing fails
        """
        try:
            debug_info(f"🎵 Attempting to merge {len(self.selected_clips)} selected clips")
            
            # Import validation functions
            from ..core.daw_validation import validate_merge_operation, safe_validate
            
            # Validate the merge operation
            is_valid, error_msg = safe_validate(validate_merge_operation, self.selected_clips, "Merge")
            if not is_valid:
                debug_warning(f"Merge validation failed: {error_msg}")
                return False
            
            # Sort clips by start time for consistent processing
            sorted_clips = sorted(self.selected_clips, key=lambda c: c.start_time)
            
            # Record action for undo
            action = MergeClipsAction(self, sorted_clips.copy())
            
            # Perform the merge
            merged_clip = self._merge_clips(sorted_clips)
            if merged_clip:
                # Store the merged clip in the action for undo
                action.merged_clip = merged_clip
                
                # Add to undo stack
                self.undo_stack.append(action)
                # Clear redo stack when new action is performed
                self.redo_stack.clear()
                debug_info(f"Successfully merged {len(sorted_clips)} clips")
                return True
            else:
                debug_error("Failed to merge clips")
                return False
                
        except Exception as e:
            debug_error(f"Error during merge operation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _merge_clips(self, clips: List['DAWClip']) -> 'DAWClip':
        """Merge multiple clips into a single clip"""
        try:
            if not clips:
                return None
            
            # Get the track
            track = self.tracks[clips[0].track_index]
            
            # Calculate merged clip properties
            start_time = clips[0].start_time
            total_duration = sum(clip.duration for clip in clips)
            
            # Create a new merged clip
            merged_clip = self._create_merged_clip(clips, start_time, total_duration)
            if not merged_clip:
                return None
            
            # Remove original clips
            for clip in clips:
                track.remove_clip(clip)
                if clip in self.selected_clips:
                    self.selected_clips.remove(clip)
            
            # Add merged clip
            track.add_clip(merged_clip)
            self.selected_clips.append(merged_clip)
            
            # Auto-arrange to ensure no overlap
            self._auto_arrange_clips()
            
            # Clear waveform cache since clips have changed
            self.clear_waveform_cache()
            
            return merged_clip
            
        except Exception as e:
            debug_error(f"Error in _merge_clips: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_merged_clip(self, clips: List['DAWClip'], start_time: float, total_duration: float) -> 'DAWClip':
        """Create a new clip by merging multiple clips"""
        try:
            # Create a name for the merged clip
            base_name = clips[0].name.split('_')[0]  # Get base name without slice numbers
            merged_name = f"{base_name}_merged"
            
            debug_info(f"Creating merged clip: {merged_name} ({total_duration:.2f}s) from {len(clips)} clips")
            
            # Concatenate the audio data from all clips
            from ..core.config import SR
            import numpy as np
            
            # Calculate total samples needed
            total_samples = int(total_duration * SR)
            
            # Get the number of channels from the first clip
            if not hasattr(clips[0].alien_track, 'array') or clips[0].alien_track.array is None:
                debug_error(f"First clip {clips[0].name} has no audio array")
                return None
            
            num_channels = clips[0].alien_track.array.shape[1] if len(clips[0].alien_track.array.shape) > 1 else 1
            
            # Create empty array for merged audio
            if num_channels == 1:
                merged_array = np.zeros(total_samples, dtype=clips[0].alien_track.array.dtype)
            else:
                merged_array = np.zeros((total_samples, num_channels), dtype=clips[0].alien_track.array.dtype)
            
            # Fill the merged array with audio data from each clip
            current_sample = 0
            for clip in clips:
                if not hasattr(clip.alien_track, 'array') or clip.alien_track.array is None:
                    debug_warning(f"Clip {clip.name} has no audio array, skipping")
                    continue
                
                clip_samples = clip.alien_track.array.shape[0]
                if current_sample + clip_samples <= total_samples:
                    if num_channels == 1:
                        merged_array[current_sample:current_sample + clip_samples] = clip.alien_track.array.flatten()
                    else:
                        merged_array[current_sample:current_sample + clip_samples, :] = clip.alien_track.array
                    current_sample += clip_samples
                    debug_info(f"Added {clip_samples} samples from {clip.name}")
                else:
                    debug_warning(f"Clip {clip.name} would exceed total duration, truncating")
                    remaining_samples = total_samples - current_sample
                    if remaining_samples > 0:
                        if num_channels == 1:
                            merged_array[current_sample:] = clip.alien_track.array[:remaining_samples].flatten()
                        else:
                            merged_array[current_sample:, :] = clip.alien_track.array[:remaining_samples, :]
                        current_sample = total_samples
            
            # Create a new AlienTrack with the merged audio
            from ..audio.alien_track import AlienTrack
            merged_track = AlienTrack(
                name=merged_name,
                array=merged_array,
                source_zone=clips[0].alien_track.source_zone,
                is_bootleg=clips[0].alien_track.is_bootleg
            )
            
            # Create a new DAWClip with the merged track
            merged_clip = DAWClip(
                alien_track=merged_track,
                start_time=start_time,
                track_index=clips[0].track_index,
                daw=self,
                duration=total_duration,
                name=merged_name
            )
            
            # Store reference to original clips for potential undo
            merged_clip.original_clips = clips
            
            debug_info(f"Successfully created merged clip: {merged_name} with {current_sample} samples")
            return merged_clip
            
        except Exception as e:
            debug_error(f"Failed to create merged clip: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _slice_clip_at_time(self, clip: 'DAWClip', slice_time: float) -> bool:
        """Slice a clip at a specific time, creating two new clips with proper audio slicing"""
        # Check if slice time is within clip bounds
        if not (clip.start_time < slice_time < clip.start_time + clip.duration):
            return False
        
        # Calculate new durations
        first_duration = slice_time - clip.start_time
        second_duration = clip.duration - first_duration
        
        if first_duration < 0.1 or second_duration < 0.1:  # Minimum clip duration
            return False
        
        debug_info(f"Slicing {clip.name} at {slice_time:.2f}s")
        debug_info(f"First clip: {first_duration:.2f}s, Second clip: {second_duration:.2f}s")
        
        # Create first clip (original start to slice point)
        first_clip = self._create_sliced_clip(
            clip, 
            0.0,  # Start from beginning of the original audio
            first_duration, 
            f"{clip.name}_1",
            clip.start_time  # Timeline position: start where original clip started
        )
        
        # Create second clip (slice point to end)
        # The start_time should be the offset from the original clip's audio start
        second_clip = self._create_sliced_clip(
            clip, 
            first_duration,  # Start from where we sliced in the original audio
            second_duration, 
            f"{clip.name}_2",
            slice_time  # Timeline position: start at the slice point
        )
        
        if not first_clip or not second_clip:
            debug_error("Failed to create sliced clips")
            return False
        
        # Remove original clip and add new ones
        track = self.tracks[clip.track_index]
        track.remove_clip(clip)
        track.add_clip(first_clip)
        track.add_clip(second_clip)
        
        # Update selection
        if clip in self.selected_clips:
            self.selected_clips.remove(clip)
            self.selected_clips.extend([first_clip, second_clip])
        
        debug_info(f"Successfully sliced {clip.name} into two clips")
        return True
    
    def _create_sliced_clip(self, original_clip: 'DAWClip', audio_start_time: float, duration: float, name: str, timeline_start_time: float) -> 'DAWClip':
        """Create a new clip with sliced audio data"""
        try:
            debug_info(f"Creating sliced clip: {name} ({duration:.2f}s) from audio offset {audio_start_time:.2f}s, timeline position {timeline_start_time:.2f}s")
            
            # Get the original audio array
            if not hasattr(original_clip.alien_track, 'array'):
                debug_error(f"Original clip {original_clip.name} has no audio array")
                return None
            
            # Calculate sample positions
            from ..core.config import SR
            start_sample = int(audio_start_time * SR)
            end_sample = int((audio_start_time + duration) * SR)
            
            # Ensure bounds are valid
            max_samples = original_clip.alien_track.array.shape[0]
            start_sample = max(0, min(max_samples - 1, start_sample))
            end_sample = max(start_sample + 1, min(max_samples, end_sample))
            
            # Slice the audio array
            sliced_array = original_clip.alien_track.array[start_sample:end_sample, :].copy()
            
            # Create a new AlienTrack with the sliced audio
            from ..audio.alien_track import AlienTrack
            sliced_track = AlienTrack(
                name=name,
                array=sliced_array,
                source_zone=original_clip.alien_track.source_zone,
                is_bootleg=original_clip.alien_track.is_bootleg
            )
            
            # Create a new DAWClip with the sliced track
            new_clip = DAWClip(
                alien_track=sliced_track,
                start_time=timeline_start_time,  # Use the timeline position, not the audio offset
                track_index=original_clip.track_index,
                daw=self
            )
            
            # Set the new duration
            new_clip.duration = duration
            new_clip.name = name
            
            debug_info(f"Successfully created sliced clip: {name}")
            return new_clip
            
        except Exception as e:
            debug_error(f"Error creating sliced clip: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def remove_selected_clips(self):
        """Remove selected clips from timeline"""
        print(f"🎵 DAW: Removing {len(self.selected_clips)} selected clips")
        
        # Stop any audio that might be playing from these clips
        for clip in self.selected_clips:
            # If this clip is currently playing, stop its audio
            if clip.alien_track.sound:
                try:
                    clip.alien_track.sound.stop()
                    print(f"🎵 Stopped audio for removed clip: {clip.name}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not stop audio for {clip.name}: {e}")
        
        removed_count = len(self.selected_clips)
        for clip in self.selected_clips:
            clip.delete()
        
        # Clear selection
        self.selected_clips.clear()
        
        # Force stop all audio to ensure no orphaned audio continues
        try:
            pygame.mixer.stop()
            print("🎵 Stopped all audio after removing clips")
        except Exception as e:
            print(f"⚠️ Warning: Could not stop all audio: {e}")
        
        print(f"✅ Removed {removed_count} clips from timeline and stopped all audio")
    
    def insert_clip_at_playhead(self, inventory: List[AlienTrack]):
        """Insert a clip from inventory at the current playhead position"""
        if not inventory:
            print("❌ No tracks in inventory!")
            return
        
        # Use the first track in inventory
        selected_track = inventory[0]
        
        # Find an empty track to insert into
        target_track_index = None
        for i, track in enumerate(self.tracks):
            if not track.has_content():
                target_track_index = i
                break
        
        if target_track_index is None:
            print("❌ All tracks are full! Remove some clips first.")
            return
        
        # Create new clip at playhead position
        new_clip = DAWClip(
            alien_track=selected_track,
            start_time=self.playhead_position,
            track_index=target_track_index,
            daw=self
        )
        
        # Add to track
        self.tracks[target_track_index].add_clip(new_clip)
        
        # Auto-arrange to avoid overlap
        self._auto_arrange_clips()
        
        print(f"✅ Inserted {selected_track.name} at {self.playhead_position:.2f}s in {self.tracks[target_track_index].name} track")
    
    def create_clip_from_time_range(self, original_clip: 'DAWClip', start_time: float, end_time: float, name: str = None) -> 'DAWClip':
        """Create a new clip from a specific time range of an existing clip"""
        if not name:
            name = f"Clip_{original_clip.name}_{start_time:.1f}s_{end_time:.1f}s"
        
        duration = end_time - start_time
        if duration < 0.1:  # Minimum clip duration
            print(f"❌ Clip duration too short: {duration:.2f}s")
            return None
        
        return self._create_sliced_clip(original_clip, start_time, duration, name)
    
    def duplicate_selected_clips(self):
        """Duplicate selected clips"""
        if not self.selected_clips:
            print("❌ No clips selected for duplication!")
            return
        
        # Record action for undo
        if self.selected_clips:
            action = DuplicateClipsAction(self, self.selected_clips.copy())
            self.undo_stack.append(action)
            # Clear redo stack when new action is performed
            self.redo_stack.clear()
        
        duplicated_count = 0
        for clip in self.selected_clips:
            if clip.duplicate():
                duplicated_count += 1
        
        if duplicated_count > 0:
            print(f"✅ Duplicated {duplicated_count} clips")
            # Auto-arrange to avoid overlap
            self._auto_arrange_clips()
        else:
            print("❌ Failed to duplicate clips")
    
    def zoom_in(self):
        """Zoom in on timeline"""
        self.pixels_per_second = min(200, self.pixels_per_second * 1.2)
    
    def zoom_out(self):
        """Zoom out on timeline"""
        self.pixels_per_second = max(10, self.pixels_per_second / 1.2)
    
    def pan_left(self, amount: float = 5.0):
        """Pan timeline left (earlier in time)"""
        self.timeline_offset = max(self.min_timeline_offset, self.timeline_offset - amount)
        self._update_max_pan_offset()
    
    def pan_right(self, amount: float = 5.0):
        """Pan timeline right (later in time)"""
        self.timeline_offset = min(self.max_timeline_offset, self.timeline_offset + amount)
    
    def pan_left_by_visible_amount(self):
        """Pan timeline left by 1/4 of the visible time"""
        visible_duration = self.timeline_width / self.pixels_per_second
        pan_amount = visible_duration * 0.25
        self.pan_left(pan_amount)
    
    def pan_right_by_visible_amount(self):
        """Pan timeline right by 1/4 of the visible time"""
        visible_duration = self.timeline_width / self.pixels_per_second
        pan_amount = visible_duration * 0.25
        self.pan_right(pan_amount)
    
    def _update_max_pan_offset(self):
        """Update the maximum pan offset based on content and zoom level"""
        # Calculate how much content we can see
        visible_duration = self.timeline_width / self.pixels_per_second
        
        # Calculate how much we can pan right (to see later content)
        content_duration = max(self.max_duration, self._get_content_duration())
        self.max_timeline_offset = max(0, content_duration - visible_duration)
        
        # Ensure current offset is within bounds
        self.timeline_offset = min(self.max_timeline_offset, self.timeline_offset)
    
    def _get_content_duration(self) -> float:
        """Get the duration of all content in the timeline"""
        max_end_time = 0
        for track in self.tracks:
            for clip in track.clips:
                end_time = clip.start_time + clip.duration
                max_end_time = max(max_end_time, end_time)
        return max_end_time + 10  # Add some padding
    
    def _handle_scroll_bar_interaction(self, ev):
        """Handle scroll bar mouse interactions"""
        x, y = ev.pos
        
        # Check if click is on scroll bar area
        scroll_height = 20
        scroll_y = self.timeline_y - scroll_height - 5
        
        if not (self.timeline_x <= x <= self.timeline_x + self.timeline_width and
                scroll_y <= y <= scroll_y + scroll_height):
            return
        
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            # Left click on scroll bar - jump to position and start drag
            self._scroll_bar_click(x)
            self.scroll_bar_dragging = True
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            # End scroll bar drag
            self.scroll_bar_dragging = False
        elif ev.type == pygame.MOUSEMOTION and getattr(self, 'scroll_bar_dragging', False):
            # Drag scroll bar
            self._scroll_bar_drag(x)
    
    def _scroll_bar_click(self, x: int):
        """Handle click on scroll bar to jump to position"""
        # Calculate total content duration
        total_duration = max(self.max_duration, self._get_content_duration())
        if total_duration <= 0:
            return
        
        # Calculate click position as percentage of scroll bar
        click_percent = (x - self.timeline_x) / self.timeline_width
        
        # Set timeline offset based on click position
        new_offset = click_percent * total_duration
        self.timeline_offset = max(0, min(self.max_timeline_offset, new_offset))
        
        # Update max pan offset
        self._update_max_pan_offset()
    
    def _scroll_bar_drag(self, x: int):
        """Handle dragging the scroll bar"""
        # Calculate total content duration
        total_duration = max(self.max_duration, self._get_content_duration())
        if total_duration <= 0:
            return
        
        # Calculate drag position as percentage of scroll bar
        drag_percent = (x - self.timeline_x) / self.timeline_width
        
        # Set timeline offset based on drag position
        new_offset = drag_percent * total_duration
        self.timeline_offset = max(0, min(self.max_timeline_offset, new_offset))
        
        # Update max pan offset
        self._update_max_pan_offset()
    
    def update(self, dt):
        """Update DAW state"""
        if self.is_playing:
            # Update playhead position based on playback speed and time
            old_position = self.playhead_position
            self.playhead_position += dt * self.playback_speed
            
            # Handle looping
            if self.is_looping and self.playhead_position >= self.loop_end:
                self.playhead_position = self.loop_start
                # Reset all tracks for looping
                for track in self.tracks:
                    track.set_playhead_position(self.loop_start)
            elif self.playhead_position >= self.max_duration:
                # Reached end of timeline
                if self.is_looping:
                    # Loop back to start
                    self.playhead_position = 0.0
                    for track in self.tracks:
                        track.set_playhead_position(0.0)
                    print("🎵 DAW: Looping back to start")
                else:
                    # Stop playback
                    self.stop()
                    print("🎵 DAW: Reached end of timeline, stopping playback")
                    return
            
            # Update track playback with current playhead position
            for track in self.tracks:
                track.update_playback(dt, self.playhead_position)
        
        # Auto-save
        self.last_auto_save += dt
        if self.last_auto_save >= self.auto_save_interval:
            self._auto_save()
            self.last_auto_save = 0
    
    def _auto_save(self):
        """Auto-save current project"""
        # Implementation for auto-save functionality
        pass
    
    def add_track_from_inventory(self, alien_track: AlienTrack, daw_track_index: int) -> bool:
        """Add a track from inventory to a DAW track"""
        if daw_track_index >= len(self.tracks):
            return False
        
        # Create a new DAWClip from the AlienTrack
        new_clip = DAWClip(
            alien_track=alien_track,
            start_time=self.playhead_position,  # Start at current playhead position
            track_index=daw_track_index,
            daw=self
        )
        
        # Record action for undo
        action = AddClipAction(self, new_clip, daw_track_index)
        self.undo_stack.append(action)
        # Clear redo stack when new action is performed
        self.redo_stack.clear()
        
        # Add the clip to the track
        self.tracks[daw_track_index].add_clip(new_clip)
        
        # Auto-arrange to avoid overlap
        self._auto_arrange_clips()
        
        # Update pan offset limits after adding content
        self._update_max_pan_offset()
        
        print(f"🎵 Added {alien_track.name} to {self.tracks[daw_track_index].name} track at {self.playhead_position:.2f}s")
        return True
    
    def _auto_arrange_clips(self):
        """Automatically arrange clips to avoid overlap"""
        for track in self.tracks:
            if len(track.clips) > 1:
                # Sort clips by start time
                track.clips.sort(key=lambda clip: clip.start_time)
                
                # Check for overlaps and adjust positions
                for i in range(len(track.clips) - 1):
                    current_clip = track.clips[i]
                    next_clip = track.clips[i + 1]
                    
                    # If clips overlap, move the next one
                    if current_clip.start_time + current_clip.duration > next_clip.start_time:
                        next_clip.start_time = current_clip.start_time + current_clip.duration + 0.1  # Small gap
        
        # Update pan offset limits after arranging clips
        self._update_max_pan_offset()
    
    def render(self, screen, inventory: List[AlienTrack] = None):
        """Render the complete DAW interface as a full-screen overlay"""
        from ..core.debug import debug_info

        
        # Create a full-screen surface for the DAW
        daw_surface = pygame.Surface((self.daw_width, self.daw_height))
        
        # Fill with semi-transparent dark background
        daw_surface.fill((15, 20, 35))  # Darker background for professional look
        
        # Add a subtle gradient effect
        for y in range(self.daw_height):
            alpha = int(255 * (0.8 + 0.2 * (y / self.daw_height)))
            color = (max(0, 15 - alpha//20), max(0, 20 - alpha//20), max(0, 35 - alpha//20))
            pygame.draw.line(daw_surface, color, (0, y), (self.daw_width, y))
        
        # Add main border
        pygame.draw.rect(daw_surface, UI_OUTLINE, 
                        (0, 0, self.daw_width, self.daw_height), 3)
        
        # Add corner accents
        accent_color = (100, 150, 255)
        corner_size = 20
        # Top-left corner
        pygame.draw.rect(daw_surface, accent_color, (0, 0, corner_size, 3))
        pygame.draw.rect(daw_surface, accent_color, (0, 0, 3, corner_size))
        # Top-right corner
        pygame.draw.rect(daw_surface, accent_color, (self.daw_width - corner_size, 0, corner_size, 3))
        pygame.draw.rect(daw_surface, accent_color, (self.daw_width - 3, 0, 3, corner_size))
        # Bottom-left corner
        pygame.draw.rect(daw_surface, accent_color, (0, self.daw_height - 3, corner_size, 3))
        pygame.draw.rect(daw_surface, accent_color, (0, self.daw_height - corner_size, 3, corner_size))
        # Bottom-right corner
        pygame.draw.rect(daw_surface, accent_color, (self.daw_width - corner_size, self.daw_height - 3, corner_size, 3))
        pygame.draw.rect(daw_surface, accent_color, (self.daw_width - 3, self.daw_height - corner_size, 3, corner_size))
        
        # Render title with larger, more prominent styling
        title = self.FONT.render("COSMIC DAW - Digital Audio Workstation", True, HILITE)
        title_rect = title.get_rect(center=(self.daw_width // 2, 30))
        # Add title background
        title_bg = pygame.Surface((title.get_width() + 40, title.get_height() + 20))
        title_bg.fill((25, 35, 60))
        title_bg.set_alpha(200)
        title_bg_rect = title_bg.get_rect(center=(self.daw_width // 2, 30))
        daw_surface.blit(title_bg, title_bg_rect)
        daw_surface.blit(title, title_rect)
        
        # Subtitle highlighting new features
        subtitle = self.SMALL.render("Advanced Audio Editing, Slicing & Merging", True, INFO_COLOR)
        subtitle_rect = subtitle.get_rect(center=(self.daw_width // 2, 55))
        daw_surface.blit(subtitle, subtitle_rect)
        
        # Show inventory info
        if inventory:
            inventory_text = self.SMALL.render(f"Tracks in Inventory: {len(inventory)} | Press I to add tracks!", True, INFO_COLOR)
            daw_surface.blit(inventory_text, (20, 60))
        else:
            no_tracks_text = self.SMALL.render("No tracks in inventory - collect some music first!", True, ERR)
            daw_surface.blit(no_tracks_text, (20, 60))
        
        # Show selection info
        if self.selected_clips:
            if len(self.selected_clips) >= 2:
                selection_text = self.SMALL.render(f"Selected: {len(self.selected_clips)} clips | DELETE to remove, CTRL+C to copy, M to merge", True, HILITE)
            else:
                selection_text = self.SMALL.render(f"Selected: {len(self.selected_clips)} clip | DELETE to remove, CTRL+C to copy", True, HILITE)
            daw_surface.blit(selection_text, (20, 80))
        
        # Render transport panel
        self._render_transport_panel(daw_surface)
        
        # Render track controls
        self._render_track_controls(daw_surface)
        
        # Render timeline
        self._render_timeline(daw_surface)
        
        # Render effects panel
        if self.show_effects_panel:
            self._render_effects_panel(daw_surface)
        
        # Render mixer panel
        if self.show_mixer_panel:
            self._render_mixer_panel(daw_surface)
        
        # Render help text
        self._render_help_text(daw_surface)
        
        # Render track selection menu if open
        if self.show_track_menu:
            self._render_track_menu(daw_surface, inventory)
        
        # Render DAW track selection if open
        if self.show_daw_track_selection:
            self._render_daw_track_selection(daw_surface, inventory)
        
        # Add exit instruction
        exit_text = self.SMALL.render("Press ESC to exit DAW and return to overworld", True, (150, 150, 150))
        daw_surface.blit(exit_text, (self.daw_width - 350, self.daw_height - 30))
        
        # Blit the full DAW surface to screen
        screen.blit(daw_surface, (self.daw_x, self.daw_y))
        

    
    def _render_transport_panel(self, surface):
        """Render the transport controls"""
        y = 100  # Moved down to make room for title
        
        # Transport buttons
        button_width = 80  # Larger buttons for bigger screen
        button_height = 35
        button_spacing = 15
        
        # Play/Stop button
        button_color = HILITE if self.is_playing else (100, 100, 100)
        pygame.draw.rect(surface, button_color, (10, y, button_width, button_height))
        pygame.draw.rect(surface, UI_OUTLINE, (10, y, button_width, button_height), 1)
        
        button_text = "STOP" if self.is_playing else "PLAY"
        text = self.SMALL.render(button_text, True, TEXT_COLOR)
        text_rect = text.get_rect(center=(10 + button_width//2, y + button_height//2))
        surface.blit(text, text_rect)
        
        # Stop button (always visible)
        x = 10 + button_width + button_spacing
        pygame.draw.rect(surface, (255, 100, 100), (x, y, button_width, button_height))
        pygame.draw.rect(surface, UI_OUTLINE, (x, y, button_width, button_height), 1)
        
        text = self.SMALL.render("STOP", True, TEXT_COLOR)
        text_rect = text.get_rect(center=(x + button_width//2, y + button_height//2))
        surface.blit(text, text_rect)
        
        # Record button
        x += button_width + button_spacing
        pygame.draw.rect(surface, ERR, (x, y, button_width, button_height))
        pygame.draw.rect(surface, UI_OUTLINE, (x, y, button_width, button_height), 1)
        
        text = self.SMALL.render("REC", True, TEXT_COLOR)
        text_rect = text.get_rect(center=(x + button_width//2, y + button_height//2))
        surface.blit(text, text_rect)
        
        # Loop button
        x += button_width + button_spacing + 20
        button_color = HILITE if self.is_looping else (100, 100, 100)
        pygame.draw.rect(surface, button_color, (x, y, button_width, button_height))
        pygame.draw.rect(surface, UI_OUTLINE, (x, y, button_width, button_height), 1)
        
        text = self.SMALL.render("LOOP", True, TEXT_COLOR)
        text_rect = text.get_rect(center=(x + button_width//2, y + button_height//2))
        surface.blit(text, text_rect)
        
        # Time display
        x += button_width + button_spacing + 20
        time_text = self.MONO.render(f"Time: {self.playhead_position:.2f}s", True, TEXT_COLOR)
        surface.blit(time_text, (x, y + 5))
        
        # BPM and transport info
        x += 150
        bpm_text = self.SMALL.render(f"BPM: 120 | Speed: {self.playback_speed:.1f}x", True, INFO_COLOR)
        surface.blit(bpm_text, (x, y + 8))
    
    def _render_track_controls(self, surface):
        """Render track control panel"""
        y = 160  # Moved down to make room for transport panel
        
        for i, track in enumerate(self.tracks):
            track_y = y + i * (self.track_height + self.track_spacing)
            
            # Track background
            track_color = track.color if not track.is_muted else (80, 80, 80)
            # Add playing indicator
            if track.is_playing:
                track_color = (min(255, track_color[0] + 50), 
                               min(255, track_color[1] + 50), 
                               min(255, track_color[2] + 50))
            pygame.draw.rect(surface, track_color, (20, track_y, 220, self.track_height))  # Wider track controls
            pygame.draw.rect(surface, UI_OUTLINE, (20, track_y, 220, self.track_height), 1)
            
            # Track name with mute/solo indicators
            name_text = self.SMALL.render(track.name, True, TEXT_COLOR)
            surface.blit(name_text, (25, track_y + 5))
            
            # Show track status with mute/solo info
            if track.has_content():
                clip_names = [clip.name for clip in track.clips]
                playing_indicator = "▶ PLAYING " if track.is_playing else "✓ Has Content: "
                status_text = self.SMALL.render(f"{playing_indicator}{', '.join(clip_names)}", True, (0, 255, 0))
            else:
                status_text = self.SMALL.render("Empty - Press I to add track", True, INFO_COLOR)
            surface.blit(status_text, (25, track_y + 18))
            
            # Show mute/solo status
            if track.is_muted:
                mute_status = self.SMALL.render("🔇 MUTED", True, (255, 100, 100))
                surface.blit(mute_status, (25, track_y + 35))
            elif track.is_soloed:
                solo_status = self.SMALL.render("🎵 SOLO", True, (255, 255, 100))
                surface.blit(solo_status, (25, track_y + 35))
            
            # Track controls
            control_y = track_y + 25
            
            # Mute button
            mute_color = ERR if track.is_muted else (100, 100, 100)
            pygame.draw.rect(surface, mute_color, (25, control_y, 20, 20))
            pygame.draw.rect(surface, UI_OUTLINE, (25, control_y, 20, 20), 1)
            mute_text = self.SMALL.render("M", True, TEXT_COLOR)
            surface.blit(mute_text, (30, control_y + 2))
            
            # Solo button
            solo_color = HILITE if track.is_soloed else (100, 100, 100)
            pygame.draw.rect(surface, solo_color, (50, control_y, 20, 20))
            pygame.draw.rect(surface, UI_OUTLINE, (50, control_y, 20, 20), 1)
            solo_text = self.SMALL.render("S", True, TEXT_COLOR)
            surface.blit(solo_text, (55, control_y + 2))
            
            # Record button
            rec_color = ERR if track.is_recording else (100, 100, 100)
            pygame.draw.rect(surface, rec_color, (75, control_y, 20, 20))
            pygame.draw.rect(surface, UI_OUTLINE, (75, control_y, 20, 20), 1)
            rec_text = self.SMALL.render("R", True, TEXT_COLOR)
            surface.blit(rec_text, (80, control_y + 2))
            
            # Effects button
            pygame.draw.rect(surface, (80, 80, 120), (95, control_y, 20, 20))
            pygame.draw.rect(surface, UI_OUTLINE, (95, control_y, 20, 20), 1)
            fx_text = self.SMALL.render("FX", True, TEXT_COLOR)
            surface.blit(fx_text, (97, control_y + 2))
            
            # Volume slider
            vol_y = control_y + 25
            pygame.draw.rect(surface, (60, 60, 60), (25, vol_y, 100, 8))
            vol_width = int(100 * track.volume)
            pygame.draw.rect(surface, HILITE, (25, vol_y, vol_width, 8))
            pygame.draw.rect(surface, UI_OUTLINE, (25, vol_y, 100, 8), 1)
            
            # Volume text
            vol_text = self.SMALL.render(f"Vol: {track.volume:.1f}", True, TEXT_COLOR)
            surface.blit(vol_text, (25, vol_y + 10))
    
    def _render_timeline(self, surface):
        """Render the main timeline"""
        # Timeline background
        pygame.draw.rect(surface, (20, 25, 40), 
                        (self.timeline_x, self.timeline_y, 
                         self.timeline_width, self.timeline_height))
        pygame.draw.rect(surface, UI_OUTLINE, 
                        (self.timeline_x, self.timeline_y, 
                         self.timeline_width, self.timeline_height), 1)
        
        # Scroll bar above the timeline
        self._render_scroll_bar(surface)
        
        # Grid lines
        self._render_grid(surface)
        
        # Track lanes
        self._render_track_lanes(surface)
        
        # Clips
        self._render_clips(surface)
        
        # Playhead
        self._render_playhead(surface)
        
        # Time markers
        self._render_time_markers(surface)
    
    def _render_grid(self, surface):
        """Render timeline grid"""
        # Calculate visible time range based on panning
        start_time = self.timeline_offset
        end_time = start_time + (self.timeline_width / self.pixels_per_second)
        
        # Vertical time lines
        for i in range(int(start_time), int(end_time) + 1):
            x = self.timeline_x + ((i - start_time) * self.pixels_per_second)
            if self.timeline_x <= x <= self.timeline_x + self.timeline_width:
                color = UI_OUTLINE if i % 5 == 0 else GRID
                pygame.draw.line(surface, color, (x, self.timeline_y), 
                               (x, self.timeline_y + self.timeline_height), 1)
        
        # Horizontal track lines
        for i in range(len(self.tracks) + 1):
            y = self.timeline_y + (i * (self.track_height + self.track_spacing))
            if y <= self.timeline_y + self.timeline_height:
                pygame.draw.line(surface, GRID, (self.timeline_x, y), 
                               (self.timeline_x + self.timeline_width, y), 1)
    
    def _render_track_lanes(self, surface):
        """Render track lanes"""
        for i, track in enumerate(self.tracks):
            y = self.timeline_y + i * (self.track_height + self.track_spacing)
            
            # Track lane background
            lane_color = (30, 35, 50) if i % 2 == 0 else (35, 40, 55)
            pygame.draw.rect(surface, lane_color, 
                           (self.timeline_x, y, self.timeline_width, self.track_height))
    
    def _render_clips(self, surface):
        """Render all clips on the timeline with waveforms"""
        from ..core.debug import debug_info

        # Set clipping rect to prevent rendering outside timeline area
        original_clip = surface.get_clip()
        timeline_clip = pygame.Rect(self.timeline_x, self.timeline_y, self.timeline_width, self.timeline_height)
        surface.set_clip(timeline_clip)
        
        for track in self.tracks:
            for clip in track.clips:
                # Calculate clip position accounting for panning
                adjusted_timeline_x = self.timeline_x - (self.timeline_offset * self.pixels_per_second)
                clip.render(surface, adjusted_timeline_x, self.timeline_y, 
                           self.pixels_per_second, self.track_height, self.track_spacing)
                # Render waveform for the clip
                self._render_clip_waveform(surface, clip, adjusted_timeline_x, self.timeline_y, 
                                         self.pixels_per_second, self.track_height, self.track_spacing)
        
        # Restore original clipping
        surface.set_clip(original_clip)
    
    def _render_clip_waveform(self, surface, clip, timeline_x, timeline_y, pixels_per_second, track_height, track_spacing):
        """Render waveform visualization for a clip"""
        try:
            from ..core.debug import debug_info
            
            # Calculate clip position and dimensions
            clip_x = timeline_x + int(clip.start_time * pixels_per_second)
            track_index = clip.track_index
            clip_y = timeline_y + track_index * (track_height + track_spacing)
            clip_width = int(clip.duration * pixels_per_second)
            

            
            # Skip if clip is too narrow to render waveform
            if clip_width < 10:
                debug_info(f"Clip {clip.name} too narrow ({clip_width}px), skipping waveform")
                return
            
            # Get audio data for waveform generation
            if not hasattr(clip.alien_track, 'array'):
                debug_info(f"Clip {clip.name} has no 'array' attribute")
                # Try to render a fallback waveform for testing
                self._render_fallback_waveform(surface, clip, clip_x, clip_y, clip_width, track_height)
                return
                
            if clip.alien_track.array is None:
                debug_info(f"Clip {clip.name} has None array")
                # Try to render a fallback waveform for testing
                self._render_fallback_waveform(surface, clip, clip_x, clip_y, clip_width, track_height)
                return
            
            audio_data = clip.alien_track.array

            
            # Generate waveform points (with caching)
            waveform_points = self._generate_waveform_points_cached(clip.alien_track, clip_width)
            
            if not waveform_points:
                debug_info(f"Clip {clip.name} generated no waveform points")
                # Try to render a fallback waveform for testing
                self._render_fallback_waveform(surface, clip, clip_x, clip_y, clip_width, track_height)
                return
            

            
            # Draw waveform
            self._draw_waveform(surface, clip, waveform_points, clip_x, clip_y, clip_width, track_height)
            
        except Exception as e:
            from ..core.debug import debug_warning
            debug_warning(f"Error rendering waveform for clip {clip.name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_waveform_points_cached(self, alien_track, target_width):
        """Generate waveform points from audio data with caching"""
        try:
            from ..core.debug import debug_info
            
            # Create cache key based on track and width
            cache_key = f"{alien_track.name}_{target_width}"

            
            # Check if we have cached waveform
            if cache_key in self.waveform_cache:
                
                return self.waveform_cache[cache_key]
            

            
            # Generate new waveform
            waveform_points = self._generate_waveform_points(alien_track.array, target_width)
            
            if waveform_points:
                # Cache the result
                self._add_to_waveform_cache(cache_key, waveform_points)
                debug_info(f"Cached waveform for {cache_key}")
            else:
                debug_info(f"Failed to generate waveform points for {cache_key}")
            
            return waveform_points
            
        except Exception as e:
            from ..core.debug import debug_warning
            debug_warning(f"Error generating cached waveform points: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_waveform_points(self, audio_data, target_width):
        """Generate waveform points from audio data"""
        try:
            from ..core.debug import debug_info
            
            debug_info(f"Generating waveform points: audio shape={audio_data.shape}, target_width={target_width}")
            
            if len(audio_data.shape) == 1:
                # Mono audio
                samples = audio_data
                debug_info(f"Mono audio: {len(samples)} samples")
            elif len(audio_data.shape) == 2:
                # Stereo audio - use average of both channels
                samples = (audio_data[:, 0] + audio_data[:, 1]) / 2
                debug_info(f"Stereo audio: {len(samples)} samples")
            else:
                debug_info(f"Unexpected audio shape: {audio_data.shape}")
                return None
            
            # Calculate how many samples to average per pixel
            samples_per_pixel = max(1, len(samples) // target_width)
            debug_info(f"Samples per pixel: {samples_per_pixel}")
            
            # Generate waveform points with improved scaling
            waveform_points = []
            
            # First pass: calculate statistics for better scaling
            all_rms_values = []
            for i in range(target_width):
                start_sample = i * samples_per_pixel
                end_sample = min(start_sample + samples_per_pixel, len(samples))
                
                if start_sample < len(samples):
                    segment = samples[start_sample:end_sample]
                    rms = np.sqrt(np.mean(segment ** 2))
                    all_rms_values.append(rms)
            
            if not all_rms_values:
                return None
            
            # Calculate statistics for adaptive scaling
            min_rms = min(all_rms_values)
            max_rms = max(all_rms_values)
            mean_rms = np.mean(all_rms_values)
            std_rms = np.std(all_rms_values)
            
            debug_info(f"RMS stats: min={min_rms:.6f}, max={max_rms:.6f}, mean={mean_rms:.6f}, std={std_rms:.6f}")
            
            # Use adaptive scaling based on the actual audio content
            if max_rms > 0:
                # Method 1: Logarithmic scaling with adaptive range
                log_min = np.log10(max(min_rms, 1e-10))
                log_max = np.log10(max_rms)
                log_range = log_max - log_min
                
                # Method 2: Standard deviation based scaling
                std_threshold = mean_rms + 2 * std_rms
                
                for rms in all_rms_values:
                    if rms > 0:
                        # Combine both scaling methods
                        # Log scaling for overall range
                        log_normalized = (np.log10(rms) - log_min) / log_range if log_range > 0 else 0
                        
                        # Standard deviation scaling for detail
                        std_normalized = min(1.0, rms / std_threshold) if std_threshold > 0 else 0
                        
                        # Blend the two approaches (70% log, 30% std for better detail)
                        normalized = 0.7 * log_normalized + 0.3 * std_normalized
                        
                        # Apply additional compression for better visibility
                        normalized = np.power(normalized, 0.7)  # Gamma correction
                        
                        # CRITICAL: Make very quiet sections much smaller
                        # If the audio is very quiet relative to the track's own content, scale it down
                        quiet_scaling = 1.0
                        
                        # Apply different scaling based on the selected mode
                        if hasattr(self, 'waveform_scaling_mode') and self.waveform_scaling_mode == 4:
                            # Mode 4: Quiet audio optimized - very aggressive scaling
                            if rms < (mean_rms * 0.02):  # Less than 2% of mean
                                quiet_scaling = 0.05  # Make it 5% of normal size
                            elif rms < (mean_rms * 0.05):  # Less than 5% of mean
                                quiet_scaling = 0.1  # Make it 10% of normal size
                            elif rms < (mean_rms * 0.1):  # Less than 10% of mean
                                quiet_scaling = 0.2  # Make it 20% of normal size
                            elif rms < (mean_rms * 0.2):  # Less than 20% of mean
                                quiet_scaling = 0.4  # Make it 40% of normal size
                        else:
                            # Standard scaling for other modes
                            if rms < (mean_rms * 0.05):  # Less than 5% of mean
                                quiet_scaling = 0.1  # Make it 10% of normal size
                            elif rms < (mean_rms * 0.1):  # Less than 10% of mean
                                quiet_scaling = 0.25  # Make it 25% of normal size
                            elif rms < (mean_rms * 0.2):  # Less than 20% of mean
                                quiet_scaling = 0.5  # Make it 50% of normal size
                            elif rms < (mean_rms * 0.3):  # Less than 30% of mean
                                quiet_scaling = 0.7  # Make it 70% of normal size
                        
                        # Apply the quiet section scaling
                        normalized *= quiet_scaling
                        
                        # Ensure we're in 0-1 range
                        normalized = max(0, min(1, normalized))
                    else:
                        normalized = 0
                    
                    waveform_points.append(normalized)
            else:
                # All silent - return zeros
                waveform_points = [0] * target_width
            
            debug_info(f"Generated {len(waveform_points)} waveform points")
            return waveform_points
            
        except Exception as e:
            from ..core.debug import debug_warning
            debug_warning(f"Error generating waveform points: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _add_to_waveform_cache(self, cache_key, waveform_points):
        """Add waveform to cache, managing cache size"""
        try:
            # Add to cache
            self.waveform_cache[cache_key] = waveform_points
            
            # Manage cache size
            if len(self.waveform_cache) > self.waveform_cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self.waveform_cache))
                del self.waveform_cache[oldest_key]
                
        except Exception as e:
            from ..core.debug import debug_warning
            debug_warning(f"Error adding to waveform cache: {e}")
    
    def clear_waveform_cache(self):
        """Clear the waveform cache (useful when clips change)"""
        self.waveform_cache.clear()
        from ..core.debug import debug_info
        debug_info("Waveform cache cleared")
    
    def _render_fallback_waveform(self, surface, clip, x, y, width, height):
        """Render a simple fallback waveform for testing when audio data isn't available"""
        try:
            from ..core.debug import debug_info
            
            debug_info(f"Rendering fallback waveform for {clip.name}")
            
            # Calculate waveform area
            waveform_height = height - 20
            waveform_y = y + 10
            
            # Create a simple test waveform (sine wave pattern)
            import math
            test_points = []
            for i in range(width):
                # Create a simple sine wave pattern
                sine_value = math.sin(i * 0.1) * 0.5 + 0.5  # Normalize to 0-1
                test_points.append(sine_value)
            
            # Draw the test waveform
            if len(test_points) > 1:
                points = []
                for i, amplitude in enumerate(test_points):
                    point_x = x + i
                    scaled_amplitude = int(amplitude * waveform_height * 0.3)
                    point_y = waveform_y + (waveform_height // 2) - scaled_amplitude
                    points.append((point_x, point_y))
                
                # Draw in a distinctive color (purple) to show it's a test waveform
                test_color = (200, 100, 255)
                pygame.draw.lines(surface, test_color, False, points, 2)
                
                # Add a label to show it's a test waveform
                try:
                    font = pygame.font.Font(None, 16)
                    label = font.render("TEST", True, test_color)
                    surface.blit(label, (x + 5, waveform_y + 5))
                except:
                    pass
                
                debug_info(f"Rendered fallback waveform for {clip.name}")
            
        except Exception as e:
            from ..core.debug import debug_warning
            debug_warning(f"Error rendering fallback waveform: {e}")
    
    def _draw_waveform(self, surface, clip, waveform_points, x, y, width, height):
        """Draw the waveform on the surface"""
        try:
            # Calculate waveform area (leave some padding)
            waveform_height = height - 20  # Leave space for clip name
            waveform_y = y + 10  # Center vertically with padding
            
            # Determine waveform color based on clip state with better contrast
            if clip in self.selected_clips:
                # Use orange/red for selected - much better contrast than yellow
                waveform_color = (255, 140, 60)  # Orange for selected
                fill_color = (255, 100, 40, 120)  # Semi-transparent orange fill
                border_color = (200, 80, 20)  # Darker orange border
            elif clip.is_playing:
                waveform_color = (60, 255, 100)  # Bright green for playing
                fill_color = (40, 200, 80, 100)  # Semi-transparent green fill
                border_color = (20, 150, 60)  # Darker green border
            else:
                waveform_color = (100, 180, 255)  # Bright blue for normal
                fill_color = (80, 140, 200, 80)  # Semi-transparent blue fill
                border_color = (60, 100, 150)  # Darker blue border
            
            # Draw waveform as filled area with mirror effect
            if len(waveform_points) > 1:
                # Create mirror waveform points
                mirror_points = []
                base_points = []
                
                for i, amplitude in enumerate(waveform_points):
                    point_x = x + i
                    # Enhanced scaling: use more height and apply compression for better visibility
                    enhanced_amplitude = np.power(amplitude, 0.8)  # Enhance quiet parts
                    scaled_amplitude = int(enhanced_amplitude * waveform_height * 0.45)  # Increased to 45% for better visibility
                    
                    # Upper waveform (positive)
                    upper_y = waveform_y + (waveform_height // 2) - scaled_amplitude
                    base_points.append((point_x, upper_y))
                    
                    # Lower waveform (mirror)
                    lower_y = waveform_y + (waveform_height // 2) + scaled_amplitude
                    mirror_points.append((point_x, lower_y))
                
                # Draw the main waveform lines
                pygame.draw.lines(surface, waveform_color, False, base_points, 2)
                pygame.draw.lines(surface, waveform_color, False, mirror_points, 2)
                
                # Fill the area between the lines for a more professional look
                if len(base_points) > 1 and len(mirror_points) > 1:
                    # Create a polygon for filling
                    fill_points = base_points + mirror_points[::-1]  # Reverse mirror points
                    if len(fill_points) > 2:
                        # Draw the filled polygon with semi-transparent color
                        # Convert RGBA to RGB for pygame.draw.polygon
                        fill_rgb = fill_color[:3]  # Remove alpha channel
                        
                        # Create a temporary surface for the fill with proper transparency
                        fill_surface = pygame.Surface((width, waveform_height), pygame.SRCALPHA)
                        fill_surface.fill((0, 0, 0, 0))  # Transparent background
                        
                        # Draw the polygon on the fill surface
                        pygame.draw.polygon(fill_surface, fill_color, fill_points)
                        
                        # Blit the fill surface onto the main surface
                        surface.blit(fill_surface, (x, waveform_y))
                        
                        # Add a subtle inner glow effect by drawing a slightly smaller polygon
                        if len(fill_points) > 2:
                            # Scale down the fill points slightly for inner glow
                            inner_points = []
                            for px, py in fill_points:
                                # Move points slightly toward center
                                center_x = x + width // 2
                                center_y = waveform_y + waveform_height // 2
                                dx = px - center_x
                                dy = py - center_y
                                # Scale down by 0.8 for inner glow
                                inner_x = center_x + dx * 0.8
                                inner_y = center_y + dy * 0.8
                                inner_points.append((inner_x, inner_y))
                            
                            # Draw inner glow with higher transparency
                            inner_color = (*fill_rgb, 40)  # Very transparent
                            pygame.draw.polygon(fill_surface, inner_color, inner_points)
                            surface.blit(fill_surface, (x, waveform_y))
            
            # Draw a subtle background for the waveform area
            waveform_rect = pygame.Rect(x, waveform_y, width, waveform_height)
            pygame.draw.rect(surface, (20, 30, 40), waveform_rect, 1)
            
            # Add a subtle grid overlay for better visual reference
            self._draw_waveform_grid(surface, x, waveform_y, width, waveform_height)
            
            # Draw a subtle center line for reference
            center_y = waveform_y + (waveform_height // 2)
            center_color = (80, 80, 80)  # Subtle gray
            pygame.draw.line(surface, center_color, (x, center_y), (x + width, center_y), 1)
            
        except Exception as e:
            from ..core.debug import debug_warning
            debug_warning(f"Error drawing waveform: {e}")
    
    def _draw_waveform_grid(self, surface, x, y, width, height):
        """Draw a subtle grid overlay on the waveform"""
        try:
            # Draw horizontal grid lines
            grid_color = (30, 40, 50)
            grid_alpha = 30
            
            # Create grid surface with transparency
            grid_surface = pygame.Surface((width, height))
            grid_surface.set_alpha(grid_alpha)
            grid_surface.fill((0, 0, 0))
            
            # Horizontal lines (every 25% of height)
            for i in range(1, 4):
                line_y = y + (height * i // 4)
                pygame.draw.line(grid_surface, grid_color, (0, line_y - y), (width, line_y - y), 1)
            
            # Vertical lines (every 20% of width)
            for i in range(1, 5):
                line_x = (width * i // 5)
                pygame.draw.line(grid_surface, grid_color, (line_x, 0), (line_x, height), 1)
            
            # Blit the grid
            surface.blit(grid_surface, (x, y))
            
        except Exception as e:
            from ..core.debug import debug_warning
            debug_warning(f"Error drawing waveform grid: {e}")
    
    def _render_playhead(self, surface):
        """Render the playhead"""
        # Calculate playhead position relative to current pan offset
        relative_position = self.playhead_position - self.timeline_offset
        x = self.timeline_x + (relative_position * self.pixels_per_second)
        
        # Only show playhead if it's within the visible timeline area
        if self.timeline_x <= x <= self.timeline_x + self.timeline_width:
            pygame.draw.line(surface, PLAYHEAD, (x, self.timeline_y), 
                           (x, self.timeline_y + self.timeline_height), 3)
    
    def _render_time_markers(self, surface):
        """Render time markers"""
        # Calculate visible time range based on panning
        start_time = self.timeline_offset
        end_time = start_time + (self.timeline_width / self.pixels_per_second)
        
        for i in range(0, int(end_time) + 1, 5):
            if i < start_time:
                continue
            x = self.timeline_x + ((i - start_time) * self.pixels_per_second)
            if self.timeline_x <= x <= self.timeline_x + self.timeline_width:
                time_text = self.SMALL.render(f"{i}s", True, INFO_COLOR)
                surface.blit(time_text, (x + 2, self.timeline_y + 2))
    
    def _render_scroll_bar(self, surface):
        """Render the timeline scroll bar above the tracks"""
        # Scroll bar dimensions
        scroll_height = 20
        scroll_y = self.timeline_y - scroll_height - 5
        
        # Calculate total content duration
        total_duration = max(self.max_duration, self._get_content_duration())
        if total_duration <= 0:
            return
        
        # Calculate visible duration
        visible_duration = self.timeline_width / self.pixels_per_second
        
        # Calculate scroll bar thumb position and size
        thumb_width = max(20, (visible_duration / total_duration) * self.timeline_width)
        thumb_x = self.timeline_x + (self.timeline_offset / total_duration) * self.timeline_width
        
        # Draw scroll bar background
        pygame.draw.rect(surface, (40, 45, 60), 
                        (self.timeline_x, scroll_y, self.timeline_width, scroll_height))
        pygame.draw.rect(surface, UI_OUTLINE, 
                        (self.timeline_x, scroll_y, self.timeline_width, scroll_height), 1)
        
        # Draw scroll bar thumb
        pygame.draw.rect(surface, (100, 150, 255), 
                        (thumb_x, scroll_y, thumb_width, scroll_height))
        pygame.draw.rect(surface, (150, 200, 255), 
                        (thumb_x, scroll_y, thumb_width, scroll_height), 1)
        
        # Draw current time indicator on scroll bar
        current_time_x = self.timeline_x + ((self.playhead_position - self.timeline_offset) / total_duration) * self.timeline_width
        if self.timeline_x <= current_time_x <= self.timeline_x + self.timeline_width:
            pygame.draw.line(surface, PLAYHEAD, 
                           (current_time_x, scroll_y), 
                           (current_time_x, scroll_y + scroll_height), 3)
    
    def _render_effects_panel(self, surface):
        """Render the effects panel"""
        # Effects panel implementation
        pass
    
    def _render_mixer_panel(self, surface):
        """Render the mixer panel"""
        # Mixer panel implementation
        pass
    
    def _render_help_text(self, surface):
        """Render helpful DAW instructions with high-contrast colors"""
        y = self.daw_height - 200  # More space from bottom
        
        # Create a semi-transparent black background for the help text area
        help_bg_width = self.daw_width - 40  # Full width minus margins
        help_bg_height = 200
        help_bg_rect = pygame.Rect(20, y - 10, help_bg_width, help_bg_height)
        
        # Draw black background with some transparency
        help_bg_surface = pygame.Surface((help_bg_width, help_bg_height))
        help_bg_surface.fill((0, 0, 0))  # Pure black
        help_bg_surface.set_alpha(180)  # Semi-transparent
        surface.blit(help_bg_surface, help_bg_rect)
        
        # Add a subtle border
        pygame.draw.rect(surface, (50, 50, 50), help_bg_rect, 1)
        
        help_texts = [
            "SPACE: Play/Stop | S: Stop & Reset | R: Record | L: Loop | +/-: Zoom",
            "I: Show Track Selection Menu | ESC: Exit DAW",
            "Track Addition: I → Select Track → Choose DAW Track → ENTER",
            "CTRL+Z: Undo | CTRL+Y: Redo | DELETE: Delete Selected",
            "CTRL+C: Copy | CTRL+V: Paste | CTRL+X: Cut | CTRL+D: Duplicate",
            "CTRL+S: Slice at Playhead | M: Merge Selected Clips | CTRL+R: Remove | CTRL+I: Insert at Playhead",
            "CTRL+SHIFT+S: Save Mix | CTRL+SHIFT+E: Export to WAV | CTRL+SHIFT+L: Load Mix",
            "Click: Select Clips | Drag: Move Clips | CTRL+Click: Multi-select",
            "Mouse Wheel: Zoom | SHIFT+Mouse Wheel: Pan Left/Right | LEFT/RIGHT: Pan by 1/4 visible time",
            "F1/F2: DEBUG - Test slice functionality | F3: Mute | F4: Solo | F5: Test Mute/Solo | F6: Clear Waveform Cache | F7: Test Waveform System | F8: Create Test Clip | F9: Toggle Waveform Scaling",
            "🎵 NEW: Mix saving, loading, and WAV export now available!",
            "🎵 NEW: Timeline panning and extended duration (20 minutes)!",
            "🎵 NEW: Scroll bar above timeline for precise navigation!"
        ]
        
        # Use bright blue text for high contrast
        help_color = (100, 200, 255)  # Bright blue
        
        for i, text in enumerate(help_texts):
            help_surface = self.SMALL.render(text, True, help_color)
            surface.blit(help_surface, (25, y + i * 25))  # Slightly indented from background edge
    
    def _render_track_menu(self, surface, inventory):
        """Render the track selection menu"""
        if not inventory:
            return
        
        # Menu background - larger for bigger screen
        menu_width = 600
        menu_height = 400
        menu_x = (self.daw_width - menu_width) // 2
        menu_y = (self.daw_height - menu_height) // 2
        
        # Semi-transparent background
        menu_surface = pygame.Surface((menu_width, menu_height))
        menu_surface.set_alpha(230)
        menu_surface.fill((20, 25, 40))
        pygame.draw.rect(menu_surface, UI_OUTLINE, (0, 0, menu_width, menu_height), 2)
        surface.blit(menu_surface, (menu_x, menu_y))
        
        # Menu title
        title = self.FONT.render("SELECT TRACK TO ADD", True, HILITE)
        title_rect = title.get_rect(center=(menu_x + menu_width//2, menu_y + 20))
        surface.blit(title, title_rect)
        
        # Track list
        y_offset = 60
        for i, track in enumerate(inventory):
            # Track background
            track_color = HILITE if i == self.selected_inventory_index else (60, 65, 80)
            track_rect = (menu_x + 10, menu_y + y_offset + i * 40, menu_width - 20, 35)
            pygame.draw.rect(surface, track_color, track_rect)
            pygame.draw.rect(surface, UI_OUTLINE, track_rect, 1)
            
            # Track name
            name_text = self.SMALL.render(track.name, True, TEXT_COLOR)
            surface.blit(name_text, (menu_x + 20, menu_y + y_offset + i * 40 + 5))
            
            # Track info
            info_text = self.SMALL.render(
                f"Duration: {track.duration:.1f}s | Best match: {track.get_best_species_match()[0]}", 
                True, INFO_COLOR
            )
            surface.blit(info_text, (menu_x + 20, menu_y + y_offset + i * 40 + 20))
        
        # Instructions
        instructions = [
            "↑↓: Navigate | ENTER: Select | ESC: Cancel",
            f"Selected: {inventory[self.selected_inventory_index].name if inventory else 'None'}"
        ]
        
        for i, instruction in enumerate(instructions):
            inst_text = self.SMALL.render(instruction, True, INFO_COLOR)
            surface.blit(inst_text, (menu_x + 10, menu_y + menu_height - 40 + i * 20))
    
    def _render_daw_track_selection(self, surface, inventory):
        """Render the DAW track selection menu"""
        if not inventory:
            return
        
        # Menu background - larger for bigger screen
        menu_width = 600
        menu_height = 400
        menu_x = (self.daw_width - menu_width) // 2
        menu_y = (self.daw_height - menu_height) // 2
        
        # Semi-transparent background
        menu_surface = pygame.Surface((menu_width, menu_height))
        menu_surface.set_alpha(230)
        menu_surface.fill((20, 25, 40))
        pygame.draw.rect(menu_surface, UI_OUTLINE, (0, 0, menu_width, menu_height), 2)
        surface.blit(menu_surface, (menu_x, menu_y))
        
        # Menu title
        title = self.FONT.render("CHOOSE DAW TRACK", True, HILITE)
        title_rect = title.get_rect(center=(menu_x + menu_width//2, menu_y + 20))
        surface.blit(title, title_rect)
        
        # Show selected inventory track
        if inventory:
            selected_track = inventory[self.selected_inventory_index]
            track_info = self.SMALL.render(
                f"Adding: {selected_track.name} ({selected_track.duration:.1f}s)", 
                True, INFO_COLOR
            )
            surface.blit(track_info, (menu_x + 20, menu_y + 40))
        
        # Track list
        y_offset = 80  # Moved down to make room for track info
        for i, track in enumerate(self.tracks):
            # Track background
            track_color = HILITE if i == self.selected_daw_track_index else (60, 65, 80)
            track_rect = (menu_x + 10, menu_y + y_offset + i * 40, menu_width - 20, 35)
            pygame.draw.rect(surface, track_color, track_rect)
            pygame.draw.rect(surface, UI_OUTLINE, track_rect, 1)
            
            # Track name
            name_text = self.SMALL.render(track.name, True, TEXT_COLOR)
            surface.blit(name_text, (menu_x + 20, menu_y + y_offset + i * 40 + 5))
            
            # Track info
            if track.has_content():
                clip_names = [clip.name for clip in track.clips]
                status = f"Has: {', '.join(clip_names)}"
                info_color = (255, 100, 100)  # Red for full tracks
            else:
                status = "Empty - Available"
                info_color = (0, 255, 0)  # Green for empty tracks
            
            info_text = self.SMALL.render(status, True, info_color)
            surface.blit(info_text, (menu_x + 20, menu_y + y_offset + i * 40 + 20))
        
        # Instructions
        instructions = [
            "↑↓: Navigate | ENTER: Select | ESC: Cancel",
            f"Selected DAW Track: {self.tracks[self.selected_daw_track_index].name if self.tracks else 'None'}"
        ]
        
        for i, instruction in enumerate(instructions):
            inst_text = self.SMALL.render(instruction, True, INFO_COLOR)
            surface.blit(inst_text, (menu_x + 10, menu_y + menu_height - 40 + i * 20))
    
    def _select_clip_at_position(self, track_index: int, time_position: float):
        """Select clip at given position"""
        if 0 <= track_index < len(self.tracks):
            track = self.tracks[track_index]
            clicked_clip = None
            
            for clip in track.clips:
                if (clip.start_time <= time_position <= 
                    clip.start_time + clip.duration):
                    clicked_clip = clip
                    break
            
            if clicked_clip:
                # Handle selection with modifier keys
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    # Add/remove from selection
                    if clicked_clip in self.selected_clips:
                        self.selected_clips.remove(clicked_clip)
                        clicked_clip.is_selected = False
                        print(f"🎵 Deselected clip: {clicked_clip.name}")
                    else:
                        self.selected_clips.append(clicked_clip)
                        clicked_clip.is_selected = True
                        print(f"🎵 Added to selection: {clicked_clip.name}")
                elif pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    # Add to selection
                    if clicked_clip not in self.selected_clips:
                        self.selected_clips.append(clicked_clip)
                        clicked_clip.is_selected = True
                        print(f"🎵 Added to selection: {clicked_clip.name}")
                else:
                    # Single selection
                    # Clear previous selection
                    for clip in self.selected_clips:
                        clip.is_selected = False
                    self.selected_clips.clear()
                    
                    # Select new clip
                    self.selected_clips = [clicked_clip]
                    clicked_clip.is_selected = True
                    print(f"🎵 Selected clip: {clicked_clip.name}")
            else:
                # Clicked on empty space - clear selection
                for clip in self.selected_clips:
                    clip.is_selected = False
                self.selected_clips.clear()
                print("🎵 Cleared clip selection")
    
    def _show_context_menu(self, track_index: int, time_position: float):
        """Show context menu for timeline"""
        # Context menu implementation
        pass

    def _update_solo_states(self):
        """Update solo states - mute all tracks except soloed ones"""
        has_soloed = any(track.is_soloed for track in self.tracks)
        
        for track in self.tracks:
            if has_soloed:
                # If any track is soloed, mute non-soloed tracks
                should_mute = not track.is_soloed
                if should_mute != track.is_muted:
                    track.is_muted = should_mute
                    if should_mute and track.is_playing:
                        track._stop_audio_playback()
                        track._audio_started = False
                    elif not should_mute and track.is_playing and track.current_clip:
                        track._start_audio_playback()
                        track._audio_started = True
            else:
                # No tracks are soloed, unmute all tracks
                if track.is_muted:
                    track.is_muted = False
                    if track.is_playing and track.current_clip:
                        track._start_audio_playback()
                        track._audio_started = True

    def remove_time_range_from_clip(self, clip: 'DAWClip', start_time: float, end_time: float) -> bool:
        """Remove a time range from a clip, creating two new clips around the removed section"""
        if not (clip.start_time < start_time < end_time < clip.start_time + clip.duration):
            print(f"❌ Invalid time range for clip {clip.name}")
            return False
        
        # Calculate the two remaining sections
        first_start = clip.start_time
        first_end = start_time
        second_start = end_time
        second_end = clip.start_time + clip.duration
        
        first_duration = first_end - first_start
        second_duration = second_end - second_start
        
        # Check minimum durations
        if first_duration < 0.1 and second_duration < 0.1:
            print(f"❌ Both remaining sections would be too short")
            return False
        
        print(f"🎵 Removing time range {start_time:.2f}s to {end_time:.2f}s from {clip.name}")
        
        # Create clips for the remaining sections
        new_clips = []
        
        if first_duration >= 0.1:
            first_clip = self._create_sliced_clip(
                clip, first_start, first_duration, f"{clip.name}_before"
            )
            if first_clip:
                new_clips.append(first_clip)
                print(f"✅ Created first section: {first_duration:.2f}s")
        
        if second_duration >= 0.1:
            second_clip = self._create_sliced_clip(
                clip, second_start, second_duration, f"{clip.name}_after"
            )
            if second_clip:
                new_clips.append(second_clip)
                print(f"✅ Created second section: {second_duration:.2f}s")
        
        if not new_clips:
            print("❌ Failed to create any remaining sections")
            return False
        
        # Remove original clip and add new ones
        track = self.tracks[clip.track_index]
        track.remove_clip(clip)
        
        for new_clip in new_clips:
            track.add_clip(new_clip)
        
        # Update selection
        if clip in self.selected_clips:
            self.selected_clips.remove(clip)
            self.selected_clips.extend(new_clips)
        
        print(f"✅ Successfully removed time range, created {len(new_clips)} new clips")
        return True


# Duplicate DAWTrack class removed - using the first one at line 118
    
# Constructor removed - using the first DAWTrack class
    
# Methods removed - using the first DAWTrack class
    
# Toggle methods removed - using the first DAWTrack class
    
# start_playback method removed - using the first DAWTrack class
    
    def stop_playback(self):
        """Stop track playback"""
        debug_enter("stop_playback", "cosmic_daw.py")
        self.is_playing = False
        
        if self.current_clip:
            debug_info(f"Track {self.name}: Stopping playback of {self.current_clip.name}")
            # Stop the audio
            self._stop_audio_playback()
        else:
            debug_info(f"Track {self.name}: No current clip to stop")
        
        self.current_clip = None
        self.playback_position = 0.0  # Reset playback position
        self._audio_started = False  # Reset audio state
        
        # Clear all audio references
        self.current_channel = None
        self.current_sliced_sound = None
        
        debug_exit("stop_playback", "cosmic_daw.py")
    
    def set_playhead_position(self, new_position: float):
        """Set the playhead to a new position and update audio accordingly"""
        debug_enter("set_playhead_position", "cosmic_daw.py")
        print(f"🎵 Track {self.name}: Setting playhead to {new_position:.2f}s")
        
        if self.is_playing:
            # Stop current audio
            if self.current_clip:
                self._stop_audio_playback()
                self._audio_started = False
            
            # Update position and find new clip
            self.playback_position = new_position
            self._find_current_clip(new_position)
            
            # Start new audio if we have a clip
            if self.current_clip:
                self._start_audio_playback()
                self._audio_started = True
        else:
            # Just update position for when we start playing
            self.playback_position = new_position
            self._find_current_clip(new_position)
        debug_exit("set_playhead_position", "cosmic_daw.py")
    
    def _start_audio_playback(self):
        """Actually start playing the audio"""
        debug_enter("_start_audio_playback", "cosmic_daw.py")
        
        if self.current_clip and self.current_clip.alien_track.sound:
            try:
                # Calculate offset into the clip's audio data
                clip_start_time = self.current_clip.start_time
                timeline_position = self.playback_position
                
                # The offset is how far into the clip we are
                offset_seconds = timeline_position - clip_start_time
                
                # Ensure offset is within the clip's bounds
                if offset_seconds < 0:
                    offset_seconds = 0
                elif offset_seconds >= self.current_clip.duration:
                    offset_seconds = self.current_clip.duration - 0.1  # Slight margin
                
                if offset_seconds > 0:
                    # We need to start from the middle of the clip
                    # Create a sliced sound object starting from the offset
                    try:
                        # Use the track's make_slice_sound method to create a sound starting from offset
                        sliced_sound = self.current_clip.alien_track.make_slice_sound(
                            offset_seconds, 
                            self.current_clip.duration
                        )
                        debug_info(f"Created sliced sound from {offset_seconds:.2f}s to {self.current_clip.duration:.2f}s")
                        
                        # Store the sliced sound reference for stopping later
                        self.current_sliced_sound = sliced_sound
                        
                        # Play the sliced sound (it will start from the beginning of the slice)
                        channel = sliced_sound.play()
                        if channel:
                            # Store the channel reference for stopping later
                            self.current_channel = channel
                            debug_info(f"Audio started: {self.current_clip.name} from offset {offset_seconds:.2f}s (timeline: {timeline_position:.2f}s)")
                        else:
                            debug_error(f"Failed to get audio channel for sliced {self.current_clip.name}")
                    except Exception as e:
                        debug_error(f"Error creating sliced sound: {e}")
                        # Fallback to original method
                        channel = self.current_clip.alien_track.sound.play()
                        if channel:
                            self.current_channel = channel
                            channel.set_pos(int(offset_seconds * 1000))
                            debug_info(f"Audio started (fallback): {self.current_clip.name} at offset {offset_seconds:.2f}s")
                else:
                    # Start from beginning of clip
                    channel = self.current_clip.alien_track.sound.play()
                    if channel:
                        self.current_channel = channel
                        debug_info(f"Audio started: {self.current_clip.name} from beginning (timeline: {timeline_position:.2f}s)")
                    else:
                        debug_error(f"Failed to get audio channel for {self.current_clip.name}")
                        
            except Exception as e:
                debug_error(f"Error playing audio: {e}")
                import traceback
                traceback.print_exc()
        else:
            debug_warning("Cannot play audio - missing clip or sound")
            if not self.current_clip:
                debug_warning("No current_clip")
            elif not hasattr(self.current_clip, 'alien_track'):
                debug_warning("No alien_track on clip")
            elif not self.current_clip.alien_track.sound:
                debug_warning("No sound on alien_track")
        debug_exit("_start_audio_playback", "cosmic_daw.py")
    
    def _stop_audio_playback(self):
        """Stop the currently playing audio"""
        debug_enter("_stop_audio_playback", "cosmic_daw.py")
        
        try:
            # Stop the current channel if we have one
            if self.current_channel:
                self.current_channel.stop()
                debug_info(f"Stopped audio channel for {self.current_clip.name if self.current_clip else 'track'}")
                self.current_channel = None
            
            # Stop the original sound as well (in case it's still playing)
            if self.current_clip and self.current_clip.alien_track.sound:
                self.current_clip.alien_track.sound.stop()
                debug_info(f"Stopped original sound for {self.current_clip.name}")
            
            # Clear the sliced sound reference
            if self.current_sliced_sound:
                # Try to stop the sliced sound if it has a stop method
                try:
                    if hasattr(self.current_sliced_sound, 'stop'):
                        self.current_sliced_sound.stop()
                        debug_info(f"Stopped sliced sound for {self.current_clip.name if self.current_clip else 'track'}")
                except Exception as e:
                    debug_warning(f"Could not stop sliced sound: {e}")
                self.current_sliced_sound = None
            
            debug_info(f"Audio stopped: {self.current_clip.name if self.current_clip else 'track'}")
            
            # Note: Removed global pygame.mixer.stop() to prevent stopping other tracks
                
        except Exception as e:
            debug_error(f"Error stopping audio: {e}")
            import traceback
            traceback.print_exc()
        
        debug_exit("_stop_audio_playback", "cosmic_daw.py")
    
    def update_playback(self, dt: float, current_time: float):
        """Update track playback"""
        if not self.is_playing:
            return
        
        # Update our playback position to match the DAW's timeline
        self.playback_position = current_time
        
        # Find which clip should be playing at this time
        old_clip = self.current_clip
        self._find_current_clip(current_time)
        
        # If we have a new clip or the same clip but audio hasn't started
        if self.current_clip:
            if old_clip != self.current_clip or not hasattr(self, '_audio_started') or not self._audio_started:
                # New clip or need to start audio
                print(f"🎵 Track {self.name}: Playing {self.current_clip.name} at timeline position {current_time:.2f}s")
                self._start_audio_playback()
                self._audio_started = True
            else:
                # Same clip, just log progress every second
                clip_time = current_time - self.current_clip.start_time
                if int(clip_time) % 1 == 0 and clip_time > 0:
                    print(f"🎵 Track {self.name}: Playing {self.current_clip.name} at {clip_time:.1f}s")
        else:
            # No clip at this time - stop audio if we were playing something
            if old_clip and hasattr(self, '_audio_started') and self._audio_started:
                print(f"🎵 Track {self.name}: No clip at time {current_time:.2f}s, stopping audio")
                self._stop_audio_playback()
                self._audio_started = False
    
    def _find_current_clip(self, time: float):
        """Find which clip should be playing at given time"""
        old_clip = self.current_clip
        self.current_clip = None
        
        for clip in self.clips:
            if (clip.start_time <= time <= 
                clip.start_time + clip.duration):
                self.current_clip = clip
                break
        
        # If we switched to a different clip, handle the transition
        if self.current_clip != old_clip:
            if old_clip:
                # Stop the old clip
                self._stop_audio_playback()
                self._audio_started = False
            if self.current_clip and self.is_playing:
                # Start the new clip
                self._start_audio_playback()
                self._audio_started = True

    # ========== MIX MANAGEMENT METHODS ==========
    
    def save_current_mix(self):
        """Save the current mix to a .mix file"""
        debug_enter("save_current_mix", "cosmic_daw.py")
        
        try:
            # Prompt user for mix name
            mix_name = self._prompt_mix_name()
            if not mix_name:
                debug_info("User cancelled save operation")
                return
            
            # Add debug info
            debug_info(f"Attempting to save mix with name: {mix_name}")
            debug_info(f"Number of tracks: {len(self.tracks)}")
            debug_info(f"MixManager instance: {self.mix_manager}")
            
            # Save the mix using MixManager
            if self.mix_manager.save_mix(self, mix_name):
                debug_info(f"Mix saved successfully: {mix_name}")
                # Show success message to user
                self._show_message(f"Mix saved: {mix_name}")
            else:
                debug_error("Failed to save mix")
                self._show_message("Failed to save mix", is_error=True)
                
        except Exception as e:
            debug_error(f"Error saving mix: {e}")
            import traceback
            traceback.print_exc()
            self._show_message("Error saving mix", is_error=True)
        
        debug_exit("save_current_mix", "cosmic_daw.py")
    
    def export_current_mix(self):
        """Export the current mix to a WAV file"""
        debug_enter("export_current_mix", "cosmic_daw.py")
        
        try:
            # Prompt user for mix name
            mix_name = self._prompt_mix_name()
            if not mix_name:
                debug_info("User cancelled export operation")
                return
            
            # Export the mix using MixManager
            if self.mix_manager.export_mix_to_wav(self, mix_name):
                debug_info(f"Mix exported successfully: {mix_name}")
                # Show success message to user
                self._show_message(f"Mix exported: {mix_name}.wav")
            else:
                debug_error("Failed to export mix")
                self._show_message("Failed to export mix", is_error=True)
                
        except Exception as e:
            debug_error(f"Error exporting mix: {e}")
            self._show_message("Error exporting mix", is_error=True)
        
        debug_exit("export_current_mix", "cosmic_daw.py")
    
    def load_saved_mix(self):
        """Load a saved mix from a .mix file"""
        debug_enter("load_saved_mix", "cosmic_daw.py")
        
        try:
            # Get list of saved mixes
            saved_mixes = self.mix_manager.list_saved_mixes()
            
            if not saved_mixes:
                debug_info("No saved mixes found")
                self._show_message("No saved mixes found")
                return
            
            # Show selection dialog
            selected_mix = self._show_selection_dialog("Select mix to load:", saved_mixes)
            
            if selected_mix:
                # Load the mix - pass the current inventory
                inventory = getattr(self, 'current_inventory', [])
                debug_info(f"Loading mix '{selected_mix}' with inventory: {len(inventory) if inventory else 0} tracks")
                if self.mix_manager.load_mix(self, selected_mix, inventory):
                    debug_info(f"Mix loaded successfully: {selected_mix}")
                    self._show_message(f"Mix loaded: {selected_mix}")
                    
                    # Refresh DAW state after loading
                    self._refresh_daw_after_load()
                else:
                    debug_error("Failed to load mix")
                    self._show_message("Failed to load mix", is_error=True)
            else:
                debug_info("Load cancelled by user")
                
        except Exception as e:
            debug_error(f"Error loading mix: {e}")
            self._show_message("Error loading mix", is_error=True)
        
        debug_exit("load_saved_mix", "cosmic_daw.py")
    
    def _show_message(self, message: str, is_error: bool = False):
        """Show a temporary message to the user"""
        # For now, just print to console - in the future this could show a UI overlay
        if is_error:
            print(f"❌ {message}")
        else:
            print(f"✅ {message}")
    
    def _prompt_mix_name(self) -> str:
        """Prompt user for a mix name using an in-game text input dialog"""
        return self._show_text_input_dialog("Enter mix name:", "Mix")
    
    def _show_text_input_dialog(self, prompt: str, default_text: str = "") -> str:
        """Show an in-game text input dialog"""
        # Create a simple text input dialog
        dialog_width = 400
        dialog_height = 150
        dialog_x = (self.daw_width - dialog_width) // 2
        dialog_y = (self.daw_height - dialog_height) // 2
        
        # Initialize text input
        text_input = default_text
        cursor_pos = len(text_input)
        cursor_visible = True
        cursor_timer = 0
        
        # Colors
        dialog_bg = (40, 40, 40)
        text_color = (255, 255, 255)
        cursor_color = (255, 255, 255)
        border_color = (100, 100, 100)
        
        # Font
        font = pygame.font.Font(None, 32)
        small_font = pygame.font.Font(None, 24)
        
        # Input loop
        input_active = True
        while input_active:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        input_active = False
                        break
                    elif event.key == pygame.K_ESCAPE:
                        return ""  # Cancel
                    elif event.key == pygame.K_BACKSPACE:
                        if cursor_pos > 0:
                            text_input = text_input[:cursor_pos-1] + text_input[cursor_pos:]
                            cursor_pos -= 1
                    elif event.key == pygame.K_DELETE:
                        if cursor_pos < len(text_input):
                            text_input = text_input[:cursor_pos] + text_input[cursor_pos+1:]
                    elif event.key == pygame.K_LEFT:
                        cursor_pos = max(0, cursor_pos - 1)
                    elif event.key == pygame.K_RIGHT:
                        cursor_pos = min(len(text_input), cursor_pos + 1)
                    elif event.key == pygame.K_HOME:
                        cursor_pos = 0
                    elif event.key == pygame.K_END:
                        cursor_pos = len(text_input)
                    elif event.unicode and len(text_input) < 50:  # Limit to 50 characters
                        text_input = text_input[:cursor_pos] + event.unicode + text_input[cursor_pos:]
                        cursor_pos += 1
            
            # Update cursor blink
            cursor_timer += 1
            if cursor_timer > 30:  # Blink every 30 frames
                cursor_visible = not cursor_visible
                cursor_timer = 0
            
            # Draw dialog
            screen = pygame.display.get_surface()
            
            # Draw background overlay
            overlay = pygame.Surface((self.daw_width, self.daw_height))
            overlay.set_alpha(128)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            
            # Draw dialog box
            dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
            pygame.draw.rect(screen, dialog_bg, dialog_rect)
            pygame.draw.rect(screen, border_color, dialog_rect, 2)
            
            # Draw prompt
            prompt_surface = font.render(prompt, True, text_color)
            prompt_rect = prompt_surface.get_rect(midtop=(dialog_x + dialog_width//2, dialog_y + 20))
            screen.blit(prompt_surface, prompt_rect)
            
            # Draw text input area
            input_rect = pygame.Rect(dialog_x + 20, dialog_y + 60, dialog_width - 40, 30)
            pygame.draw.rect(screen, (20, 20, 20), input_rect)
            pygame.draw.rect(screen, border_color, input_rect, 1)
            
            # Draw text
            text_surface = font.render(text_input, True, text_color)
            screen.blit(text_surface, (input_rect.x + 5, input_rect.y + 5))
            
            # Draw cursor
            if cursor_visible:
                cursor_x = input_rect.x + 5 + font.size(text_input[:cursor_pos])[0]
                pygame.draw.line(screen, cursor_color, 
                               (cursor_x, input_rect.y + 5), 
                               (cursor_x, input_rect.y + input_rect.height - 5), 2)
            
            # Draw instructions
            instructions = small_font.render("Press ENTER to confirm, ESC to cancel", True, (150, 150, 150))
            instructions_rect = instructions.get_rect(midtop=(dialog_x + dialog_width//2, dialog_y + 110))
            screen.blit(instructions, instructions_rect)
            
            pygame.display.flip()
            pygame.time.wait(16)  # ~60 FPS
        
        # Sanitize filename
        import re
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', text_input.strip())
        return sanitized[:50]  # Limit length
    
    def _show_selection_dialog(self, prompt: str, options: List[str]) -> str:
        """Show an in-game selection dialog"""
        if not options:
            return ""
        
        # Create a simple selection dialog
        dialog_width = 500
        dialog_height = 200 + min(len(options) * 30, 300)  # Dynamic height based on options
        dialog_x = (self.daw_width - dialog_width) // 2
        dialog_y = (self.daw_height - dialog_height) // 2
        
        # Colors
        dialog_bg = (40, 40, 40)
        text_color = (255, 255, 255)
        selected_color = (100, 150, 255)
        border_color = (100, 100, 100)
        hover_color = (60, 60, 60)
        
        # Font
        font = pygame.font.Font(None, 32)
        small_font = pygame.font.Font(None, 24)
        
        # Selection state
        selected_index = 0
        hover_index = -1
        
        # Input loop
        input_active = True
        while input_active:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        input_active = False
                        break
                    elif event.key == pygame.K_ESCAPE:
                        return ""  # Cancel
                    elif event.key == pygame.K_UP:
                        selected_index = max(0, selected_index - 1)
                    elif event.key == pygame.K_DOWN:
                        selected_index = min(len(options) - 1, selected_index + 1)
                    elif event.key == pygame.K_HOME:
                        selected_index = 0
                    elif event.key == pygame.K_END:
                        selected_index = len(options) - 1
                elif event.type == pygame.MOUSEMOTION:
                    # Check if mouse is hovering over an option
                    mouse_x, mouse_y = event.pos
                    hover_index = -1
                    for i, option in enumerate(options):
                        option_rect = pygame.Rect(dialog_x + 20, dialog_y + 60 + i * 30, dialog_width - 40, 25)
                        if option_rect.collidepoint(mouse_x, mouse_y):
                            hover_index = i
                            break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mouse_x, mouse_y = event.pos
                        for i, option in enumerate(options):
                            option_rect = pygame.Rect(dialog_x + 20, dialog_y + 60 + i * 30, dialog_width - 40, 25)
                            if option_rect.collidepoint(mouse_x, mouse_y):
                                selected_index = i
                                input_active = False
                                break
            
            # Draw dialog
            screen = pygame.display.get_surface()
            
            # Draw background overlay
            overlay = pygame.Surface((self.daw_width, self.daw_height))
            overlay.set_alpha(128)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            
            # Draw dialog box
            dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
            pygame.draw.rect(screen, dialog_bg, dialog_rect)
            pygame.draw.rect(screen, border_color, dialog_rect, 2)
            
            # Draw prompt
            prompt_surface = font.render(prompt, True, text_color)
            prompt_rect = prompt_surface.get_rect(midtop=(dialog_x + dialog_width//2, dialog_y + 20))
            screen.blit(prompt_surface, prompt_rect)
            
            # Draw options
            for i, option in enumerate(options):
                option_rect = pygame.Rect(dialog_x + 20, dialog_y + 60 + i * 30, dialog_width - 40, 25)
                
                # Determine background color
                if i == selected_index:
                    bg_color = selected_color
                elif i == hover_index:
                    bg_color = hover_color
                else:
                    bg_color = (20, 20, 20)
                
                pygame.draw.rect(screen, bg_color, option_rect)
                pygame.draw.rect(screen, border_color, option_rect, 1)
                
                # Draw option text
                option_surface = small_font.render(option, True, text_color)
                screen.blit(option_surface, (option_rect.x + 5, option_rect.y + 5))
            
            # Draw instructions
            instructions = small_font.render("Use UP/DOWN arrows or mouse to select, ENTER to confirm, ESC to cancel", True, (150, 150, 150))
            instructions_rect = instructions.get_rect(midtop=(dialog_x + dialog_width//2, dialog_y + dialog_height - 30))
            screen.blit(instructions, instructions_rect)
            
            pygame.display.flip()
            pygame.time.wait(16)  # ~60 FPS
        
        return options[selected_index] if selected_index < len(options) else ""
    
    def _refresh_daw_after_load(self):
        """Refresh DAW state after loading a mix"""
        debug_enter("_refresh_daw_after_load", "cosmic_daw.py")
        
        try:
            # Stop any current audio playback
            self._stop_audio_playback()
            self._audio_started = False
            
            # Reset playhead to beginning
            self.playhead_position = 0.0
            
            # Clear any selections
            self.selected_clips.clear()
            self.selected_tracks.clear()
            
            # Update max duration based on loaded clips
            max_duration = 0
            for track in self.tracks:
                for clip in track.clips:
                    end_time = clip.start_time + clip.duration
                    max_duration = max(max_duration, end_time)
            
            if max_duration > 0:
                self.max_duration = max_duration + 10  # Add some padding
            
            # Clear waveform cache to force regeneration
            self.clear_waveform_cache()
            
            # Verify the loaded state
            self._verify_loaded_state()
            
            debug_info(f"DAW refreshed after load: {len(self.tracks)} tracks, max_duration={self.max_duration}s")
            
        except Exception as e:
            debug_error(f"Error refreshing DAW after load: {e}")
        
        debug_exit("_refresh_daw_after_load", "cosmic_daw.py")
    
    def _verify_loaded_state(self):
        """Verify that the loaded state is valid"""
        debug_enter("_verify_loaded_state", "cosmic_daw.py")
        
        try:
            debug_info(f"Verifying loaded state: {len(self.tracks)} tracks")
            
            for i, track in enumerate(self.tracks):
                debug_info(f"Track {i}: {track.name}, {len(track.clips)} clips")
                
                for j, clip in enumerate(track.clips):
                    debug_info(f"  Clip {j}: {clip.name}, alien_track={clip.alien_track.name if clip.alien_track else 'None'}, duration={clip.duration}s")
                    
                    # Check if clip has audio data
                    if clip.alien_track and hasattr(clip.alien_track, 'array'):
                        audio_shape = clip.alien_track.array.shape if clip.alien_track.array is not None else 'None'
                        debug_info(f"    Audio data: {audio_shape}")
                    else:
                        debug_warn(f"    No audio data available for clip {clip.name}")
            
            # Check if any tracks have clips with audio
            tracks_with_audio = 0
            for track in self.tracks:
                for clip in track.clips:
                    if clip.alien_track and hasattr(clip.alien_track, 'array') and clip.alien_track.array is not None:
                        tracks_with_audio += 1
                        break
            
            debug_info(f"Tracks with audio: {tracks_with_audio}/{len(self.tracks)}")
            
        except Exception as e:
            debug_error(f"Error verifying loaded state: {e}")
        
        debug_exit("_verify_loaded_state", "cosmic_daw.py")


class DAWAction:
    """Base class for undoable DAW actions"""
    
    def __init__(self, daw: CosmicDAW):
        self.daw = daw
        self.timestamp = pygame.time.get_ticks()
    
    def execute(self):
        """Execute the action"""
        pass
    
    def undo(self):
        """Undo the action"""
        pass
    
    def redo(self):
        """Redo the action"""
        pass


class AddClipAction(DAWAction):
    """Action for adding a clip to the DAW"""
    
    def __init__(self, daw: CosmicDAW, clip: DAWClip, track_index: int):
        super().__init__(daw)
        self.clip = clip
        self.track_index = track_index
    
    def execute(self):
        """Add the clip"""
        self.daw.tracks[self.track_index].add_clip(self.clip)
    
    def undo(self):
        """Remove the clip"""
        self.daw.tracks[self.track_index].remove_clip(self.clip)
    
    def redo(self):
        """Re-add the clip"""
        self.execute()


class DeleteClipAction(DAWAction):
    """Action for deleting a clip from the DAW"""
    
    def __init__(self, daw: CosmicDAW, clip: DAWClip, track_index: int):
        super().__init__(daw)
        self.clip = clip
        self.track_index = track_index
    
    def execute(self):
        """Delete the clip"""
        self.daw.tracks[self.track_index].remove_clip(self.clip)
    
    def undo(self):
        """Restore the clip"""
        self.daw.tracks[self.track_index].add_clip(self.clip)
    
    def redo(self):
        """Re-delete the clip"""
        self.execute()


class CutClipsAction(DAWAction):
    """Action for cutting clips to clipboard"""
    
    def __init__(self, daw: CosmicDAW, clips: List[DAWClip]):
        super().__init__(daw)
        self.clips = clips
        # Store the clips' original positions for undo
        self.clip_data = []
        for clip in clips:
            self.clip_data.append({
                'clip': clip,
                'track_index': clip.track_index,
                'start_time': clip.start_time
            })
    
    def execute(self):
        """Cut the clips (already done in the calling method)"""
        pass  # Cutting is handled by the calling method
    
    def undo(self):
        """Restore the clips to their original positions"""
        for data in self.clip_data:
            clip = data['clip']
            track_index = data['track_index']
            start_time = data['start_time']
            
            # Restore clip properties
            clip.track_index = track_index
            clip.start_time = start_time
            
            # Add back to the track
            if track_index < len(self.daw.tracks):
                self.daw.tracks[track_index].add_clip(clip)
    
    def redo(self):
        """Re-cut the clips"""
        # Remove clips again
        for data in self.clip_data:
            clip = data['clip']
            track_index = data['track_index']
            if track_index < len(self.daw.tracks):
                if clip in self.daw.tracks[track_index].clips:
                    self.daw.tracks[track_index].remove_clip(clip)


class SliceClipsAction(DAWAction):
    """Action for slicing clips at a specific time"""
    
    def __init__(self, daw: CosmicDAW, clips: List[DAWClip], slice_time: float):
        super().__init__(daw)
        self.clips = clips
        self.slice_time = slice_time
        # Store the clips' original state for undo
        self.original_clips = []
        for clip in clips:
            self.original_clips.append({
                'clip': clip,
                'track_index': clip.track_index,
                'start_time': clip.start_time,
                'duration': clip.duration,
                'name': clip.name
            })
        # Store the created slices for undo
        self.created_slices = []
    
    def execute(self):
        """Slice the clips (already done in the calling method)"""
        pass  # Slicing is handled by the calling method
    
    def undo(self):
        """Restore the original clips by removing slices and recreating originals"""
        # Remove all slices that were created during this operation
        if hasattr(self, 'created_slices') and self.created_slices:
            for slice_data in self.created_slices:
                clip = slice_data['clip']
                track_index = slice_data['track_index']
                if track_index < len(self.daw.tracks):
                    if clip in self.daw.tracks[track_index].clips:
                        self.daw.tracks[track_index].remove_clip(clip)
                        debug_info(f"Removed slice: {clip.name}")
        
        # For now, we can't fully recreate the original clips because we don't have the original audio data
        # But we can at least remove the slices, which is better than nothing
        debug_info(f"Undo: Removed {len(self.created_slices) if hasattr(self, 'created_slices') else 0} slices")
        debug_warning("Undo for slicing: Slices removed but original clips not fully restored (audio data reconstruction needed)")
    
    def redo(self):
        """Re-slice the clips"""
        # This would also be complex
        debug_warning("Redo for slicing is not fully implemented yet")
    
    def capture_created_slices(self):
        """Capture the created slices for proper undo"""
        # Find all clips that were created during slicing
        self.created_slices = []
        
        for track in self.daw.tracks:
            for clip in track.clips:
                # Check if this clip was created during our slicing operation
                # Look for clips with names that indicate they were sliced
                if any(original['name'] in clip.name for original in self.original_clips):
                    self.created_slices.append({
                        'clip': clip,
                        'track_index': clip.track_index,
                        'start_time': clip.start_time
                    })
        
        debug_info(f"Captured {len(self.created_slices)} created slices for undo")


class PasteClipsAction(DAWAction):
    """Action for pasting clips from clipboard"""
    
    def __init__(self, daw: CosmicDAW, clipboard_clips: List[DAWClip], paste_time: float):
        super().__init__(daw)
        self.clipboard_clips = clipboard_clips
        self.paste_time = paste_time
        # Store the pasted clips for undo
        self.pasted_clips = []
    
    def execute(self):
        """Paste the clips (already done in the calling method)"""
        pass  # Pasting is handled by the calling method
    
    def undo(self):
        """Remove the pasted clips"""
        # This would need to track what was actually pasted
        # For now, just log that undo isn't fully implemented for pasting
        debug_warning("Undo for pasting is not fully implemented yet")
    
    def redo(self):
        """Re-paste the clips"""
        # This would also need to track what was pasted
        debug_warning("Redo for pasting is not fully implemented yet")


class DuplicateClipsAction(DAWAction):
    """Action for duplicating clips"""
    
    def __init__(self, daw: CosmicDAW, original_clips: List[DAWClip]):
        super().__init__(daw)
        self.original_clips = original_clips
        # Store the duplicated clips for undo
        self.duplicated_clips = []
    
    def execute(self):
        """Duplicate the clips (already done in the calling method)"""
        pass  # Duplication is handled by the calling method
    
    def undo(self):
        """Remove the duplicated clips"""
        # This would need to track what was actually duplicated
        # For now, just log that undo isn't fully implemented for duplication
        debug_warning("Undo for duplication is not fully implemented yet")
    
    def redo(self):
        """Re-duplicate the clips"""
        # This would also need to track what was duplicated
        debug_warning("Redo for duplication is not fully implemented yet")


class MergeClipsAction(DAWAction):
    """Action for merging multiple clips into one"""
    
    def __init__(self, daw: CosmicDAW, original_clips: List[DAWClip]):
        super().__init__(daw)
        self.original_clips = original_clips
        # Store the original clips' state for undo
        self.original_clips_state = []
        for clip in original_clips:
            self.original_clips_state.append({
                'clip': clip,
                'track_index': clip.track_index,
                'start_time': clip.start_time,
                'duration': clip.duration,
                'name': clip.name
            })
        # Store the merged clip for undo
        self.merged_clip = None
    
    def execute(self):
        """Merge the clips (already done in the calling method)"""
        pass  # Merging is handled by the calling method
    
    def undo(self):
        """Restore the original clips by removing the merged clip and recreating originals"""
        if not self.merged_clip:
            debug_warning("No merged clip to undo")
            return
        
        # Remove the merged clip
        track = self.daw.tracks[self.merged_clip.track_index]
        if self.merged_clip in track.clips:
            track.remove_clip(self.merged_clip)
        
        # Recreate the original clips
        for clip_state in self.original_clips_state:
            # Create a new DAWClip with the original properties
            original_clip = DAWClip(
                alien_track=clip_state['clip'].alien_track,
                start_time=clip_state['start_time'],
                duration=clip_state['duration'],
                name=clip_state['clip'].name
            )
            original_clip.track_index = clip_state['track_index']
            
            # Add back to the track
            if clip_state['track_index'] < len(self.daw.tracks):
                self.daw.tracks[clip_state['track_index']].add_clip(original_clip)
        
        # Auto-arrange to ensure no overlap
        self.daw._auto_arrange_clips()
        
        debug_info(f"Undo: Restored {len(self.original_clips_state)} original clips")
    
    def redo(self):
        """Re-merge the clips"""
        # This would need to re-apply the merge operation
        debug_warning("Redo for merging is not fully implemented yet")


class MoveClipAction(DAWAction):
    """Action for moving a clip to a new position"""
    
    def __init__(self, daw: CosmicDAW, clip: DAWClip, 
                 old_track_index: int, old_start_time: float,
                 new_track_index: int, new_start_time: float):
        super().__init__(daw)
        self.clip = clip
        self.old_track_index = old_track_index
        self.old_start_time = old_start_time
        self.new_track_index = new_track_index
        self.new_start_time = new_start_time
        
        # Store the original state of ALL clips on both tracks to handle auto-arrange bumps
        self.old_track_clips_state = []
        self.new_track_clips_state = []
        
        # Store old track state (excluding the moved clip)
        if old_track_index < len(daw.tracks):
            old_track = daw.tracks[old_track_index]
            for other_clip in old_track.clips:
                if other_clip != clip:  # Don't store the clip we're moving
                    self.old_track_clips_state.append({
                        'clip': other_clip,
                        'start_time': other_clip.start_time
                    })
        
        # Store new track state (excluding the moved clip)
        if new_track_index < len(daw.tracks):
            new_track = daw.tracks[new_track_index]
            for other_clip in new_track.clips:
                if other_clip != clip:  # Don't store the clip we're moving
                    self.new_track_clips_state.append({
                        'clip': other_clip,
                        'start_time': other_clip.start_time
                    })
    
    def capture_final_state(self):
        """Capture the final state after the move and auto-arrange for proper undo"""
        # This method should be called after the move and auto-arrange are complete
        # Store the final positions of all clips on both tracks
        self.final_old_track_clips_state = []
        self.final_new_track_clips_state = []
        
        # Capture final old track state
        if self.old_track_index < len(self.daw.tracks):
            old_track = self.daw.tracks[self.old_track_index]
            for other_clip in old_track.clips:
                if other_clip != self.clip:  # Don't store the clip we moved
                    self.final_old_track_clips_state.append({
                        'clip': other_clip,
                        'start_time': other_clip.start_time
                    })
        
        # Capture final new track state
        if self.new_track_index < len(self.daw.tracks):
            new_track = self.daw.tracks[self.new_track_index]
            for other_clip in new_track.clips:
                if other_clip != self.clip:  # Don't store the clip we moved
                    self.final_new_track_clips_state.append({
                        'clip': other_clip,
                        'start_time': other_clip.start_time
                    })
    
    def execute(self):
        """Move the clip (already done in the calling method)"""
        pass  # Moving is handled by the calling method
    
    def undo(self):
        """Restore the clip and all bumped clips to their original positions"""
        # First, restore the moved clip to its original position
        if self.clip in self.daw.tracks[self.new_track_index].clips:
            self.daw.tracks[self.new_track_index].remove_clip(self.clip)
        
        self.clip.track_index = self.old_track_index
        self.clip.start_time = self.old_start_time
        self.daw.tracks[self.old_track_index].add_clip(self.clip)
        
        # Now restore all the bumped clips on both tracks to their original positions
        # Restore old track clips to their original positions (before the move)
        if self.old_track_index < len(self.daw.tracks):
            old_track = self.daw.tracks[self.old_track_index]
            for clip_state in self.old_track_clips_state:
                clip_state['clip'].start_time = clip_state['start_time']
        
        # Restore new track clips to their original positions (before the move)
        if self.new_track_index < len(self.daw.tracks):
            new_track = self.daw.tracks[self.new_track_index]
            for clip_state in self.new_track_clips_state:
                clip_state['clip'].start_time = clip_state['start_time']
        
        print(f"🎵 Undo: Moved {self.clip.name} back to {self.daw.tracks[self.old_track_index].name} at {self.old_start_time:.2f}s")
        print(f"🎵 Undo: Restored {len(self.old_track_clips_state) + len(self.new_track_clips_state)} bumped clips to original positions")
    
    def redo(self):
        """Re-move the clip to the new position"""
        # Remove from current position
        if self.clip in self.daw.tracks[self.old_track_index].clips:
            self.daw.tracks[self.old_track_index].remove_clip(self.clip)
        
        # Move to new position
        self.clip.track_index = self.new_track_index
        self.clip.start_time = self.new_start_time
        self.daw.tracks[self.new_track_index].add_clip(self.clip)
        
        # Re-apply auto-arrange to bump other clips
        self.daw._auto_arrange_clips()
        
        print(f"🎵 Redo: Moved {self.clip.name} to {self.daw.tracks[self.new_track_index].name} at {self.new_start_time:.2f}s")


# ========== ACTION CLASSES ==========

class DeleteClipsAction(DAWAction):
    """Action for deleting multiple clips from the DAW"""
    
    def __init__(self, daw: CosmicDAW, clips: List[DAWClip]):
        super().__init__(daw)
        self.clips = clips
        # Store the clips' original positions for undo
        self.clip_data = []
        for clip in clips:
            self.clip_data.append({
                'clip': clip,
                'track_index': clip.track_index,
                'start_time': clip.start_time
            })
    
    def execute(self):
        """Delete the clips"""
        for clip in self.clips:
            if clip in self.daw.tracks[clip.track_index].clips:
                self.daw.tracks[clip.track_index].remove_clip(clip)
    
    def undo(self):
        """Restore the clips to their original positions"""
        for data in self.clip_data:
            clip = data['clip']
            track_index = data['track_index']
            start_time = data['start_time']
            
            # Restore clip properties
            clip.track_index = track_index
            clip.start_time = start_time
            
            # Add back to the track
            if track_index < len(self.daw.tracks):
                self.daw.tracks[track_index].add_clip(clip)
    
    def redo(self):
        """Re-delete the clips"""
        self.execute()
