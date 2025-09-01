import os
import json
import numpy as np
import pygame
from typing import List, Dict, Optional, Tuple
from ..core.debug import debug_enter, debug_exit, debug_info, debug_warning, debug_error
# Forward references to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .cosmic_daw import CosmicDAW, DAWClip, DAWTrack

class MixManager:
    """Manages saving, loading, and exporting DAW mixes"""
    
    def __init__(self, game_state):
        self.game_state = game_state
        self.mixes_dir = "mixes"  # Directory to store mix files
        self.ensure_mixes_directory()
    
    def ensure_mixes_directory(self):
        """Create the mixes directory if it doesn't exist"""
        try:
            if not os.path.exists(self.mixes_dir):
                os.makedirs(self.mixes_dir)
                debug_info(f"Created mixes directory: {self.mixes_dir}")
        except Exception as e:
            debug_error(f"Failed to create mixes directory: {e}")
    
    def save_mix(self, daw: 'CosmicDAW', mix_name: str) -> bool:
        """
        Save the current DAW state to a .mix file
        
        Args:
            daw: The CosmicDAW instance to save
            mix_name: Name for the mix (user will be prompted if None)
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        debug_enter("save_mix", "mix_manager.py", mix_name=mix_name)
        
        try:
            # If no name provided, prompt user
            if not mix_name:
                mix_name = self._prompt_mix_name()
                if not mix_name:
                    debug_warning("No mix name provided, save cancelled")
                    return False
            
            # Add debug info
            debug_info(f"Creating mix data for DAW with {len(daw.tracks)} tracks")
            debug_info(f"Mixes directory: {self.mixes_dir}")
            debug_info(f"Current working directory: {os.getcwd()}")
            
            # Create mix data structure
            mix_data = self._create_mix_data(daw, mix_name)
            debug_info(f"Mix data created successfully: {len(mix_data.get('tracks', []))} tracks")
            
            # Generate filename
            filename = self._sanitize_filename(mix_name)
            filepath = os.path.join(self.mixes_dir, f"{filename}.mix")
            debug_info(f"Target filepath: {filepath}")
            
            # Ensure directory exists
            os.makedirs(self.mixes_dir, exist_ok=True)
            debug_info(f"Directory ensured: {self.mixes_dir}")
            
            # Test file creation
            test_filepath = os.path.join(self.mixes_dir, "test.txt")
            try:
                with open(test_filepath, 'w') as f:
                    f.write("Test file")
                debug_info(f"Test file created successfully: {test_filepath}")
                # Clean up test file
                os.remove(test_filepath)
                debug_info("Test file cleaned up")
            except Exception as test_e:
                debug_error(f"Test file creation failed: {test_e}")
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(mix_data, f, indent=2)
            
            debug_info(f"Mix saved successfully: {filepath}")
            debug_exit("save_mix", "mix_manager.py")
            return True
            
        except Exception as e:
            debug_error(f"Failed to save mix: {e}")
            import traceback
            traceback.print_exc()
            debug_exit("save_mix", "mix_manager.py")
            return False
    
    def load_mix(self, daw: 'CosmicDAW', mix_name: str, inventory: List['AlienTrack'] = None) -> bool:
        """
        Load a saved mix into the DAW
        
        Args:
            daw: The CosmicDAW instance to load into
            mix_name: Name of the mix to load
            inventory: Current inventory of alien tracks to match against
        
        Returns:
            bool: True if load was successful, False otherwise
        """
        debug_enter("load_mix", "mix_manager.py", mix_name=mix_name)
        
        try:
            # Generate filename
            filename = self._sanitize_filename(mix_name)
            filepath = os.path.join(self.mixes_dir, f"{filename}.mix")
            
            if not os.path.exists(filepath):
                debug_error(f"Mix file not found: {filepath}")
                return False
            
            # Load mix data
            with open(filepath, 'r') as f:
                mix_data = json.load(f)
            
            # Restore DAW state
            self._restore_daw_state(daw, mix_data, inventory)
            
            debug_info(f"Mix loaded successfully: {filepath}")
            debug_exit("load_mix", "mix_manager.py")
            return True
            
        except Exception as e:
            debug_error(f"Failed to load mix: {e}")
            debug_exit("load_mix", "mix_manager.py")
            return False
    
    def export_mix_to_wav(self, daw: 'CosmicDAW', mix_name: str, output_path: str = None) -> bool:
        """
        Export the current mix to a WAV file
        
        Args:
            daw: The CosmicDAW instance to export
            mix_name: Name for the exported file
            output_path: Optional custom output path
        
        Returns:
            bool: True if export was successful, False otherwise
        """
        debug_enter("export_mix_to_wav", "mix_manager.py", mix_name=mix_name, output_path=output_path)
        
        try:
            # Generate output path
            if not output_path:
                filename = self._sanitize_filename(mix_name)
                output_path = os.path.join(self.mixes_dir, f"{filename}.wav")
            
            # Render the mix to audio
            audio_data = self._render_mix_to_audio(daw)
            
            # Save as WAV file
            self._save_wav_file(audio_data, output_path)
            
            debug_info(f"Mix exported successfully: {output_path}")
            debug_exit("export_mix_to_wav", "mix_manager.py")
            return True
            
        except Exception as e:
            debug_error(f"Failed to export mix: {e}")
            debug_exit("export_mix_to_wav", "mix_manager.py")
            return False
    
    def list_saved_mixes(self) -> List[str]:
        """Get a list of all saved mix names"""
        try:
            if not os.path.exists(self.mixes_dir):
                return []
            
            mix_files = [f for f in os.listdir(self.mixes_dir) if f.endswith('.mix')]
            mix_names = [os.path.splitext(f)[0] for f in mix_files]
            return mix_names
        except Exception as e:
            debug_error(f"Failed to list saved mixes: {e}")
            return []
    
    def _create_mix_data(self, daw: 'CosmicDAW', mix_name: str) -> Dict:
        """Create a data structure representing the current DAW state"""
        debug_info(f"Creating mix data for {mix_name}")
        debug_info(f"DAW has {len(daw.tracks)} tracks")
        
        mix_data = {
            "name": mix_name,
            "created": pygame.time.get_ticks(),
            "version": "1.0",
            "tracks": [],
            "playhead_position": daw.playhead_position,
            "pixels_per_second": daw.pixels_per_second,
            "max_duration": daw.max_duration
        }
        
        debug_info(f"Basic mix data created: playhead={daw.playhead_position}, pixels_per_second={daw.pixels_per_second}")
        
        # Save track data
        for i, track in enumerate(daw.tracks):
            debug_info(f"Processing track {i}: {track.name}")
            track_data = {
                "index": track.track_index,
                "muted": track.is_muted,
                "soloed": track.is_soloed,
                "volume": track.volume,
                "pan": track.pan,
                "clips": []
            }
            
            debug_info(f"Track {i} has {len(track.clips)} clips")
            
            # Save clip data
            for j, clip in enumerate(track.clips):
                debug_info(f"Processing clip {j}: {clip.name}")
                clip_data = {
                    "name": clip.name,
                    "start_time": clip.start_time,
                    "duration": clip.duration,
                    "track_index": clip.track_index,
                    "alien_track_name": clip.alien_track.name if clip.alien_track else None
                }
                track_data["clips"].append(clip_data)
                debug_info(f"Clip {j} data: {clip_data}")
            
            mix_data["tracks"].append(track_data)
        
        debug_info(f"Final mix data has {len(mix_data['tracks'])} tracks")
        return mix_data
    
    def _restore_daw_state(self, daw: 'CosmicDAW', mix_data: Dict, inventory: List['AlienTrack'] = None):
        """Restore DAW state from saved mix data"""
        # Clear current state
        daw.tracks.clear()
        daw.selected_clips.clear()
        daw.selected_tracks.clear()
        
        # Restore basic settings
        daw.playhead_position = mix_data.get("playhead_position", 0.0)
        daw.pixels_per_second = mix_data.get("pixels_per_second", 50)
        daw.max_duration = mix_data.get("max_duration", 1200.0)  # Default to 20 minutes
        
        # Restore tracks
        tracks_to_restore = mix_data.get("tracks", [])
        debug_info(f"Restoring {len(tracks_to_restore)} tracks")
        debug_info(f"Track data: {tracks_to_restore}")
        
        # Define track names and colors (matching the DAW's default setup)
        track_names = [f"Track {i + 1}" for i in range(8)]  # 8 tracks max
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
        
        for i, track_data in enumerate(tracks_to_restore):
            debug_info(f"Processing track {i}: {track_data}")
            
            # Import here to avoid circular import issues
            from .cosmic_daw import DAWTrack
            
            track_index = track_data["index"]
            track_name = track_names[track_index] if track_index < len(track_names) else f"Track {track_index + 1}"
            track_color = track_colors[track_index] if track_index < len(track_colors) else (150, 150, 150)
            
            debug_info(f"Creating track: name='{track_name}', color={track_color}, index={track_index}")
            
            track = None
            try:
                track = DAWTrack(
                    name=track_name,
                    color=track_color,
                    track_index=track_index,
                    daw=daw
                )
                track.is_muted = track_data.get("muted", False)
                track.is_soloed = track_data.get("soloed", False)
                track.volume = track_data.get("volume", 1.0)
                track.pan = track_data.get("pan", 0.0)
                
                debug_info(f"Restored track {track.name} (index {track.track_index})")
            except Exception as e:
                debug_error(f"Failed to create track {track_name} at index {track_index}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Only proceed if track was created successfully
            if track is None:
                continue
                
            # Restore clips with proper alien_track references
            for clip_data in track_data.get("clips", []):
                # Try to find the alien_track in current inventory
                alien_track = self._find_alien_track_by_name(daw, clip_data.get("alien_track_name"), inventory)
                
                if alien_track:
                    # Import here to avoid circular import issues
                    from .cosmic_daw import DAWClip
                    clip = DAWClip(
                        alien_track=alien_track,
                        start_time=clip_data["start_time"],
                        track_index=clip_data["track_index"],
                        daw=daw,
                        duration=clip_data["duration"],
                        name=clip_data["name"]
                    )
                    track.clips.append(clip)
                    debug_info(f"Restored clip '{clip.name}' with alien_track '{alien_track.name}'")
                else:
                    debug_warning(f"Could not find alien_track '{clip_data.get('alien_track_name')}' in inventory - clip '{clip_data['name']}' will not play")
                    # Create a placeholder clip for visual purposes
                    from .cosmic_daw import DAWClip
                    clip = DAWClip(
                        alien_track=None,
                        start_time=clip_data["start_time"],
                        track_index=clip_data["track_index"],
                        daw=daw,
                        duration=clip_data["duration"],
                        name=clip_data["name"]
                    )
                    track.clips.append(clip)
            
            # Add the successfully created track to the DAW
            daw.tracks.append(track)
        
        debug_info(f"DAW state restored: {len(daw.tracks)} tracks, playhead at {daw.playhead_position}s")
    
    def _render_mix_to_audio(self, daw: 'CosmicDAW') -> np.ndarray:
        """Render the current mix to audio data"""
        # This is a simplified renderer - in a full implementation,
        # you'd want to properly mix all tracks with effects
        
        # Calculate total duration
        max_duration = 0
        for track in daw.tracks:
            for clip in track.clips:
                end_time = clip.start_time + clip.duration
                max_duration = max(max_duration, end_time)
        
        if max_duration == 0:
            max_duration = 10.0  # Default 10 seconds if no clips
        
        # Create output array (stereo, 44.1kHz)
        sample_rate = 44100
        total_samples = int(max_duration * sample_rate)
        output = np.zeros((total_samples, 2), dtype=np.float32)
        
        # Mix all tracks
        for track in daw.tracks:
            if track.is_muted:
                continue
                
            for clip in track.clips:
                if clip.alien_track and hasattr(clip.alien_track, 'array'):
                    # Get clip audio data
                    start_sample = int(clip.start_time * sample_rate)
                    end_sample = int((clip.start_time + clip.duration) * sample_rate)
                    
                    # Ensure we don't go out of bounds
                    start_sample = max(0, start_sample)
                    end_sample = min(total_samples, end_sample)
                    
                    if start_sample < end_sample:
                        # Get the clip's audio data
                        clip_audio = clip.alien_track.array
                        if clip_audio is not None and len(clip_audio) > 0:
                            # Resample if necessary and apply track volume
                            clip_samples = end_sample - start_sample
                            if len(clip_audio) >= clip_samples:
                                # Apply track volume and pan
                                track_audio = clip_audio[:clip_samples].astype(np.float32) / 32768.0
                                track_audio *= track.volume
                                
                                # Simple panning (left/right balance)
                                if track.pan < 0:  # Left
                                    track_audio[:, 1] *= (1.0 + track.pan)
                                elif track.pan > 0:  # Right
                                    track_audio[:, 0] *= (1.0 - track.pan)
                                
                                # Add to output
                                output[start_sample:end_sample] += track_audio
        
        # Normalize output
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.8  # Leave some headroom
        
        return output
    
    def _save_wav_file(self, audio_data: np.ndarray, output_path: str):
        """Save audio data as a WAV file"""
        import wave
        
        # Convert to 16-bit PCM
        audio_16bit = (audio_data * 32767).astype(np.int16)
        
        # Save as WAV
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(44100)  # 44.1kHz
            wav_file.writeframes(audio_16bit.tobytes())
    
    def _prompt_mix_name(self) -> str:
        """Prompt user for mix name - simplified for now"""
        # In a full implementation, this would show a text input dialog
        # For now, return a default name
        return f"Mix_{pygame.time.get_ticks()}"
    
    def _sanitize_filename(self, filename: str) -> str:
        """Convert mix name to safe filename"""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 50:
            filename = filename[:50]
        
        return filename.strip()
    
    def _find_alien_track_by_name(self, daw: 'CosmicDAW', alien_track_name: str, inventory: List['AlienTrack'] = None):
        """Find an alien track in the current inventory by name"""
        if not alien_track_name:
            return None
        
        # Use the passed inventory if available
        if inventory:
            for track in inventory:
                if track.name == alien_track_name:
                    return track
        
        # If we can't find it, return None
        debug_warning(f"Could not find alien track '{alien_track_name}' in current inventory")
        return None

