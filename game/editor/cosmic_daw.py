"""
Refactored CosmicDAW - Main DAW class using extracted modules
This is the new, cleaner version that replaces the monolithic cosmic_daw.py
"""

import pygame
import sys
from typing import List, Optional, Tuple
from ..core.debug import debug_info, debug_error, debug_warning
from ..core.config import WIDTH, HEIGHT, TRACK_COLORS
from .daw_clip import DAWClip
from .daw_track import DAWTrack
from .daw_ui import DAWUI
from .daw_audio import DAWAudio
from .daw_timeline import DAWTimeline
from .daw_clip_operations import DAWClipOperations
from .mix_manager import MixManager


class CosmicDAW:
    """Main DAW class - now much cleaner and more maintainable"""
    
    def __init__(self, fonts, game_state):
        """Initialize the refactored DAW"""
        self.fonts = fonts
        self.game_state = game_state
        
        # Core dimensions
        self.daw_width = WIDTH
        self.daw_height = HEIGHT
        self.max_tracks = 8
        self.track_height = 80
        self.track_spacing = 10
        self.max_duration = 1200.0  # 20 minutes
        
        # Initialize component modules
        self.ui = DAWUI(self)
        self.audio = DAWAudio(self)
        self.timeline = DAWTimeline(self)
        self.clip_ops = DAWClipOperations(self)
        
        # Initialize tracks
        self.tracks = []
        self._initialize_tracks()
        
        # Initialize mix manager
        self.mix_manager = MixManager(game_state)
        
        # UI state
        self.text_input_active = False
        self.text_input_prompt = ""
        self.text_input_text = ""
        self.text_input_title = ""
        
        self.selection_dialog_active = False
        self.selection_dialog_title = ""
        self.selection_dialog_options = []
        self.selection_dialog_selected = 0
        
        # Mouse state
        self.mouse_pos = (0, 0)
        self.mouse_buttons = (False, False, False)
        self.dragging = False
        
        # DAW state
        self.should_close = False
        
        # Debug info
        debug_info(f"🎵 Refactored CosmicDAW initialized with {len(self.tracks)} tracks")
    
    def _initialize_tracks(self):
        """Initialize the track list"""
        self.tracks = []
        
        for i in range(self.max_tracks):
            track_name = f"Track {i + 1}"
            track_color = TRACK_COLORS[i % len(TRACK_COLORS)]
            track = DAWTrack(track_name, track_color, i, self)
            self.tracks.append(track)
    
    # Property accessors for backward compatibility
    @property
    def is_playing(self):
        return self.audio.is_playing
    
    @property
    def playhead_position(self):
        return self.audio.playhead_position
    
    @playhead_position.setter
    def playhead_position(self, value):
        self.audio.playhead_position = value
    
    @property
    def pixels_per_second(self):
        return self.timeline.pixels_per_second
    
    @property
    def timeline_offset(self):
        return self.timeline.timeline_offset
    
    @property
    def selected_clips(self):
        return self.clip_ops.selected_clips
    
    @property
    def clipboard_clips(self):
        return self.clip_ops.clipboard_clips
    
    # Timeline properties
    @property
    def timeline_x(self):
        return self.timeline.timeline_x
    
    @property
    def timeline_y(self):
        return self.timeline.timeline_y
    
    @property
    def timeline_width(self):
        return self.timeline.timeline_width
    
    @property
    def timeline_height(self):
        return self.timeline.timeline_height
    
    # Audio control methods
    def toggle_playback(self):
        """Toggle playback on/off"""
        self.audio.toggle_playback()
    
    def stop(self):
        """Stop playback"""
        self.audio.stop_playback()
    
    def seek_to_position(self, position: float):
        """Seek to a specific time position"""
        self.audio.seek_to_position(position)
    
    # Timeline control methods
    def zoom_in(self):
        """Zoom in on the timeline"""
        self.timeline.zoom_in()
    
    def zoom_out(self):
        """Zoom out on the timeline"""
        self.timeline.zoom_out()
    
    def pan_left(self, amount: float):
        """Pan the timeline left"""
        self.timeline.pan_left(amount)
    
    def pan_right(self, amount: float):
        """Pan the timeline right"""
        self.timeline.pan_right(amount)
    
    def pan_left_by_visible_amount(self):
        """Pan left by one quarter of visible time"""
        self.timeline.pan_left_by_visible_amount()
    
    def pan_right_by_visible_amount(self):
        """Pan right by one quarter of visible time"""
        self.timeline.pan_right_by_visible_amount()
    
    # Clip operation methods
    def add_track_from_inventory(self, alien_track, track_index: int) -> bool:
        """Add a track from inventory"""
        return self.clip_ops.add_track_from_inventory(alien_track, track_index)
    
    def duplicate_selected_clips(self):
        """Duplicate selected clips"""
        self.clip_ops.duplicate_selected_clips()
    
    def delete_selected_clips(self):
        """Delete selected clips"""
        self.clip_ops.delete_selected_clips()
    
    def merge_selected_clips(self):
        """Merge selected clips"""
        self.clip_ops.merge_selected_clips()
    
    def slice_clips_at_playhead(self):
        """Slice clips at playhead"""
        self.clip_ops.slice_clips_at_playhead()
    
    def copy_selected_clips(self):
        """Copy selected clips to clipboard"""
        self.clip_ops.copy_selected_clips()
    
    def paste_clips_from_clipboard(self, paste_time: float, target_track: int = None):
        """Paste clips from clipboard"""
        self.clip_ops.paste_clips_from_clipboard(paste_time, target_track)
    
    # Undo/Redo methods
    def undo(self):
        """Undo last action"""
        self.clip_ops.undo()
    
    def redo(self):
        """Redo last undone action"""
        self.clip_ops.redo()
    
    # Mix management methods
    def save_current_mix(self):
        """Save the current mix"""
        try:
            mix_name = self._prompt_mix_name()
            if mix_name:
                success = self.mix_manager.save_mix(mix_name, self)
                if success:
                    debug_info(f"✅ Mix '{mix_name}' saved successfully")
                else:
                    debug_error(f"❌ Failed to save mix '{mix_name}'")
            else:
                debug_info("Mix save cancelled")
        except Exception as e:
            debug_error(f"❌ Error saving mix: {e}")
    
    def export_current_mix(self):
        """Export the current mix as WAV"""
        try:
            mix_name = self._prompt_mix_name()
            if mix_name:
                success = self.mix_manager.export_mix(mix_name, self)
                if success:
                    debug_info(f"✅ Mix '{mix_name}' exported successfully")
                else:
                    debug_error(f"❌ Failed to export mix '{mix_name}'")
            else:
                debug_info("Mix export cancelled")
        except Exception as e:
            debug_error(f"❌ Error exporting mix: {e}")
    
    def load_saved_mix(self):
        """Load a saved mix"""
        try:
            saved_mixes = self.mix_manager.list_saved_mixes()
            if not saved_mixes:
                debug_info("No saved mixes found")
                return
            
            # Show selection dialog
            self._show_selection_dialog("Select mix to load:", saved_mixes)
            
        except Exception as e:
            debug_error(f"❌ Error loading mix: {e}")
    
    def _prompt_mix_name(self) -> Optional[str]:
        """Prompt user for mix name"""
        self.text_input_active = True
        self.text_input_prompt = "Enter mix name:"
        self.text_input_text = ""
        self.text_input_title = "Mix"
        
        # Wait for user input
        while self.text_input_active:
            pygame.time.wait(100)
        
        return self.text_input_text if self.text_input_text else None
    
    def _show_text_input_dialog(self, prompt: str, title: str = "Input"):
        """Show text input dialog"""
        self.text_input_active = True
        self.text_input_prompt = prompt
        self.text_input_text = ""
        self.text_input_title = title
    
    def _show_selection_dialog(self, title: str, options: List[str]):
        """Show selection dialog"""
        self.selection_dialog_active = True
        self.selection_dialog_title = title
        self.selection_dialog_options = options
        self.selection_dialog_selected = 0
    
    def _handle_text_input_event(self, event):
        """Handle text input dialog events"""
        if not self.text_input_active:
            return False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Confirm input
                self.text_input_active = False
                return True
            elif event.key == pygame.K_ESCAPE:
                # Cancel input
                self.text_input_active = False
                self.text_input_text = ""
                return True
            elif event.key == pygame.K_BACKSPACE:
                # Remove last character
                self.text_input_text = self.text_input_text[:-1]
                return True
            elif event.unicode.isprintable():
                # Add character
                self.text_input_text += event.unicode
                return True
        
        return False
    
    def _handle_selection_dialog_event(self, event):
        """Handle selection dialog events"""
        if not self.selection_dialog_active:
            return False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Confirm selection
                selected_option = self.selection_dialog_options[self.selection_dialog_selected]
                self.selection_dialog_active = False
                
                # Check if this is a mix loading dialog or track addition dialog
                if hasattr(self, 'current_inventory') and self.current_inventory:
                    # This is a track addition dialog
                    self._add_selected_track_to_daw(selected_option)
                else:
                    # This is a mix loading dialog
                    success = self.mix_manager.load_mix(selected_option, self)
                    if success:
                        debug_info(f"✅ Mix '{selected_option}' loaded successfully")
                        self._refresh_daw_after_load()
                    else:
                        debug_error(f"❌ Failed to load mix '{selected_option}'")
                
                return True
            elif event.key == pygame.K_ESCAPE:
                # Cancel selection
                self.selection_dialog_active = False
                return True
            elif event.key == pygame.K_UP:
                # Move selection up
                self.selection_dialog_selected = max(0, self.selection_dialog_selected - 1)
                return True
            elif event.key == pygame.K_DOWN:
                # Move selection down
                self.selection_dialog_selected = min(len(self.selection_dialog_options) - 1, 
                                                   self.selection_dialog_selected + 1)
                return True
        
        return False
    
    def _refresh_daw_after_load(self):
        """Refresh DAW state after loading a mix"""
        # Stop any playing audio
        self.audio.stop_playback()
        
        # Reset playhead
        self.audio.playhead_position = 0.0
        
        # Clear selections
        self.clip_ops.clear_selection()
        
        # Update timeline pan limits
        self.timeline._update_max_pan_offset()
        
        # Clear waveform cache if it exists
        if hasattr(self, 'waveform_cache'):
            self.waveform_cache.clear()
        
        debug_info("🎵 DAW refreshed after loading mix")
    
    def _verify_loaded_state(self):
        """Verify the loaded DAW state"""
        debug_info(f"🎵 DAW state verification:")
        debug_info(f"  - Tracks: {len(self.tracks)}")
        debug_info(f"  - Total clips: {sum(len(track.clips) for track in self.tracks)}")
        debug_info(f"  - Playhead: {self.audio.playhead_position:.2f}s")
        debug_info(f"  - Timeline offset: {self.timeline.timeline_offset:.2f}s")
    
    def _has_soloed_tracks(self) -> bool:
        """Check if any tracks are soloed"""
        return any(track.is_soloed for track in self.tracks)
    
    def _auto_arrange_clips(self):
        """Auto-arrange clips to prevent overlaps"""
        self.clip_ops._auto_arrange_clips()
    
    def _select_clip_at_position(self, track_index: int, time_position: float):
        """Select a clip at a specific position"""
        self.timeline._select_clip_at_position(track_index, time_position)
    
    def _add_track_from_inventory(self):
        """Add a track from inventory to the DAW"""
        if not hasattr(self, 'current_inventory') or not self.current_inventory:
            debug_info("No tracks in inventory to add")
            return
        
        # Show selection dialog for inventory tracks
        inventory_names = [track.name for track in self.current_inventory]
        self._show_selection_dialog("Select track to add:", inventory_names)
        
        # The actual track addition will happen when user confirms selection
        # in the _handle_selection_dialog_event method
    
    def _add_selected_track_to_daw(self, track_name: str):
        """Add the selected track from inventory to the DAW"""
        if not hasattr(self, 'current_inventory') or not self.current_inventory:
            debug_error("No inventory available")
            return
        
        # Find the track in inventory by name
        selected_track = None
        for track in self.current_inventory:
            if track.name == track_name:
                selected_track = track
                break
        
        if not selected_track:
            debug_error(f"Track '{track_name}' not found in inventory")
            return
        
        # Find an empty track slot
        empty_track_index = None
        for i, track in enumerate(self.tracks):
            if len(track.clips) == 0:
                empty_track_index = i
                break
        
        if empty_track_index is None:
            debug_error("No empty tracks available")
            return
        
        # Add the track to the DAW
        success = self.add_track_from_inventory(selected_track, empty_track_index)
        if success:
            debug_info(f"✅ Added '{track_name}' to Track {empty_track_index + 1}")
        else:
            debug_error(f"❌ Failed to add '{track_name}' to DAW")
    
    # Event handling
    def handle_event(self, event, inventory=None):
        """Handle pygame events"""
        # Store inventory reference for use in clip operations
        if inventory is not None:
            self.current_inventory = inventory
        
        # Handle text input dialog
        if self._handle_text_input_event(event):
            return True
        
        # Handle selection dialog
        if self._handle_selection_dialog_event(event):
            return True
        
        # Handle keyboard events
        if event.type == pygame.KEYDOWN:
            return self._handle_key_down(event)
        
        # Handle mouse events
        elif event.type == pygame.MOUSEBUTTONDOWN:
            return self._handle_mouse_down(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            return self._handle_mouse_up(event)
        elif event.type == pygame.MOUSEMOTION:
            return self._handle_mouse_motion(event)
        elif event.type == pygame.MOUSEWHEEL:
            return self._handle_mouse_wheel(event)
        
        return False
    
    def _handle_key_down(self, event):
        """Handle keyboard key down events"""
        # Check for CTRL+SHIFT combinations first
        if event.mod & pygame.KMOD_CTRL and event.mod & pygame.KMOD_SHIFT:
            if event.key == pygame.K_s:  # CTRL+SHIFT+S
                self.save_current_mix()
                return True
            elif event.key == pygame.K_e:  # CTRL+SHIFT+E
                self.export_current_mix()
                return True
            elif event.key == pygame.K_l:  # CTRL+SHIFT+L
                self.load_saved_mix()
                return True
        
        # Check for CTRL combinations
        elif event.mod & pygame.KMOD_CTRL:
            if event.key == pygame.K_z:  # CTRL+Z
                self.undo()
                return True
            elif event.key == pygame.K_y:  # CTRL+Y
                self.redo()
                return True
            elif event.key == pygame.K_i:  # CTRL+I
                # Insert clip (placeholder)
                debug_info("Insert clip functionality not yet implemented in refactored version")
                return True
        
        # Regular key handling
        else:
            if event.key == pygame.K_SPACE:
                self.toggle_playback()
                return True
            elif event.key == pygame.K_r:
                self.audio.toggle_recording()
                return True
            elif event.key == pygame.K_s:
                self.stop()
                return True
            elif event.key == pygame.K_m:
                # Toggle mute for selected tracks
                debug_info("Mute toggle not yet implemented in refactored version")
                return True
            elif event.key == pygame.K_l:
                self.audio.toggle_loop()
                return True
            elif event.key == pygame.K_DELETE:
                self.delete_selected_clips()
                return True
            elif event.key == pygame.K_i:  # Regular I key to add tracks from inventory
                self._add_track_from_inventory()
                return True
            elif event.key == pygame.K_LEFT:
                self.pan_left_by_visible_amount()
                return True
            elif event.key == pygame.K_RIGHT:
                self.pan_right_by_visible_amount()
                return True
            elif event.key == pygame.K_EQUALS:
                self.zoom_in()
                return True
            elif event.key == pygame.K_MINUS:
                self.zoom_out()
                return True
            elif event.key == pygame.K_ESCAPE:
                # Close dialogs if they're active
                if self.text_input_active or self.selection_dialog_active:
                    self.text_input_active = False
                    self.selection_dialog_active = False
                    return True
                else:
                    # No dialogs active, close the DAW
                    self.should_close = True
                    return True
            elif event.key == pygame.K_RETURN:
                # Handle dialog confirmations
                return True
        
        return False
    
    def _handle_mouse_down(self, event):
        """Handle mouse button down events"""
        x, y = event.pos
        button = event.button
        
        # Handle track control button clicks first
        if button == 1:  # Left button
            control_click = self.ui.handle_track_control_click(x, y)
            if control_click:
                track_index, button_type = control_click
                track = self.tracks[track_index]
                
                if button_type == "mute":
                    track.toggle_mute()
                    return True
                elif button_type == "solo":
                    track.toggle_solo()
                    return True
                elif button_type == "record":
                    track.toggle_record()
                    return True
        
        # Handle scroll bar interaction
        if self.timeline.handle_scroll_bar_interaction(x, y, button):
            return True
        
        # Handle timeline interaction
        if self.timeline.handle_timeline_click(x, y, button):
            return True
        
        # Start drag operation
        if button == 1:  # Left button
            self.dragging = True
            self.timeline.handle_timeline_drag(x, y, button)
            return True
        
        return False
    
    def _handle_mouse_up(self, event):
        """Handle mouse button up events"""
        x, y = event.pos
        button = event.button
        
        # End drag operation
        if button == 1 and self.dragging:  # Left button
            self.dragging = False
            self.timeline.handle_timeline_drag_end(x, y, button)
            return True
        
        return False
    
    def _handle_mouse_motion(self, event):
        """Handle mouse motion events"""
        self.mouse_pos = event.pos
        
        # Update drag operation
        if self.dragging:
            self.timeline.handle_timeline_drag(event.pos[0], event.pos[1], 1)
            return True
        
        return False
    
    def _handle_mouse_wheel(self, event):
        """Handle mouse wheel events for zooming"""
        if hasattr(event, 'mod') and event.mod & pygame.KMOD_SHIFT:
            # Shift + scroll for horizontal panning
            if event.y > 0:
                self.pan_left()
            else:
                self.pan_right()
        else:
            # Regular scroll for zooming
            if event.y > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        
        return True
    
    # Missing methods that the UI needs
    def _get_content_duration(self):
        """Get the total duration of all content in the DAW"""
        max_duration = 0.0
        for track in self.tracks:
            for clip in track.clips:
                clip_end = clip.start_time + clip.duration
                max_duration = max(max_duration, clip_end)
        return max(max_duration, self.max_duration)
    
    def pan_left(self, amount):
        """Pan the timeline left"""
        self.timeline.pan_left(amount)
    
    def pan_right(self, amount):
        """Pan the timeline right"""
        self.timeline.pan_right(amount)
    
    def zoom_in(self):
        """Zoom in on the timeline"""
        self.timeline.zoom_in()
    
    def zoom_out(self):
        """Zoom out on the timeline"""
        self.timeline.zoom_out()
    
    @property
    def timeline_x(self):
        """Get timeline X position"""
        return 200  # Fixed position for now
    
    @property
    def timeline_y(self):
        """Get timeline Y position"""
        return 100  # Fixed position for now
    
    @property
    def timeline_width(self):
        """Get timeline width"""
        return self.daw_width - 400  # Full width minus margins
    
    @property
    def timeline_height(self):
        """Get timeline height"""
        return self.daw_height - 300  # Full height minus margins
    
    @property
    def max_timeline_offset(self):
        """Get maximum timeline offset for panning"""
        return max(0, self._get_content_duration() - (self.timeline_width / self.pixels_per_second))
    
    @property
    def min_timeline_offset(self):
        """Get minimum timeline offset for panning"""
        return 0.0
    
    # Update and render
    def update(self, dt: float):
        """Update DAW state"""
        # Update audio system
        self.audio.update_playback(dt)
        
        # Update timeline
        self.timeline._update_max_pan_offset()
    
    def render(self, surface, inventory):
        """Render the DAW"""
        # Render timeline
        self.ui.render_timeline(surface)
        
        # Render tracks
        self.ui.render_tracks(surface)
        
        # Render clips
        self.ui.render_clips(surface)
        
        # Render playhead on top of clips
        self.ui.render_playhead(surface)
        
        # Render help text
        self.ui.render_help_text(surface)
        
        # Render dialogs
        if self.text_input_active:
            self.ui.render_text_input_dialog(surface, self.text_input_prompt, 
                                          self.text_input_text, self.text_input_title)
        
        if self.selection_dialog_active:
            self.ui.render_selection_dialog(surface, self.selection_dialog_title,
                                         self.selection_dialog_options, 
                                         self.selection_dialog_selected)
    
    def cleanup(self):
        """Clean up DAW resources"""
        self.audio.cleanup()
        debug_info("🎵 Refactored CosmicDAW cleaned up")
    
    def close(self):
        """Mark the DAW for closing"""
        self.should_close = True
        debug_info("🎵 CosmicDAW marked for closing")
    
    def reset_state(self):
        """Reset DAW state when closing"""
        self.should_close = False
        self.text_input_active = False
        self.selection_dialog_active = False
        self.dragging = False
        debug_info("🎵 CosmicDAW state reset")
