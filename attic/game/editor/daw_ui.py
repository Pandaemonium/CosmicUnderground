"""
DAW UI rendering and interaction methods
Extracted from cosmic_daw.py for better modularity
"""

import pygame
from typing import List, Optional, Tuple
from ..core.debug import debug_info, debug_error
from ..core.config import (
    TRACK_COLORS, CLIP_FILL, CLIP_BORDER, PLAYHEAD_COLOR, 
    GRID_COLOR, TIMELINE_COLOR, TRACK_LABEL_COLOR, HELP_COLOR
)


class DAWUI:
    """Handles UI rendering and interaction for the DAW"""
    
    def __init__(self, daw):
        self.daw = daw
        
        # UI state
        self.show_help = True
        self.show_timeline = True
        self.show_grid = True
        self.show_scroll_bar = True
        
        # Colors
        self.help_text_color = (100, 200, 255)  # Bright blue
        self.help_bg_color = (0, 0, 0)  # Black
        self.help_bg_alpha = 180  # Semi-transparent
    
    def render_timeline(self, surface):
        """Render the main timeline area"""
        # Draw timeline background
        pygame.draw.rect(surface, TIMELINE_COLOR, 
                        (self.daw.timeline_x, self.daw.timeline_y, 
                         self.daw.timeline_width, self.daw.timeline_height))
        
        # Draw grid
        if self.show_grid:
            self._render_grid(surface)
        
        # Draw time markers
        self._render_time_markers(surface)
        
        # Note: Playhead is now rendered separately after clips to ensure it's on top
        
        # Draw scroll bar
        if self.show_scroll_bar:
            self._render_scroll_bar(surface)
    
    def render_tracks(self, surface):
        """Render all tracks"""
        for i, track in enumerate(self.daw.tracks):
            self._render_track(surface, track, i)
    
    def render_clips(self, surface):
        """Render all clips on all tracks"""
        # Set clipping to prevent waveforms from rendering outside timeline
        original_clip = surface.get_clip()
        surface.set_clip((self.daw.timeline_x, self.daw.timeline_y, 
                         self.daw.timeline_width, self.daw.timeline_height))
        
        # Debug: Print clip information (commented out for performance)
        total_clips = sum(len(track.clips) for track in self.daw.tracks)
        # if total_clips > 0:
        #     debug_info(f"🎵 Rendering {total_clips} clips across {len(self.daw.tracks)} tracks")
        
        for track in self.daw.tracks:
            if track.clips:
                # debug_info(f"🎵 Track {track.name} has {len(track.clips)} clips")
                for clip in track.clips:
                    # debug_info(f"🎵 Rendering clip: {clip.name} at time {clip.start_time}s on track {clip.track_index}")
                    clip.render(surface, self.daw.timeline_x, self.daw.timeline_y,
                               self.daw.pixels_per_second, self.daw.track_height, 
                               self.daw.track_spacing)
        
        # Restore original clipping
        surface.set_clip(original_clip)
    
    def render_playhead(self, surface):
        """Render the playhead line on top of everything else"""
        # Calculate playhead position accounting for timeline offset
        playhead_x = self.daw.timeline_x + ((self.daw.playhead_position - self.daw.timeline_offset) * self.daw.pixels_per_second)
        
        # Only draw if playhead is visible
        if (self.daw.timeline_x <= playhead_x <= self.daw.timeline_x + self.daw.timeline_width):
            pygame.draw.line(surface, PLAYHEAD_COLOR, 
                           (playhead_x, self.daw.timeline_y), 
                           (playhead_x, self.daw.timeline_y + self.daw.timeline_height), 3)
    
    def render_help_text(self, surface):
        """Render helpful DAW instructions with high-contrast colors"""
        y = self.daw.daw_height - 200  # More space from bottom
        
        # Create a semi-transparent black background for the help text area
        help_bg_width = self.daw.daw_width - 40  # Full width minus margins
        help_bg_height = 200
        help_bg_rect = pygame.Rect(20, y - 10, help_bg_width, help_bg_height)
        
        # Draw black background with some transparency
        help_bg_surface = pygame.Surface((help_bg_width, help_bg_height))
        help_bg_surface.fill(self.help_bg_color)  # Pure black
        help_bg_surface.set_alpha(self.help_bg_alpha)  # Semi-transparent
        surface.blit(help_bg_surface, help_bg_rect)
        
        # Draw border
        pygame.draw.rect(surface, (50, 50, 50), help_bg_rect, 2)
        
        # Draw help text with high contrast
        help_lines = [
            "SPACE: Play/Pause | R: Record | S: Stop | M: Mute | L: Loop",
            "I: Insert Clip | DELETE: Delete Selected | CTRL+Z: Undo | CTRL+Y: Redo",
            "CTRL+SHIFT+S: Save Mix | CTRL+SHIFT+E: Export WAV | CTRL+SHIFT+L: Load Mix",
            "Mouse Wheel: Zoom | SHIFT+Wheel: Pan | Arrow Keys: Pan Timeline",
            "Click Timeline: Set Playhead | Drag Clips: Move | Double-Click: Select"
        ]
        
        y_offset = y
        for line in help_lines:
            try:
                text_surface = self.daw.fonts[1].render(line, True, self.help_text_color)
                text_rect = text_surface.get_rect(left=30, top=y_offset)
                surface.blit(text_surface, text_rect)
                y_offset += 25
            except Exception as e:
                debug_info(f"Could not render help text line: {e}")
    
    def render_text_input_dialog(self, surface, prompt: str, current_text: str, title: str = "Input"):
        """Render a text input dialog overlay"""
        # Create semi-transparent overlay
        overlay = pygame.Surface((self.daw.daw_width, self.daw.daw_height))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        surface.blit(overlay, (0, 0))
        
        # Calculate dialog dimensions
        dialog_width = 400
        dialog_height = 150
        dialog_x = (self.daw.daw_width - dialog_width) // 2
        dialog_y = (self.daw.daw_height - dialog_height) // 2
        
        # Draw dialog background
        dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
        pygame.draw.rect(surface, (50, 50, 50), dialog_rect)
        pygame.draw.rect(surface, (100, 100, 100), dialog_rect, 2)
        
        # Draw title
        try:
            title_surface = self.daw.fonts[0].render(title, True, (255, 255, 255))
            title_rect = title_surface.get_rect(centerx=dialog_x + dialog_width//2, top=dialog_y + 10)
            surface.blit(title_surface, title_rect)
        except Exception as e:
            debug_info(f"Could not render dialog title: {e}")
        
        # Draw prompt
        try:
            prompt_surface = self.daw.fonts[1].render(prompt, True, (200, 200, 200))
            prompt_rect = prompt_surface.get_rect(left=dialog_x + 20, top=dialog_y + 50)
            surface.blit(prompt_surface, prompt_rect)
        except Exception as e:
            debug_info(f"Could not render dialog prompt: {e}")
        
        # Draw text input box
        input_rect = pygame.Rect(dialog_x + 20, dialog_y + 80, dialog_width - 40, 30)
        pygame.draw.rect(surface, (255, 255, 255), input_rect)
        pygame.draw.rect(surface, (100, 100, 100), input_rect, 2)
        
        # Draw current text
        try:
            text_surface = self.daw.fonts[1].render(current_text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(left=input_rect.x + 5, centery=input_rect.centery)
            surface.blit(text_surface, text_rect)
        except Exception as e:
            debug_info(f"Could not render dialog text: {e}")
        
        # Draw instructions
        try:
            instructions = "Press ENTER to confirm, ESC to cancel"
            inst_surface = self.daw.fonts[2].render(instructions, True, (150, 150, 150))
            inst_rect = inst_surface.get_rect(centerx=dialog_x + dialog_width//2, bottom=dialog_y + dialog_height - 10)
            surface.blit(inst_surface, inst_rect)
        except Exception as e:
            debug_info(f"Could not render dialog instructions: {e}")
    
    def render_selection_dialog(self, surface, title: str, options: List[str], selected_index: int):
        """Render a selection dialog overlay"""
        # Create semi-transparent overlay
        overlay = pygame.Surface((self.daw.daw_width, self.daw.daw_height))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        surface.blit(overlay, (0, 0))
        
        # Calculate dialog dimensions
        max_option_width = max(len(option) for option in options) if options else 20
        dialog_width = max(400, max_option_width * 10 + 40)
        dialog_height = 50 + len(options) * 30
        dialog_x = (self.daw.daw_width - dialog_width) // 2
        dialog_y = (self.daw.daw_height - dialog_height) // 2
        
        # Draw dialog background
        dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
        pygame.draw.rect(surface, (50, 50, 50), dialog_rect)
        pygame.draw.rect(surface, (100, 100, 100), dialog_rect, 2)
        
        # Draw title
        try:
            title_surface = self.daw.fonts[0].render(title, True, (255, 255, 255))
            title_rect = title_surface.get_rect(centerx=dialog_x + dialog_width//2, top=dialog_y + 10)
            surface.blit(title_surface, title_rect)
        except Exception as e:
            debug_info(f"Could not render selection dialog title: {e}")
        
        # Draw options
        y_offset = dialog_y + 50
        for i, option in enumerate(options):
            # Highlight selected option
            color = (255, 255, 100) if i == selected_index else (200, 200, 200)
            try:
                option_surface = self.daw.fonts[1].render(option, True, color)
                option_rect = option_surface.get_rect(left=dialog_x + 20, top=y_offset)
                surface.blit(option_surface, option_rect)
                y_offset += 30
            except Exception as e:
                debug_info(f"Could not render selection option: {e}")
        
        # Draw instructions
        try:
            instructions = "Use UP/DOWN arrows, ENTER to select, ESC to cancel"
            inst_surface = self.daw.fonts[2].render(instructions, True, (150, 150, 150))
            inst_rect = inst_surface.get_rect(centerx=dialog_x + dialog_width//2, bottom=dialog_y + dialog_height - 10)
            surface.blit(inst_surface, inst_rect)
        except Exception as e:
            debug_info(f"Could not render selection instructions: {e}")
    
    def _render_track(self, surface, track, track_index):
        """Render a single track"""
        # Calculate track position
        y = self.daw.timeline_y + (track_index * (self.daw.track_height + self.daw.track_spacing))
        
        # Draw track background
        track_color = track.color if hasattr(track, 'color') else TRACK_COLORS[track_index % len(TRACK_COLORS)]
        pygame.draw.rect(surface, track_color, 
                        (0, y, self.daw.timeline_x, self.daw.track_height))
        
        # Draw track border
        pygame.draw.rect(surface, (100, 100, 100), 
                        (0, y, self.daw.timeline_x, self.daw.track_height), 1)
        
        # Draw track name
        try:
            track_name = track.name if hasattr(track, 'name') else f"Track {track_index + 1}"
            
            # Calculate the visible clip area for this track
            visible_clip_start = self.daw.timeline_offset
            visible_clip_end = visible_clip_start + (self.daw.timeline_width / self.daw.pixels_per_second)
            
            # Find clips that are currently visible
            visible_clips = []
            for clip in track.clips:
                clip_start = clip.start_time
                clip_end = clip.start_time + clip.duration
                
                # Check if clip overlaps with visible area
                if (clip_start < visible_clip_end and clip_end > visible_clip_start):
                    visible_clips.append(clip)
            
            if visible_clips:
                # Calculate the center of visible clips for track name positioning
                visible_start = max(visible_clip_start, min(clip.start_time for clip in visible_clips))
                visible_end = min(visible_clip_end, max(clip.start_time + clip.duration for clip in visible_clips))
                visible_center = (visible_start + visible_end) / 2
                
                # Convert time to screen position
                name_x = self.daw.timeline_x + ((visible_center - self.daw.timeline_offset) * self.daw.pixels_per_second)
            else:
                # No visible clips, show name at timeline center
                name_x = self.daw.timeline_x + (self.daw.timeline_width / 2)
            
            # Use a larger font size for better visibility
            try:
                # Try to use a larger font if available
                large_font = pygame.font.Font(None, 32)  # 32 instead of default small font
                text_surface = large_font.render(track_name, True, TRACK_LABEL_COLOR)
            except:
                # Fallback to the original font
                text_surface = self.daw.fonts[1].render(track_name, True, TRACK_LABEL_COLOR)
            
            text_rect = text_surface.get_rect(centerx=name_x, centery=y + self.daw.track_height//2)
            surface.blit(text_surface, text_rect)
            
            # Debug: Print track rendering info (commented out for performance)
            # debug_info(f"🎵 Rendered track '{track_name}' at y={y}, text_rect={text_rect}")
        except Exception as e:
            debug_info(f"Could not render track name: {e}")
        
        # Draw track controls (mute, solo, record)
        self._render_track_controls(surface, track, y)
    
    def _render_track_controls(self, surface, track, y):
        """Render track control buttons"""
        control_size = 20
        control_spacing = 25
        start_x = self.daw.timeline_x - 100
        
        # Mute button
        mute_color = (255, 100, 100) if getattr(track, 'is_muted', False) else (100, 100, 100)
        mute_rect = pygame.Rect(start_x, y + 5, control_size, control_size)
        pygame.draw.rect(surface, mute_color, mute_rect)
        pygame.draw.rect(surface, (200, 200, 200), mute_rect, 1)
        
        # Draw M label
        try:
            mute_text = self.daw.fonts[2].render("M", True, (255, 255, 255))
            mute_text_rect = mute_text.get_rect(center=mute_rect.center)
            surface.blit(mute_text, mute_text_rect)
        except Exception as e:
            debug_info(f"Could not render mute label: {e}")
        
        # Solo button
        solo_color = (100, 255, 100) if getattr(track, 'is_soloed', False) else (100, 100, 100)
        solo_rect = pygame.Rect(start_x + control_spacing, y + 5, control_size, control_size)
        pygame.draw.rect(surface, solo_color, solo_rect)
        pygame.draw.rect(surface, (200, 200, 200), solo_rect, 1)
        
        # Draw S label
        try:
            solo_text = self.daw.fonts[2].render("S", True, (255, 255, 255))
            solo_text_rect = solo_text.get_rect(center=solo_rect.center)
            surface.blit(solo_text, solo_text_rect)
        except Exception as e:
            debug_info(f"Could not render solo label: {e}")
        
        # Record button
        record_color = (255, 100, 100) if getattr(track, 'is_recording', False) else (100, 100, 100)
        record_rect = pygame.Rect(start_x + control_spacing * 2, y + 5, control_size, control_size)
        pygame.draw.rect(surface, record_color, record_rect)
        pygame.draw.rect(surface, (200, 200, 200), record_rect, 1)
        
        # Draw R label
        try:
            record_text = self.daw.fonts[2].render("R", True, (255, 255, 255))
            record_text_rect = record_text.get_rect(center=record_rect.center)
            surface.blit(record_text, record_text_rect)
        except Exception as e:
            debug_info(f"Could not render record label: {e}")
    
    def _render_grid(self, surface):
        """Render the timeline grid"""
        # Draw vertical grid lines every second
        grid_spacing = self.daw.pixels_per_second
        
        for x in range(0, self.daw.timeline_width, int(grid_spacing)):
            # Calculate actual time position accounting for timeline offset
            time_pos = (x / self.daw.pixels_per_second) + self.daw.timeline_offset
            if time_pos < 0:
                continue
                
            grid_x = self.daw.timeline_x + x
            pygame.draw.line(surface, GRID_COLOR, 
                           (grid_x, self.daw.timeline_y), 
                           (grid_x, self.daw.timeline_y + self.daw.timeline_height), 1)
        
        # Draw horizontal grid lines for tracks
        for i in range(len(self.daw.tracks) + 1):
            y = self.daw.timeline_y + (i * (self.daw.track_height + self.daw.track_spacing))
            pygame.draw.line(surface, GRID_COLOR, 
                           (self.daw.timeline_x, y), 
                           (self.daw.timeline_x + self.daw.timeline_width, y), 1)
    
    def _render_time_markers(self, surface):
        """Render time markers above the timeline"""
        # Draw time markers every 5 seconds
        marker_interval = 5.0
        marker_spacing = marker_interval * self.daw.pixels_per_second
        
        for x in range(0, self.daw.timeline_width, int(marker_spacing)):
            # Calculate actual time position accounting for timeline offset
            time_pos = (x / self.daw.pixels_per_second) + self.daw.timeline_offset
            if time_pos < 0:
                continue
                
            marker_x = self.daw.timeline_x + x
            
            # Draw marker line
            pygame.draw.line(surface, (150, 150, 150), 
                           (marker_x, self.daw.timeline_y - 20), 
                           (marker_x, self.daw.timeline_y), 2)
            
            # Draw time label
            try:
                time_text = f"{time_pos:.1f}s"
                text_surface = self.daw.fonts[2].render(time_text, True, (200, 200, 200))
                text_rect = text_surface.get_rect(centerx=marker_x, top=self.daw.timeline_y - 35)
                surface.blit(text_surface, text_rect)
            except Exception as e:
                debug_info(f"Could not render time marker: {e}")
    
    def _render_scroll_bar(self, surface):
        """Render the timeline scroll bar"""
        scroll_bar_height = 20
        scroll_bar_y = self.daw.timeline_y - scroll_bar_height - 5
        
        # Calculate scroll bar dimensions
        total_content_width = self.daw._get_content_duration() * self.daw.pixels_per_second
        visible_width = self.daw.timeline_width
        
        if total_content_width <= visible_width:
            return  # No need for scroll bar
        
        # Calculate scroll bar position and size
        scroll_ratio = self.daw.timeline_offset / (self.daw.max_timeline_offset - self.daw.min_timeline_offset)
        scroll_bar_width = max(50, (visible_width / total_content_width) * visible_width)
        scroll_bar_x = self.daw.timeline_x + (scroll_ratio * (visible_width - scroll_bar_width))
        
        # Draw scroll bar background
        scroll_bg_rect = pygame.Rect(self.daw.timeline_x, scroll_bar_y, visible_width, scroll_bar_height)
        pygame.draw.rect(surface, (80, 80, 80), scroll_bg_rect)
        
        # Draw scroll bar handle
        scroll_handle_rect = pygame.Rect(scroll_bar_x, scroll_bar_y, scroll_bar_width, scroll_bar_height)
        pygame.draw.rect(surface, (150, 150, 150), scroll_handle_rect)
        pygame.draw.rect(surface, (200, 200, 200), scroll_handle_rect, 1)
        
        # Draw scroll bar border
        pygame.draw.rect(surface, (100, 100, 100), scroll_bg_rect, 1)
    
    def handle_track_control_click(self, x: int, y: int) -> Optional[Tuple[int, str]]:
        """Handle clicks on track control buttons. Returns (track_index, button_type) or None"""
        for track_index, track in enumerate(self.daw.tracks):
            track_y = self.daw.timeline_y + (track_index * (self.daw.track_height + self.daw.track_spacing))
            
            # Check if click is within track bounds
            if track_y <= y <= track_y + self.daw.track_height:
                control_size = 20
                control_spacing = 25
                start_x = self.daw.timeline_x - 100
                
                # Check mute button
                mute_rect = pygame.Rect(start_x, track_y + 5, control_size, control_size)
                if mute_rect.collidepoint(x, y):
                    return (track_index, "mute")
                
                # Check solo button
                solo_rect = pygame.Rect(start_x + control_spacing, track_y + 5, control_size, control_size)
                if solo_rect.collidepoint(x, y):
                    return (track_index, "solo")
                
                # Check record button
                record_rect = pygame.Rect(start_x + control_spacing * 2, track_y + 5, control_size, control_size)
                if record_rect.collidepoint(x, y):
                    return (track_index, "record")
        
        return None
