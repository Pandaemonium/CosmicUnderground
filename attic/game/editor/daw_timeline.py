"""
DAW Timeline management and navigation methods
Extracted from cosmic_daw.py for better modularity
"""

import pygame
from typing import List, Optional, Tuple
from ..core.debug import debug_info, debug_error, debug_warning


class DAWTimeline:
    """Handles timeline management and navigation for the DAW"""
    
    def __init__(self, daw):
        self.daw = daw
        
        # Timeline dimensions
        self.timeline_x = 200
        self.timeline_y = 100
        self.timeline_width = 800
        self.timeline_height = 600
        
        # Zoom and pan settings
        self.pixels_per_second = 50.0
        self.min_pixels_per_second = 10.0
        self.max_pixels_per_second = 200.0
        
        # Panning
        self.timeline_offset = 0.0
        self.min_timeline_offset = 0.0
        self.max_timeline_offset = 0.0
        
        # Snap settings
        self.snap_enabled = True
        self.snap_interval = 1.0  # Snap to 1-second intervals
        
        # Selection
        self.selected_clips = []
        self.selection_start = None
        self.selection_end = None
        
        # Drag and drop
        self.dragging_clip = None
        self.drag_start_pos = None
        self.drag_start_time = None
        self.drag_start_track = None
    
    def zoom_in(self):
        """Zoom in on the timeline"""
        old_zoom = self.pixels_per_second
        self.pixels_per_second = min(self.max_pixels_per_second, self.pixels_per_second * 1.2)
        
        if self.pixels_per_second != old_zoom:
            self._update_max_pan_offset()
            debug_info(f"🎵 DAW: Zoomed in to {self.pixels_per_second:.1f} px/s")
    
    def zoom_out(self):
        """Zoom out on the timeline"""
        old_zoom = self.pixels_per_second
        self.pixels_per_second = max(self.min_pixels_per_second, self.pixels_per_second / 1.2)
        
        if self.pixels_per_second != old_zoom:
            self._update_max_pan_offset()
            debug_info(f"🎵 DAW: Zoomed out to {self.pixels_per_second:.1f} px/s")
    
    def pan_left(self, amount: float):
        """Pan the timeline left by a specified amount"""
        self.timeline_offset = max(self.min_timeline_offset, self.timeline_offset - amount)
        debug_info(f"🎵 DAW: Panned left to {self.timeline_offset:.2f}s")
    
    def pan_right(self, amount: float):
        """Pan the timeline right by a specified amount"""
        self.timeline_offset = min(self.max_timeline_offset, self.timeline_offset + amount)
        debug_info(f"🎵 DAW: Panned right to {self.timeline_offset:.2f}s")
    
    def pan_left_by_visible_amount(self):
        """Pan left by one quarter of the visible time"""
        visible_time = self.timeline_width / self.pixels_per_second
        pan_amount = visible_time * 0.25
        self.pan_left(pan_amount)
    
    def pan_right_by_visible_amount(self):
        """Pan right by one quarter of the visible time"""
        visible_time = self.timeline_width / self.pixels_per_second
        pan_amount = visible_time * 0.25
        self.pan_right(pan_amount)
    
    def center_on_playhead(self):
        """Center the timeline view on the current playhead position"""
        visible_time = self.timeline_width / self.pixels_per_second
        center_time = self.daw.playhead_position
        self.timeline_offset = max(0, center_time - (visible_time / 2))
        debug_info(f"🎵 DAW: Centered on playhead at {center_time:.2f}s")
    
    def fit_content_to_view(self):
        """Adjust zoom and pan to fit all content in the view"""
        content_duration = self._get_content_duration()
        if content_duration <= 0:
            return
        
        # Calculate optimal zoom
        optimal_zoom = self.timeline_width / content_duration
        self.pixels_per_second = max(self.min_pixels_per_second, 
                                   min(self.max_pixels_per_second, optimal_zoom))
        
        # Reset pan to show beginning
        self.timeline_offset = 0.0
        
        # Update pan limits
        self._update_max_pan_offset()
        
        debug_info(f"🎵 DAW: Fitted content to view, zoom: {self.pixels_per_second:.1f} px/s")
    
    def snap_time_to_grid(self, time: float) -> float:
        """Snap a time value to the grid if snapping is enabled"""
        if not self.snap_enabled:
            return time
        
        snapped_time = round(time / self.snap_interval) * self.snap_interval
        return max(0.0, snapped_time)
    
    def toggle_snap(self):
        """Toggle snap to grid on/off"""
        self.snap_enabled = not self.snap_enabled
        debug_info(f"🎵 DAW: Snap to grid {'enabled' if self.snap_enabled else 'disabled'}")
    
    def set_snap_interval(self, interval: float):
        """Set the snap interval in seconds"""
        if interval <= 0:
            debug_warning("Snap interval must be positive")
            return
        
        self.snap_interval = interval
        debug_info(f"🎵 DAW: Snap interval set to {interval:.2f}s")
    
    def screen_to_time(self, screen_x: int) -> float:
        """Convert screen X coordinate to timeline time"""
        rel_x = screen_x - self.timeline_x
        time_position = (rel_x / self.pixels_per_second) + self.timeline_offset
        return max(0.0, time_position)
    
    def time_to_screen(self, time: float) -> int:
        """Convert timeline time to screen X coordinate"""
        rel_time = time - self.timeline_offset
        screen_x = self.timeline_x + (rel_time * self.pixels_per_second)
        return int(screen_x)
    
    def is_time_visible(self, time: float) -> bool:
        """Check if a time position is currently visible in the timeline"""
        screen_x = self.time_to_screen(time)
        return self.timeline_x <= screen_x <= self.timeline_x + self.timeline_width
    
    def get_visible_time_range(self) -> Tuple[float, float]:
        """Get the start and end times currently visible in the timeline"""
        start_time = self.timeline_offset
        end_time = start_time + (self.timeline_width / self.pixels_per_second)
        return start_time, end_time
    
    def handle_timeline_click(self, x: int, y: int, button: int) -> bool:
        """Handle clicks on the timeline area"""
        if not self._is_click_in_timeline(x, y):
            return False
        
        # Convert screen coordinates to timeline time
        time_position = self.screen_to_time(x)
        
        # Calculate track index
        track_index = self._screen_y_to_track_index(y)
        
        if 0 <= track_index < len(self.daw.tracks):
            if button == 1:  # Left click
                # Set playhead position
                self.daw.playhead_position = time_position
                
                # Update the specific track's playhead position
                if hasattr(self.daw.tracks[track_index], 'set_playhead_position'):
                    self.daw.tracks[track_index].set_playhead_position(time_position)
                
                # CRITICAL: If playback is running, seek audio to new position
                if self.daw.is_playing:
                    self.daw.seek_to_position(time_position)
                
                debug_info(f"🎵 DAW: Set playhead to {time_position:.2f}s on track {track_index}")
                return True
            
            elif button == 3:  # Right click
                # Select clip at position
                self._select_clip_at_position(track_index, time_position)
                return True
        
        return False
    
    def handle_timeline_drag(self, x: int, y: int, button: int) -> bool:
        """Handle drag operations on the timeline"""
        if not self._is_click_in_timeline(x, y):
            return False
        
        if button == 1:  # Left button drag
            if self.dragging_clip is None:
                # Start dragging
                self._start_clip_drag(x, y)
            else:
                # Continue dragging
                self._update_clip_drag(x, y)
            return True
        
        return False
    
    def handle_timeline_drag_end(self, x: int, y: int, button: int) -> bool:
        """Handle end of drag operations on the timeline"""
        if self.dragging_clip is None:
            return False
        
        if button == 1:  # Left button drag end
            self._finish_clip_drag(x, y)
            return True
        
        return False
    
    def handle_scroll_bar_interaction(self, x: int, y: int, button: int) -> bool:
        """Handle interactions with the scroll bar"""
        scroll_bar_height = 20
        scroll_bar_y = self.timeline_y - scroll_bar_height - 5
        
        # Check if click is in scroll bar area
        if (self.timeline_x <= x <= self.timeline_x + self.timeline_width and
            scroll_bar_y <= y <= scroll_bar_y + scroll_bar_height):
            
            if button == 1:  # Left click
                self._scroll_bar_click(x, y)
                return True
        
        return False
    
    def _is_click_in_timeline(self, x: int, y: int) -> bool:
        """Check if a click is within the timeline area"""
        return (self.timeline_x <= x <= self.timeline_x + self.timeline_width and
                self.timeline_y <= y <= self.timeline_y + self.timeline_height)
    
    def _screen_y_to_track_index(self, y: int) -> int:
        """Convert screen Y coordinate to track index"""
        rel_y = y - self.timeline_y
        return int(rel_y / (self.daw.track_height + self.daw.track_spacing))
    
    def _select_clip_at_position(self, track_index: int, time_position: float):
        """Select a clip at a specific position"""
        track = self.daw.tracks[track_index]
        
        # Find clip at this time position
        for clip in track.clips:
            if (clip.start_time <= time_position <= 
                clip.start_time + clip.duration):
                
                # Clear previous selection
                self.selected_clips.clear()
                
                # Select this clip
                clip.is_selected = True
                self.selected_clips.append(clip)
                
                debug_info(f"Selected clip: {clip.name}")
                break
    
    def _start_clip_drag(self, x: int, y: int):
        """Start dragging a clip"""
        time_position = self.screen_to_time(x)
        track_index = self._screen_y_to_track_index(y)
        
        # Find clip at this position
        if 0 <= track_index < len(self.daw.tracks):
            track = self.daw.tracks[track_index]
            for clip in track.clips:
                if (clip.start_time <= time_position <= 
                    clip.start_time + clip.duration):
                    
                    self.dragging_clip = clip
                    self.drag_start_pos = (x, y)
                    self.drag_start_time = clip.start_time
                    self.drag_start_track = track_index
                    
                    debug_info(f"Started dragging clip: {clip.name}")
                    break
    
    def _update_clip_drag(self, x: int, y: int):
        """Update clip drag position"""
        if self.dragging_clip is None:
            return
        
        # Calculate new time position and track index - MUST account for timeline offset (panning)
        time_position = (x - self.timeline_x) / self.pixels_per_second + self.timeline_offset
        new_track_index = self._screen_y_to_track_index(y)
        
        # Snap to grid if enabled
        if self.snap_enabled:
            time_position = self.snap_time_to_grid(time_position)
        
        # Update clip position
        self.dragging_clip.start_time = max(0.0, time_position)
        if 0 <= new_track_index < len(self.daw.tracks):
            self.dragging_clip.track_index = new_track_index
    
    def _finish_clip_drag(self, x: int, y: int):
        """Finish dragging a clip"""
        if self.dragging_clip is None:
            return
        
        # Calculate new time position and track index - MUST account for timeline offset (panning)
        new_time = (x - self.timeline_x) / self.pixels_per_second + self.timeline_offset
        new_track_index = self._screen_y_to_track_index(y)
        
        # Snap to grid if enabled
        if self.snap_enabled:
            new_time = self.snap_time_to_grid(new_time)
        
        # Ensure valid position
        new_time = max(0.0, new_time)
        if new_track_index < 0:
            new_track_index = 0
        elif new_track_index >= len(self.daw.tracks):
            new_track_index = len(self.daw.tracks) - 1
        
        # Remove from old track
        old_track = self.daw.tracks[self.drag_start_track]
        if self.dragging_clip in old_track.clips:
            old_track.remove_clip(self.dragging_clip)
        
        # Add to new track
        new_track = self.daw.tracks[new_track_index]
        self.dragging_clip.start_time = new_time
        self.dragging_clip.track_index = new_track_index
        new_track.add_clip(self.dragging_clip)
        
        # Auto-arrange to prevent overlaps
        self.daw._auto_arrange_clips()
        
        debug_info(f"Finished dragging {self.dragging_clip.name} to track {new_track.name} at {new_time:.2f}s")
        
        # Clear drag state
        self.dragging_clip = None
        self.drag_start_pos = None
        self.drag_start_time = None
        self.drag_start_track = None
    
    def _scroll_bar_click(self, x: int, y: int):
        """Handle scroll bar click"""
        scroll_bar_height = 20
        scroll_bar_y = self.timeline_y - scroll_bar_height - 5
        
        # Calculate click position relative to scroll bar
        rel_x = x - self.timeline_x
        scroll_ratio = rel_x / self.timeline_width
        
        # Calculate new timeline offset
        total_range = self.max_timeline_offset - self.min_timeline_offset
        new_offset = self.min_timeline_offset + (scroll_ratio * total_range)
        
        # Clamp to valid range
        self.timeline_offset = max(self.min_timeline_offset, 
                                 min(self.max_timeline_offset, new_offset))
        
        debug_info(f"🎵 DAW: Scrolled to {self.timeline_offset:.2f}s")
    
    def _update_max_pan_offset(self):
        """Update the maximum pan offset based on content and zoom"""
        content_duration = self._get_content_duration()
        visible_duration = self.timeline_width / self.pixels_per_second
        
        if content_duration <= visible_duration:
            self.max_timeline_offset = 0.0
        else:
            self.max_timeline_offset = content_duration - visible_duration
        
        # Ensure current offset is within bounds
        self.timeline_offset = max(self.min_timeline_offset, 
                                 min(self.max_timeline_offset, self.timeline_offset))
    
    def _get_content_duration(self) -> float:
        """Get the total duration of all content in the timeline"""
        max_duration = 0.0
        
        for track in self.daw.tracks:
            for clip in track.clips:
                clip_end = clip.start_time + clip.duration
                max_duration = max(max_duration, clip_end)
        
        return max_duration
    
    def get_timeline_info(self) -> dict:
        """Get current timeline information"""
        return {
            'pixels_per_second': self.pixels_per_second,
            'timeline_offset': self.timeline_offset,
            'min_offset': self.min_timeline_offset,
            'max_offset': self.max_timeline_offset,
            'snap_enabled': self.snap_enabled,
            'snap_interval': self.snap_interval,
            'visible_start': self.timeline_offset,
            'visible_end': self.timeline_offset + (self.timeline_width / self.pixels_per_second),
            'content_duration': self._get_content_duration()
        }
