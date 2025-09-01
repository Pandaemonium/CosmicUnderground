"""
DAW Clip editing and manipulation methods
Extracted from cosmic_daw.py for better modularity
"""

import pygame
from typing import List, Optional, Tuple
from ..core.debug import debug_info, debug_error, debug_warning
from .daw_clip import DAWClip
from .daw_track import DAWTrack
from .daw_actions import (
    AddClipAction, DeleteClipsAction, MoveClipAction, 
    DuplicateClipsAction, MergeClipsAction, SliceClipsAction
)


class DAWClipOperations:
    """Handles clip editing and manipulation operations for the DAW"""
    
    def __init__(self, daw):
        self.daw = daw
        
        # Clipboard
        self.clipboard_clips = []
        
        # Selection
        self.selected_clips = []
        
        # Undo/Redo
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_steps = 50
    
    def add_track_from_inventory(self, alien_track, track_index: int) -> bool:
        """Add a track from inventory to the specified track index"""
        try:
            if not alien_track:
                debug_error("Cannot add None track to inventory")
                return False
            
            if track_index < 0 or track_index >= len(self.daw.tracks):
                debug_error(f"Invalid track index: {track_index}")
                return False
            
            # Create new clip
            new_clip = DAWClip(
                alien_track=alien_track,
                start_time=0.0,
                track_index=track_index,
                daw=self.daw,
                duration=alien_track.duration,
                name=alien_track.name
            )
            
            # Add to track
            self.daw.tracks[track_index].add_clip(new_clip)
            
            # Auto-arrange to prevent overlaps
            self._auto_arrange_clips()
            
            # Create undo action
            action = AddClipAction(self.daw, new_clip, track_index)
            self._add_undo_action(action)
            
            print(f"🎵 Added {alien_track.name} to {self.daw.tracks[track_index].name} track at {new_clip.start_time:.2f}s")
            return True
            
        except Exception as e:
            debug_error(f"Error adding track from inventory: {e}")
            return False
    
    def duplicate_selected_clips(self):
        """Duplicate all selected clips"""
        if not self.selected_clips:
            debug_info("No clips selected for duplication")
            return
        
        try:
            # Create undo action
            action = DuplicateClipsAction(self.daw, self.selected_clips.copy())
            self._add_undo_action(action)
            
            # Duplicate each selected clip
            for clip in self.selected_clips:
                duplicated_clip = clip.duplicate()
                if duplicated_clip:
                    # Add to the same track
                    track = self.daw.tracks[clip.track_index]
                    track.add_clip(duplicated_clip)
            
            # Auto-arrange to prevent overlaps
            self._auto_arrange_clips()
            
            print(f"✅ Duplicated {len(self.selected_clips)} clips")
            
        except Exception as e:
            debug_error(f"Error duplicating clips: {e}")
    
    def delete_selected_clips(self):
        """Delete all selected clips"""
        if not self.selected_clips:
            debug_info("No clips selected for deletion")
            return
        
        try:
            # Create undo action
            action = DeleteClipsAction(self.daw, self.selected_clips.copy())
            self._add_undo_action(action)
            
            # Delete each selected clip
            for clip in self.selected_clips:
                clip.delete()
            
            # Clear selection
            self.selected_clips.clear()
            
            print(f"🗑️ Deleted {len(self.selected_clips)} clips")
            
        except Exception as e:
            debug_error(f"Error deleting clips: {e}")
    
    def merge_selected_clips(self):
        """Merge selected clips into a single clip"""
        if len(self.selected_clips) < 2:
            debug_info("Need at least 2 clips to merge")
            return
        
        try:
            # Create undo action
            action = MergeClipsAction(self.daw, self.selected_clips.copy())
            self._add_undo_action(action)
            
            # Sort clips by start time
            sorted_clips = sorted(self.selected_clips, key=lambda c: c.start_time)
            
            # Calculate merged clip properties
            first_clip = sorted_clips[0]
            last_clip = sorted_clips[-1]
            merged_start = first_clip.start_time
            merged_end = last_clip.start_time + last_clip.duration
            merged_duration = merged_end - merged_start
            
            # Create merged clip
            merged_clip = DAWClip(
                alien_track=first_clip.alien_track,  # Use first clip's track as base
                start_time=merged_start,
                track_index=first_clip.track_index,
                daw=self.daw,
                duration=merged_duration,
                name=f"Merged_{len(sorted_clips)}_clips"
            )
            
            # Remove original clips
            for clip in sorted_clips:
                clip.delete()
            
            # Add merged clip
            track = self.daw.tracks[merged_clip.track_index]
            track.add_clip(merged_clip)
            
            # Store merged clip in action for undo
            action.merged_clip = merged_clip
            
            # Clear selection and select merged clip
            self.selected_clips.clear()
            merged_clip.is_selected = True
            self.selected_clips.append(merged_clip)
            
            print(f"🔗 Merged {len(sorted_clips)} clips into single clip")
            
        except Exception as e:
            debug_error(f"Error merging clips: {e}")
    
    def slice_clips_at_playhead(self):
        """Slice selected clips at the current playhead position"""
        if not self.selected_clips:
            debug_info("No clips selected for slicing")
            return
        
        playhead_pos = self.daw.playhead_position
        
        try:
            # Create undo action
            action = SliceClipsAction(self.daw, self.selected_clips.copy())
            self._add_undo_action(action)
            
            clips_to_slice = self.selected_clips.copy()
            
            for clip in clips_to_slice:
                # Check if playhead is within this clip
                if clip.start_time <= playhead_pos <= clip.start_time + clip.duration:
                    # Calculate slice point relative to clip start
                    slice_point = playhead_pos - clip.start_time
                    
                    if slice_point > 0 and slice_point < clip.duration:
                        # Create first slice (before playhead)
                        first_slice = DAWClip(
                            alien_track=clip.alien_track,
                            start_time=clip.start_time,
                            track_index=clip.track_index,
                            daw=self.daw,
                            duration=slice_point,
                            name=f"{clip.name}_slice1"
                        )
                        
                        # Create second slice (after playhead)
                        second_slice = DAWClip(
                            alien_track=clip.alien_track,
                            start_time=playhead_pos,
                            track_index=clip.track_index,
                            daw=self.daw,
                            duration=clip.duration - slice_point,
                            name=f"{clip.name}_slice2"
                        )
                        
                        # Remove original clip
                        clip.delete()
                        
                        # Add slices
                        track = self.daw.tracks[clip.track_index]
                        track.add_clip(first_slice)
                        track.add_clip(second_slice)
                        
                        print(f"✂️ Sliced {clip.name} at {playhead_pos:.2f}s")
            
            # Capture created slices for undo
            action.capture_created_slices()
            
            # Clear selection
            self.selected_clips.clear()
            
        except Exception as e:
            debug_error(f"Error slicing clips: {e}")
    
    def copy_selected_clips(self):
        """Copy selected clips to clipboard"""
        if not self.selected_clips:
            debug_info("No clips selected for copying")
            return
        
        try:
            # Clear clipboard
            self.clipboard_clips.clear()
            
            # Copy clips to clipboard
            for clip in self.selected_clips:
                # Create a copy of the clip for clipboard
                clipboard_clip = DAWClip(
                    alien_track=clip.alien_track,
                    start_time=clip.start_time,
                    track_index=clip.track_index,
                    daw=self.daw,
                    duration=clip.duration,
                    name=clip.name
                )
                self.clipboard_clips.append(clipboard_clip)
            
            print(f"🎵 Copied {len(self.selected_clips)} clips to clipboard")
            
        except Exception as e:
            debug_error(f"Error copying clips: {e}")
    
    def paste_clips_from_clipboard(self, paste_time: float, target_track: int = None):
        """Paste clips from clipboard at specified time"""
        if not self.clipboard_clips:
            debug_info("Clipboard is empty")
            return
        
        try:
            # Create undo action
            action = DuplicateClipsAction(self.daw, self.clipboard_clips.copy())
            self._add_undo_action(action)
            
            # Paste each clipboard clip
            for i, clipboard_clip in enumerate(self.clipboard_clips):
                # Determine target track
                if target_track is None:
                    target_track = clipboard_clip.track_index
                
                # Ensure valid track index
                if target_track < 0 or target_track >= len(self.daw.tracks):
                    target_track = 0
                
                # Create new clip
                new_clip = DAWClip(
                    alien_track=clipboard_clip.alien_track,
                    start_time=paste_time + (i * 0.1),  # Offset each clip slightly
                    track_index=target_track,
                    daw=self.daw,
                    duration=clipboard_clip.duration,
                    name=f"{clipboard_clip.name}_pasted"
                )
                
                # Add to track
                self.daw.tracks[target_track].add_clip(new_clip)
            
            # Auto-arrange to prevent overlaps
            self._auto_arrange_clips()
            
            print(f"📋 Pasted {len(self.clipboard_clips)} clips from clipboard")
            
        except Exception as e:
            debug_error(f"Error pasting clips: {e}")
    
    def select_all_clips(self):
        """Select all clips in the DAW"""
        self.selected_clips.clear()
        
        for track in self.daw.tracks:
            for clip in track.clips:
                clip.is_selected = True
                self.selected_clips.append(clip)
        
        debug_info(f"Selected all {len(self.selected_clips)} clips")
    
    def clear_selection(self):
        """Clear current clip selection"""
        for clip in self.selected_clips:
            clip.is_selected = False
        
        self.selected_clips.clear()
        debug_info("Cleared clip selection")
    
    def select_clips_in_region(self, start_time: float, end_time: float, track_start: int = 0, track_end: int = None):
        """Select clips within a time and track region"""
        if track_end is None:
            track_end = len(self.daw.tracks)
        
        self.clear_selection()
        
        for track_index in range(track_start, track_end):
            if track_index >= len(self.daw.tracks):
                break
            
            track = self.daw.tracks[track_index]
            for clip in track.clips:
                # Check if clip overlaps with time region
                clip_start = clip.start_time
                clip_end = clip.start_time + clip.duration
                
                if (clip_start < end_time and clip_end > start_time):
                    clip.is_selected = True
                    self.selected_clips.append(clip)
        
        debug_info(f"Selected {len(self.selected_clips)} clips in region")
    
    def _auto_arrange_clips(self):
        """Automatically arrange clips to prevent overlaps"""
        for track in self.daw.tracks:
            if len(track.clips) <= 1:
                continue
            
            # Sort clips by start time
            sorted_clips = sorted(track.clips, key=lambda c: c.start_time)
            
            # Arrange clips sequentially
            current_time = 0.0
            for clip in sorted_clips:
                clip.start_time = current_time
                current_time += clip.duration + 0.1  # Small gap between clips
    
    def _add_undo_action(self, action):
        """Add an action to the undo stack"""
        self.undo_stack.append(action)
        
        # Limit undo stack size
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)
        
        # Clear redo stack when new action is added
        self.redo_stack.clear()
    
    def undo(self):
        """Undo the last action"""
        if not self.undo_stack:
            debug_info("Nothing to undo")
            return
        
        try:
            # Get last action
            action = self.undo_stack.pop()
            
            # Execute undo
            action.undo()
            
            # Add to redo stack
            self.redo_stack.append(action)
            
            debug_info("Undo completed")
            
        except Exception as e:
            debug_error(f"Error during undo: {e}")
    
    def redo(self):
        """Redo the last undone action"""
        if not self.redo_stack:
            debug_info("Nothing to redo")
            return
        
        try:
            # Get last undone action
            action = self.redo_stack.pop()
            
            # Execute redo
            action.redo()
            
            # Add back to undo stack
            self.undo_stack.append(action)
            
            debug_info("Redo completed")
            
        except Exception as e:
            debug_error(f"Error during redo: {e}")
    
    def can_undo(self) -> bool:
        """Check if undo is available"""
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available"""
        return len(self.redo_stack) > 0
    
    def get_selection_info(self) -> dict:
        """Get information about current selection"""
        return {
            'selected_count': len(self.selected_clips),
            'selected_clips': [clip.name for clip in self.selected_clips],
            'can_undo': self.can_undo(),
            'can_redo': self.can_redo(),
            'clipboard_count': len(self.clipboard_clips)
        }
