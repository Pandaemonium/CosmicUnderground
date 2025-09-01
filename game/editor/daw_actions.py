"""
DAW Action classes for undo/redo functionality
Extracted from cosmic_daw.py for better modularity
"""

from typing import List, TYPE_CHECKING
from ..core.debug import debug_info, debug_warning

if TYPE_CHECKING:
    from .cosmic_daw import CosmicDAW
    from .daw_clip import DAWClip


class DAWAction:
    """Base class for DAW actions that support undo/redo"""
    
    def __init__(self, daw: 'CosmicDAW'):
        self.daw = daw
    
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
    """Action for adding a clip to a track"""
    
    def __init__(self, daw: 'CosmicDAW', clip: 'DAWClip', track_index: int):
        super().__init__(daw)
        self.clip = clip
        self.track_index = track_index
    
    def execute(self):
        """Add the clip (already done in the calling method)"""
        pass  # Adding is handled by the calling method
    
    def undo(self):
        """Remove the clip"""
        if self.track_index < len(self.daw.tracks):
            track = self.daw.tracks[self.track_index]
            if self.clip in track.clips:
                track.remove_clip(self.clip)
                debug_info(f"Undo: Removed clip {self.clip.name} from track {track.name}")
    
    def redo(self):
        """Re-add the clip"""
        if self.track_index < len(self.daw.tracks):
            track = self.daw.tracks[self.track_index]
            track.add_clip(self.clip)
            debug_info(f"Redo: Added clip {self.clip.name} to track {track.name}")


class DeleteClipsAction(DAWAction):
    """Action for deleting multiple clips from the DAW"""
    
    def __init__(self, daw: 'CosmicDAW', clips: List['DAWClip']):
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


class MoveClipAction(DAWAction):
    """Action for moving a clip to a new position"""
    
    def __init__(self, daw: 'CosmicDAW', clip: 'DAWClip', 
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
                if other_clip != clip:  # Don't store the clip we moved
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


class DuplicateClipsAction(DAWAction):
    """Action for duplicating clips"""
    
    def __init__(self, daw: 'CosmicDAW', original_clips: List['DAWClip']):
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
    
    def __init__(self, daw: 'CosmicDAW', original_clips: List['DAWClip']):
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
            from .daw_clip import DAWClip
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


class SliceClipsAction(DAWAction):
    """Action for slicing clips at the playhead"""
    
    def __init__(self, daw: 'CosmicDAW', original_clips: List['DAWClip']):
        super().__init__(daw)
        self.original_clips = original_clips
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
    
    def __init__(self, daw: 'CosmicDAW', clipboard_clips: List['DAWClip'], paste_time: float):
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
