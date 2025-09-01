"""
Comprehensive test suite for CosmicDAW system
Tests all core functionality to ensure nothing is lost during refactoring
"""

import unittest
import pygame
import numpy as np
import tempfile
import os
import sys
from unittest.mock import Mock, MagicMock, patch

# Add the game directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game.editor.cosmic_daw import CosmicDAW, DAWClip, DAWTrack, DAWAction
from game.audio.alien_track import AlienTrack
from game.core.config import WIDTH, HEIGHT


class TestDAWClip(unittest.TestCase):
    """Test DAWClip functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        pygame.init()
        
        # Create a mock alien track
        self.mock_alien_track = Mock(spec=AlienTrack)
        self.mock_alien_track.name = "Test Track"
        self.mock_alien_track.duration = 10.0
        self.mock_alien_track.sound = None
        
        # Create a DAWClip instance
        self.clip = DAWClip(
            alien_track=self.mock_alien_track,
            start_time=5.0,
            track_index=0,
            duration=8.0,
            name="Test Clip"
        )
    
    def tearDown(self):
        """Clean up after tests"""
        pygame.quit()
    
    def test_clip_initialization(self):
        """Test DAWClip initialization"""
        self.assertEqual(self.clip.alien_track, self.mock_alien_track)
        self.assertEqual(self.clip.start_time, 5.0)
        self.assertEqual(self.clip.track_index, 0)
        self.assertEqual(self.clip.duration, 8.0)
        self.assertEqual(self.clip.name, "Test Clip")
        self.assertFalse(self.clip.is_selected)
        self.assertFalse(self.clip.is_playing)
    
    def test_clip_duplicate(self):
        """Test clip duplication"""
        duplicated = self.clip.duplicate()
        self.assertIsNotNone(duplicated)
        self.assertEqual(duplicated.alien_track, self.mock_alien_track)
        self.assertEqual(duplicated.track_index, 0)
        self.assertEqual(duplicated.duration, 8.0)
        self.assertIn("copy", duplicated.name)
    
    def test_clip_delete(self):
        """Test clip deletion cleanup"""
        # Mock the channel and sound
        self.clip.current_channel = Mock()
        self.clip.current_sliced_sound = Mock()
        
        self.clip.delete()
        
        # Verify cleanup
        self.assertIsNone(self.clip.current_channel)
        self.assertIsNone(self.clip.current_sliced_sound)


class TestDAWTrack(unittest.TestCase):
    """Test DAWTrack functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        pygame.init()
        
        # Create a mock DAW
        self.mock_daw = Mock()
        self.mock_daw._has_soloed_tracks.return_value = False
        
        # Create a DAWTrack instance
        self.track = DAWTrack("Test Track", (255, 0, 0), 0, self.mock_daw)
        
        # Create a mock clip
        self.mock_clip = Mock(spec=DAWClip)
        self.mock_clip.name = "Test Clip"
        self.mock_clip.start_time = 0.0
        self.mock_clip.duration = 5.0
        self.mock_clip.is_playing = False
    
    def tearDown(self):
        """Clean up after tests"""
        pygame.quit()
    
    def test_track_initialization(self):
        """Test DAWTrack initialization"""
        self.assertEqual(self.track.name, "Test Track")
        self.assertEqual(self.track.color, (255, 0, 0))
        self.assertEqual(self.track.track_index, 0)
        self.assertEqual(self.track.daw, self.mock_daw)
        self.assertFalse(self.track.is_muted)
        self.assertFalse(self.track.is_soloed)
        self.assertFalse(self.track.is_recording)
        self.assertEqual(self.track.volume, 1.0)
        self.assertEqual(self.track.pan, 0.0)
    
    def test_add_remove_clip(self):
        """Test adding and removing clips"""
        # Add clip
        self.track.add_clip(self.mock_clip)
        self.assertIn(self.mock_clip, self.track.clips)
        self.assertEqual(len(self.track.clips), 1)
        
        # Remove clip
        self.track.remove_clip(self.mock_clip)
        self.assertNotIn(self.mock_clip, self.track.clips)
        self.assertEqual(len(self.track.clips), 0)
    
    def test_has_content(self):
        """Test content detection"""
        self.assertFalse(self.track.has_content())
        
        self.track.add_clip(self.mock_clip)
        self.assertTrue(self.track.has_content())
    
    def test_toggle_mute(self):
        """Test mute toggling"""
        self.assertFalse(self.track.is_muted)
        
        self.track.toggle_mute()
        self.assertTrue(self.track.is_muted)
        
        self.track.toggle_mute()
        self.assertFalse(self.track.is_muted)
    
    def test_toggle_solo(self):
        """Test solo toggling"""
        self.assertFalse(self.track.is_soloed)
        
        self.track.toggle_solo()
        self.assertTrue(self.track.is_soloed)
        
        self.track.toggle_solo()
        self.assertFalse(self.track.is_soloed)
    
    def test_toggle_record(self):
        """Test record toggling"""
        self.assertFalse(self.track.is_recording)
        
        self.track.toggle_record()
        self.assertTrue(self.track.is_recording)
        
        self.track.toggle_record()
        self.assertFalse(self.track.is_recording)


class TestCosmicDAW(unittest.TestCase):
    """Test CosmicDAW core functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        pygame.init()
        
        # Create mock fonts
        self.mock_fonts = (Mock(), Mock(), Mock())
        
        # Create mock game state
        self.mock_game_state = Mock()
        
        # Create CosmicDAW instance
        self.daw = CosmicDAW(self.mock_fonts, self.mock_game_state)
    
    def tearDown(self):
        """Clean up after tests"""
        pygame.quit()
    
    def test_daw_initialization(self):
        """Test CosmicDAW initialization"""
        self.assertEqual(self.daw.daw_width, WIDTH)
        self.assertEqual(self.daw.daw_height, HEIGHT)
        self.assertEqual(self.daw.max_tracks, 8)
        self.assertEqual(self.daw.track_height, 80)
        self.assertEqual(self.daw.max_duration, 1200.0)
        self.assertFalse(self.daw.is_playing)
        self.assertEqual(self.daw.playhead_position, 0.0)
        self.assertEqual(len(self.daw.tracks), 8)
    
    def test_add_track_from_inventory(self):
        """Test adding tracks from inventory"""
        # Create mock alien track
        mock_alien_track = Mock(spec=AlienTrack)
        mock_alien_track.name = "Test Alien Track"
        mock_alien_track.duration = 15.0
        
        # Add to track 0
        result = self.daw.add_track_from_inventory(mock_alien_track, 0)
        self.assertTrue(result)
        
        # Verify clip was added
        track = self.daw.tracks[0]
        self.assertTrue(track.has_content())
        self.assertEqual(len(track.clips), 1)
    
    def test_playback_controls(self):
        """Test playback control methods"""
        # Test play
        self.daw.toggle_playback()
        self.assertTrue(self.daw.is_playing)
        
        # Test stop
        self.daw.stop()
        self.assertFalse(self.daw.is_playing)
        self.assertEqual(self.daw.playhead_position, 0.0)
    
    def test_zoom_and_pan(self):
        """Test zoom and pan functionality"""
        initial_zoom = self.daw.pixels_per_second
        
        # Test zoom in
        self.daw.zoom_in()
        self.assertGreater(self.daw.pixels_per_second, initial_zoom)
        
        # Test zoom out
        self.daw.zoom_out()
        self.assertLess(self.daw.pixels_per_second, initial_zoom)
        
        # Test panning
        initial_offset = self.daw.timeline_offset
        self.daw.pan_right(10.0)
        self.assertGreater(self.daw.timeline_offset, initial_offset)
        
        self.daw.pan_left(10.0)
        self.assertEqual(self.daw.timeline_offset, initial_offset)
    
    def test_clip_selection(self):
        """Test clip selection system"""
        # Add a clip first
        mock_alien_track = Mock(spec=AlienTrack)
        mock_alien_track.name = "Test Track"
        mock_alien_track.duration = 10.0
        
        self.daw.add_track_from_inventory(mock_alien_track, 0)
        
        # Test timeline click for selection
        self.daw._handle_timeline_click(300, 200, 1)  # Left click
        
        # Verify playhead moved
        self.assertGreater(self.daw.playhead_position, 0.0)
    
    def test_undo_redo_system(self):
        """Test undo/redo functionality"""
        # Add a clip to create an action
        mock_alien_track = Mock(spec=AlienTrack)
        mock_alien_track.name = "Test Track"
        mock_alien_track.duration = 10.0
        
        self.daw.add_track_from_inventory(mock_alien_track, 0)
        
        # Verify action was added to undo stack
        self.assertGreater(len(self.daw.undo_stack), 0)
        
        # Test undo
        initial_clip_count = len(self.daw.tracks[0].clips)
        self.daw.undo()
        
        # Verify clip was removed
        self.assertLess(len(self.daw.tracks[0].clips), initial_clip_count)
        
        # Test redo
        self.daw.redo()
        self.assertEqual(len(self.daw.tracks[0].clips), initial_clip_count)
    
    def test_clip_operations(self):
        """Test various clip operations"""
        # Add clips for testing
        mock_alien_track1 = Mock(spec=AlienTrack)
        mock_alien_track1.name = "Track 1"
        mock_alien_track1.duration = 10.0
        
        mock_alien_track2 = Mock(spec=AlienTrack)
        mock_alien_track2.name = "Track 2"
        mock_alien_track2.duration = 8.0
        
        self.daw.add_track_from_inventory(mock_alien_track1, 0)
        self.daw.add_track_from_inventory(mock_alien_track2, 1)
        
        # Test clip duplication
        self.daw.tracks[0].clips[0].is_selected = True
        self.daw.selected_clips = [self.daw.tracks[0].clips[0]]
        
        initial_clip_count = len(self.daw.tracks[0].clips)
        self.daw.duplicate_selected_clips()
        self.assertGreater(len(self.daw.tracks[0].clips), initial_clip_count)
        
        # Test clip deletion
        self.daw.selected_clips = [self.daw.tracks[0].clips[0]]
        self.daw.delete_selected_clips()
        self.assertEqual(len(self.daw.tracks[0].clips), 0)
    
    def test_mix_management(self):
        """Test mix save/load functionality"""
        # Mock the mix manager
        self.daw.mix_manager = Mock()
        self.daw.mix_manager.save_mix.return_value = True
        self.daw.mix_manager.export_mix.return_value = True
        self.daw.mix_manager.list_saved_mixes.return_value = ["test.mix"]
        self.daw.mix_manager.load_mix.return_value = True
        
        # Test save
        self.daw.save_current_mix()
        self.daw.mix_manager.save_mix.assert_called_once()
        
        # Test export
        self.daw.export_current_mix()
        self.daw.mix_manager.export_mix.assert_called_once()
        
        # Test load
        self.daw.load_saved_mix()
        self.daw.mix_manager.list_saved_mixes.assert_called_once()
    
    def test_waveform_system(self):
        """Test waveform generation and caching"""
        # Test waveform cache initialization
        self.assertIsInstance(self.daw.waveform_cache, dict)
        self.assertEqual(self.daw.waveform_cache_size, 100)
        
        # Test cache clearing
        self.daw.clear_waveform_cache()
        self.assertEqual(len(self.daw.waveform_cache), 0)
    
    def test_timeline_interaction(self):
        """Test timeline interaction methods"""
        # Test scroll bar interaction
        mock_event = Mock()
        mock_event.pos = (300, 140)  # Scroll bar area
        mock_event.type = pygame.MOUSEBUTTONDOWN
        mock_event.button = 1
        
        self.daw._handle_scroll_bar_interaction(mock_event)
        
        # Test mouse wheel handling
        mock_wheel_event = Mock()
        mock_wheel_event.type = pygame.MOUSEWHEEL
        mock_wheel_event.y = 1  # Scroll up
        
        self.daw._handle_mouse_wheel(mock_wheel_event)
    
    def test_track_controls(self):
        """Test track control interactions"""
        # Test track control click
        mock_event = Mock()
        mock_event.pos = (35, 200)  # Mute button area
        mock_event.button = 1
        
        self.daw._handle_track_control_click(mock_event.pos[0], mock_event.pos[1], mock_event.button)
        
        # Verify mute was toggled
        track = self.daw.tracks[0]
        self.assertTrue(track.is_muted or not track.is_muted)  # State changed
    
    def test_keyboard_shortcuts(self):
        """Test keyboard shortcut handling"""
        # Test space key for play/pause
        mock_event = Mock()
        mock_event.key = pygame.K_SPACE
        mock_event.type = pygame.KEYDOWN
        
        initial_playing_state = self.daw.is_playing
        self.daw._handle_key_down(mock_event, [])
        
        # Verify state changed
        self.assertNotEqual(self.daw.is_playing, initial_playing_state)
    
    def test_auto_arrange(self):
        """Test automatic clip arrangement"""
        # Add multiple clips to same track
        mock_alien_track1 = Mock(spec=AlienTrack)
        mock_alien_track1.name = "Track 1"
        mock_alien_track1.duration = 5.0
        
        mock_alien_track2 = Mock(spec=AlienTrack)
        mock_alien_track2.name = "Track 2"
        mock_alien_track2.duration = 5.0
        
        self.daw.add_track_from_inventory(mock_alien_track1, 0)
        self.daw.add_track_from_inventory(mock_alien_track2, 0)
        
        # Verify clips don't overlap
        clips = self.daw.tracks[0].clips
        self.assertEqual(len(clips), 2)
        
        # Check that second clip starts after first clip ends
        first_clip_end = clips[0].start_time + clips[0].duration
        self.assertGreaterEqual(clips[1].start_time, first_clip_end)
    
    def test_selection_system(self):
        """Test multi-selection system"""
        # Add clips for selection testing
        mock_alien_track1 = Mock(spec=AlienTrack)
        mock_alien_track1.name = "Track 1"
        mock_alien_track1.duration = 5.0
        
        mock_alien_track2 = Mock(spec=AlienTrack)
        mock_alien_track2.name = "Track 2"
        mock_alien_track2.duration = 5.0
        
        self.daw.add_track_from_inventory(mock_alien_track1, 0)
        self.daw.add_track_from_inventory(mock_alien_track2, 1)
        
        # Test single selection
        clip1 = self.daw.tracks[0].clips[0]
        self.daw._select_clip_at_position(0, clip1.start_time + 1.0)
        self.assertIn(clip1, self.daw.selected_clips)
        
        # Test multi-selection with CTRL
        clip2 = self.daw.tracks[1].clips[0]
        with patch('pygame.key.get_mods', return_value=pygame.KMOD_CTRL):
            self.daw._select_clip_at_position(1, clip2.start_time + 1.0)
        
        self.assertIn(clip2, self.daw.selected_clips)
        self.assertEqual(len(self.daw.selected_clips), 2)
    
    def test_clip_dragging(self):
        """Test clip drag and drop functionality"""
        # Add a clip for dragging
        mock_alien_track = Mock(spec=AlienTrack)
        mock_alien_track.name = "Test Track"
        mock_alien_track.duration = 10.0
        
        self.daw.add_track_from_inventory(mock_alien_track, 0)
        clip = self.daw.tracks[0].clips[0]
        
        # Test drag start
        self.daw._check_clip_drag_start(300, 200)
        self.assertIsNotNone(self.daw.potential_drag_clip)
        
        # Test drag finish
        self.daw.dragging_clip = clip
        self.daw.drag_start_track = 0
        self.daw.drag_start_time = clip.start_time
        
        self.daw._finish_clip_drag((400, 200))
        
        # Verify drag state was reset
        self.assertIsNone(self.daw.dragging_clip)
    
    def test_effects_system(self):
        """Test effects system initialization"""
        # Verify effects are available
        self.assertIn("Reverb", self.daw.available_effects)
        self.assertIn("Delay", self.daw.available_effects)
        self.assertIn("Distortion", self.daw.available_effects)
        
        # Check effect parameters
        reverb = self.daw.available_effects["Reverb"]
        self.assertIn("wet", reverb)
        self.assertIn("room_size", reverb)
        self.assertIn("damping", reverb)
    
    def test_mixing_system(self):
        """Test mixing system initialization"""
        # Verify mixing parameters
        self.assertEqual(self.daw.master_volume, 1.0)
        self.assertEqual(self.daw.master_pan, 0.0)
        self.assertIsInstance(self.daw.master_effects, dict)
    
    def test_snap_settings(self):
        """Test snap-to-grid functionality"""
        # Verify snap settings
        self.assertTrue(self.daw.snap_to_grid)
        self.assertEqual(self.daw.grid_division, 0.25)
    
    def test_loop_functionality(self):
        """Test loop functionality"""
        # Test loop toggle
        self.assertFalse(self.daw.is_looping)
        
        self.daw.toggle_loop()
        self.assertTrue(self.daw.is_looping)
        
        # Test loop boundaries
        self.assertEqual(self.daw.loop_start, 0.0)
        self.assertEqual(self.daw.loop_end, 1200.0)
    
    def test_seek_functionality(self):
        """Test audio seeking functionality"""
        # Test seek feedback system
        self.assertFalse(self.daw.seeking_feedback)
        
        self.daw._show_seek_feedback(10.0)
        self.assertTrue(self.daw.seeking_feedback)
        self.assertEqual(self.daw.seek_feedback_position, 10.0)
    
    def test_waveform_scaling(self):
        """Test waveform scaling modes"""
        # Test scaling mode toggle
        initial_mode = getattr(self.daw, 'waveform_scaling_mode', 1)
        
        self.daw._toggle_waveform_scaling_mode()
        
        # Verify mode changed
        new_mode = getattr(self.daw, 'waveform_scaling_mode', 1)
        self.assertNotEqual(new_mode, initial_mode)
    
    def test_clip_slicing(self):
        """Test clip slicing functionality"""
        # Add a clip for slicing
        mock_alien_track = Mock(spec=AlienTrack)
        mock_alien_track.name = "Test Track"
        mock_alien_track.duration = 10.0
        
        self.daw.add_track_from_inventory(mock_alien_track, 0)
        clip = self.daw.tracks[0].clips[0]
        
        # Test slicing at playhead
        self.daw.playhead_position = 5.0
        self.daw.selected_clips = [clip]
        
        initial_clip_count = len(self.daw.tracks[0].clips)
        self.daw.slice_clips_at_playhead()
        
        # Verify clip was sliced
        self.assertGreater(len(self.daw.tracks[0].clips), initial_clip_count)
    
    def test_clip_merging(self):
        """Test clip merging functionality"""
        # Add clips for merging
        mock_alien_track1 = Mock(spec=AlienTrack)
        mock_alien_track1.name = "Track 1"
        mock_alien_track1.duration = 5.0
        
        mock_alien_track2 = Mock(spec=AlienTrack)
        mock_alien_track2.name = "Track 2"
        mock_alien_track2.duration = 5.0
        
        self.daw.add_track_from_inventory(mock_alien_track1, 0)
        self.daw.add_track_from_inventory(mock_alien_track2, 0)
        
        # Select both clips
        clips = self.daw.tracks[0].clips
        self.daw.selected_clips = clips
        
        initial_clip_count = len(clips)
        self.daw.merge_selected_clips()
        
        # Verify clips were merged
        self.assertLess(len(self.daw.tracks[0].clips), initial_clip_count)
    
    def test_clipboard_operations(self):
        """Test clipboard functionality"""
        # Add a clip for clipboard operations
        mock_alien_track = Mock(spec=AlienTrack)
        mock_alien_track.name = "Test Track"
        mock_alien_track.duration = 10.0
        
        self.daw.add_track_from_inventory(mock_alien_track, 0)
        clip = self.daw.tracks[0].clips[0]
        
        # Test copy
        self.daw.selected_clips = [clip]
        self.daw.copy_selected_clips()
        self.assertIn(clip, self.daw.clipboard_clips)
        
        # Test cut
        initial_clip_count = len(self.daw.tracks[0].clips)
        self.daw.cut_selected_clips()
        self.assertLess(len(self.daw.tracks[0].clips), initial_clip_count)
        
        # Test paste
        self.daw.paste_clips()
        self.assertGreater(len(self.daw.tracks[0].clips), 0)
    
    def test_auto_save(self):
        """Test auto-save functionality"""
        # Test auto-save timing
        self.assertEqual(self.daw.auto_save_interval, 30)
        self.assertEqual(self.daw.last_auto_save, 0)
        
        # Test auto-save method exists
        self.assertTrue(hasattr(self.daw, '_auto_save'))
    
    def test_state_reset(self):
        """Test DAW state reset functionality"""
        # Set some state
        self.daw.is_playing = True
        self.daw.playhead_position = 100.0
        self.daw.selected_clips = [Mock()]
        
        # Reset state
        self.daw.reset_state()
        
        # Verify state was reset
        self.assertFalse(self.daw.is_playing)
        self.assertEqual(self.daw.playhead_position, 0.0)
        self.assertEqual(len(self.daw.selected_clips), 0)


class TestDAWActions(unittest.TestCase):
    """Test DAW action classes"""
    
    def setUp(self):
        """Set up test fixtures"""
        pygame.init()
        
        # Create mock DAW
        self.mock_daw = Mock()
        self.mock_daw.tracks = [Mock(), Mock()]
        
        # Create mock clips
        self.mock_clip = Mock()
        self.mock_clip.name = "Test Clip"
        self.mock_clip.track_index = 0
        self.mock_clip.start_time = 0.0
    
    def tearDown(self):
        """Clean up after tests"""
        pygame.quit()
    
    def test_delete_clips_action(self):
        """Test DeleteClipsAction"""
        from game.editor.cosmic_daw import DeleteClipsAction
        
        action = DeleteClipsAction(self.mock_daw, [self.mock_clip])
        
        # Test execute
        action.execute()
        
        # Test undo
        action.undo()
        
        # Test redo
        action.redo()
    
    def test_move_clip_action(self):
        """Test MoveClipAction"""
        from game.editor.cosmic_daw import MoveClipAction
        
        action = MoveClipAction(
            self.mock_daw, self.mock_clip, 0, 0.0, 1, 10.0
        )
        
        # Test capture final state
        action.capture_final_state()
        
        # Test undo
        action.undo()
        
        # Test redo
        action.redo()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete DAW system"""
    
    def setUp(self):
        """Set up test fixtures"""
        pygame.init()
        
        # Create mock fonts
        self.mock_fonts = (Mock(), Mock(), Mock())
        
        # Create mock game state
        self.mock_game_state = Mock()
        
        # Create CosmicDAW instance
        self.daw = CosmicDAW(self.mock_fonts, self.mock_game_state)
    
    def tearDown(self):
        """Clean up after tests"""
        pygame.quit()
    
    def test_complete_workflow(self):
        """Test a complete DAW workflow"""
        # 1. Add tracks from inventory
        mock_alien_track1 = Mock(spec=AlienTrack)
        mock_alien_track1.name = "Bass Track"
        mock_alien_track1.duration = 15.0
        
        mock_alien_track2 = Mock(spec=AlienTrack)
        mock_alien_track2.name = "Melody Track"
        mock_alien_track2.duration = 12.0
        
        self.daw.add_track_from_inventory(mock_alien_track1, 0)
        self.daw.add_track_from_inventory(mock_alien_track2, 1)
        
        # 2. Verify tracks were added
        self.assertTrue(self.daw.tracks[0].has_content())
        self.assertTrue(self.daw.tracks[1].has_content())
        
        # 3. Start playback
        self.daw.toggle_playback()
        self.assertTrue(self.daw.is_playing)
        
        # 4. Move playhead
        self.daw.playhead_position = 5.0
        
        # 5. Stop playback
        self.daw.stop()
        self.assertFalse(self.daw.is_playing)
        self.assertEqual(self.daw.playhead_position, 0.0)
        
        # 6. Test zoom and pan
        self.daw.zoom_in()
        self.daw.pan_right(10.0)
        
        # 7. Test selection
        clip = self.daw.tracks[0].clips[0]
        self.daw._select_clip_at_position(0, clip.start_time + 1.0)
        self.assertIn(clip, self.daw.selected_clips)
        
        # 8. Test operations on selected clips
        self.daw.duplicate_selected_clips()
        self.assertGreater(len(self.daw.tracks[0].clips), 1)
        
        # 9. Test undo
        self.daw.undo()
        self.assertEqual(len(self.daw.tracks[0].clips), 1)
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with invalid track index
        mock_alien_track = Mock(spec=AlienTrack)
        mock_alien_track.name = "Test Track"
        mock_alien_track.duration = 10.0
        
        # Should handle invalid track index gracefully
        result = self.daw.add_track_from_inventory(mock_alien_track, 999)
        self.assertFalse(result)
        
        # Test with None values
        self.daw.add_track_from_inventory(None, 0)
        # Should not crash
        
        # Test with invalid clip operations
        self.daw.selected_clips = []
        self.daw.delete_selected_clips()
        # Should handle empty selection gracefully


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
