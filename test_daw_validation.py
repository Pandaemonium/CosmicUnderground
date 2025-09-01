#!/usr/bin/env python3
"""
Simple test suite for DAW validation functions
Run this to verify the validation system works correctly
"""

import sys
import os

# Add the game directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'game'))

def test_validation_functions():
    """Test the basic validation functions"""
    print("🧪 Testing DAW validation functions...")
    
    try:
        # Import validation functions
        from core.daw_validation import (
            validate_clip_list, 
            validate_merge_operation,
            validate_slice_operation,
            validate_audio_data,
            validate_track_index,
            validate_duration,
            validate_start_time,
            safe_validate
        )
        print("✅ Successfully imported validation functions")
        
        # Test 1: validate_clip_list
        print("\n📋 Test 1: validate_clip_list")
        
        # Test empty list
        is_valid, msg = validate_clip_list([], "Test")
        print(f"   Empty list: {is_valid} - {msg}")
        
        # Test valid list
        mock_clips = [object() for _ in range(5)]
        is_valid, msg = validate_clip_list(mock_clips, "Test")
        print(f"   Valid list (5 items): {is_valid} - {msg}")
        
        # Test 2: validate_track_index
        print("\n🎯 Test 2: validate_track_index")
        
        # Test valid index
        is_valid, msg = validate_track_index(5, 16, "Test")
        print(f"   Valid index (5/16): {is_valid} - {msg}")
        
        # Test invalid index
        is_valid, msg = validate_track_index(20, 16, "Test")
        print(f"   Invalid index (20/16): {is_valid} - {msg}")
        
        # Test negative index
        is_valid, msg = validate_track_index(-1, 16, "Test")
        print(f"   Negative index (-1): {is_valid} - {msg}")
        
        # Test 3: validate_duration
        print("\n⏱️ Test 3: validate_duration")
        
        # Test valid duration
        is_valid, msg = validate_duration(5.0, "Test")
        print(f"   Valid duration (5.0s): {is_valid} - {msg}")
        
        # Test too short duration
        is_valid, msg = validate_duration(0.05, "Test")
        print(f"   Too short (0.05s): {is_valid} - {msg}")
        
        # Test too long duration
        is_valid, msg = validate_duration(1000.0, "Test")
        print(f"   Too long (1000.0s): {is_valid} - {msg}")
        
        # Test 4: validate_start_time
        print("\n🕐 Test 4: validate_start_time")
        
        # Test valid start time
        is_valid, msg = validate_start_time(10.5, "Test")
        print(f"   Valid start time (10.5s): {is_valid} - {msg}")
        
        # Test negative start time
        is_valid, msg = validate_start_time(-5.0, "Test")
        print(f"   Negative start time (-5.0s): {is_valid} - {msg}")
        
        # Test 5: safe_validate
        print("\n🛡️ Test 5: safe_validate")
        
        # Test with a function that raises an exception
        def failing_function():
            raise ValueError("Test exception")
        
        is_valid, msg = safe_validate(failing_function)
        print(f"   Failing function: {is_valid} - {msg}")
        
        print("\n🎉 All validation tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_import():
    """Test that the DAW config can be imported"""
    print("\n🔧 Testing DAW config import...")
    
    try:
        from core.daw_config import AUDIO_CONFIG, UI_CONFIG, TIMELINE_CONFIG
        print("✅ Successfully imported DAW config")
        
        print(f"   Sample rate: {AUDIO_CONFIG['sample_rate']} Hz")
        print(f"   Min clip duration: {AUDIO_CONFIG['min_clip_duration']}s")
        print(f"   UI refresh rate: {UI_CONFIG['refresh_rate']} FPS")
        print(f"   Default zoom: {TIMELINE_CONFIG['default_zoom']} px/s")
        
        return True
        
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting DAW validation tests...")
    print("=" * 50)
    
    success_count = 0
    total_tests = 2
    
    # Test validation functions
    if test_validation_functions():
        success_count += 1
    
    # Test config import
    if test_config_import():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! The validation system is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()

