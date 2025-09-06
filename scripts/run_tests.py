#!/usr/bin/env python3
"""
Test runner for CosmicDAW system
Run this before refactoring to ensure all functionality is preserved
"""

import sys
import os
import unittest
import pygame

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

def run_daw_tests():
    """Run all DAW-related tests"""
    print("🧪 Running CosmicDAW Test Suite...")
    print("=" * 50)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    if not os.path.exists(start_dir):
        print(f"❌ Tests directory not found: {start_dir}")
        return False
    
    # Discover tests
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Tests Run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed! Safe to proceed with refactoring.")
        return True
    else:
        print("\n❌ Some tests failed. Fix issues before refactoring.")
        return False

def run_specific_test(test_name):
    """Run a specific test class or method"""
    print(f"🧪 Running specific test: {test_name}")
    print("=" * 50)
    
    # Import the test module
    try:
        from tests.test_cosmic_daw import *
    except ImportError as e:
        print(f"❌ Could not import test module: {e}")
        return False
    
    # Find the test class
    test_class = None
    for name in dir(sys.modules[__name__]):
        if name.startswith('Test') and name.endswith(test_name):
            test_class = globals()[name]
            break
    
    if not test_class:
        print(f"❌ Test class not found: {test_name}")
        return False
    
    # Run the specific test
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def main():
    """Main test runner"""
    print("🎵 CosmicDAW Test Runner")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        # Run all tests
        success = run_daw_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
