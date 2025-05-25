#!/usr/bin/env python3
"""
UAP Analysis Setup Test
======================

Quick test to verify the analysis pipeline is properly organized and functional.
"""

import sys
from pathlib import Path
import importlib.util

def test_file_structure():
    """Test that all required files and directories exist."""
    print("Testing file structure...")
    
    required_files = [
        "README.md",
        "requirements.txt", 
        "run_analysis.py",
        "configs/analysis_config.yaml",
        "src/analyzers/legacy_comprehensive_analyzer.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_directory_structure():
    """Test that the directory structure is correct."""
    print("Testing directory structure...")
    
    required_dirs = [
        "src/analyzers",
        "src/processors", 
        "src/visualizers",
        "src/utils",
        "configs",
        "data/raw",
        "data/processed",
        "results",
        "tests"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories present")
        return True

def test_imports():
    """Test that key modules can be imported."""
    print("Testing imports...")
    
    # Add src to path
    sys.path.insert(0, str(Path("src")))
    
    try:
        # Test legacy analyzer import
        spec = importlib.util.spec_from_file_location(
            "legacy_analyzer", 
            "src/analyzers/legacy_comprehensive_analyzer.py"
        )
        legacy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy_module)
        
        # Test that UAPAnalyzer class exists
        if hasattr(legacy_module, 'UAPAnalyzer'):
            print("✅ Legacy UAPAnalyzer can be imported")
            return True
        else:
            print("❌ UAPAnalyzer class not found in legacy module")
            return False
            
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_organization():
    """Test that data has been properly organized."""
    print("Testing data organization...")
    
    # Check if original video is in raw data
    raw_data_path = Path("data/raw")
    video_files = list(raw_data_path.glob("*.mov")) + list(raw_data_path.glob("*.mp4"))
    
    if video_files:
        print(f"✅ Found {len(video_files)} video file(s) in data/raw")
    else:
        print("⚠️ No video files found in data/raw (this is OK if you haven't added any yet)")
    
    # Check if frames are in processed data
    processed_path = Path("data/processed/frames")
    if processed_path.exists():
        frame_count = len(list(processed_path.glob("*.png")))
        print(f"✅ Found {frame_count} processed frames")
    else:
        print("⚠️ No processed frames found (this is OK if you haven't run analysis yet)")
    
    # Check if analysis results exist
    results_path = Path("results")
    if any(results_path.iterdir()):
        print("✅ Analysis results directory contains data")
    else:
        print("⚠️ No analysis results found (this is OK if you haven't run analysis yet)")
    
    return True

def main():
    """Run all tests."""
    print("🔬 UAP Analysis Pipeline Setup Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_directory_structure,
        test_imports,
        test_data_organization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 Setup is complete and ready to use!")
        print("\nTo run analysis on a video:")
        print("  python run_analysis.py your_video.mp4")
        print("\nTo run quick analysis:")
        print("  python run_analysis.py --quick your_video.mp4")
    else:
        print("⚠️ Some issues found. Please check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)