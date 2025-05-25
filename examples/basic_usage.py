#!/usr/bin/env python3
"""
Basic Usage Example for UAP Video Analysis Suite
Shows how to use the analysis tools programmatically
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from analyzers.motion_analyzer import MotionAnalyzer
from analyzers.atmospheric_analyzer import AtmosphericAnalyzer
from processors.frame_processor import FrameProcessor

def basic_analysis_example():
    """
    Example of basic programmatic usage
    """
    print("🛸 UAP Analysis Suite - Basic Usage Example")
    print("=" * 50)
    
    # Video file path (replace with your video)
    video_path = "data/raw/sample_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("Please add a video file to analyze or update the path.")
        return
    
    # Initialize components
    frame_processor = FrameProcessor()
    motion_analyzer = MotionAnalyzer()
    atmospheric_analyzer = AtmosphericAnalyzer()
    
    try:
        # Step 1: Process video frames
        print("📹 Processing video frames...")
        frames = frame_processor.extract_frames(video_path, max_frames=100)
        print(f"✅ Extracted {len(frames)} frames")
        
        # Step 2: Motion analysis
        print("🎯 Analyzing motion patterns...")
        motion_results = motion_analyzer.analyze(frames)
        print(f"✅ Motion analysis complete - detected {motion_results.get('object_count', 0)} objects")
        
        # Step 3: Atmospheric analysis
        print("🌪️ Analyzing atmospheric effects...")
        atmospheric_results = atmospheric_analyzer.analyze(frames)
        print(f"✅ Atmospheric analysis complete - anomaly score: {atmospheric_results.get('anomaly_score', 0):.3f}")
        
        # Step 4: Display results
        print("\n📊 Analysis Results Summary:")
        print("-" * 30)
        print(f"Motion Objects Detected: {motion_results.get('object_count', 0)}")
        print(f"Average Motion Energy: {motion_results.get('avg_energy', 0):.3f}")
        print(f"Atmospheric Anomaly Score: {atmospheric_results.get('anomaly_score', 0):.3f}")
        print(f"Heat Distortion Detected: {'Yes' if atmospheric_results.get('heat_distortion', False) else 'No'}")
        
        return {
            'motion': motion_results,
            'atmospheric': atmospheric_results
        }
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

def gui_launch_example():
    """
    Example of launching the GUI programmatically
    """
    print("🖥️ Launching GUI interface...")
    
    try:
        import subprocess
        subprocess.run([sys.executable, "stable_gui.py"])
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")

def command_line_example():
    """
    Example of using command line interface
    """
    print("💻 Command Line Usage Examples:")
    print("-" * 40)
    
    examples = [
        "# Quick analysis",
        "python run_advanced_analysis.py video.mp4 --quick",
        "",
        "# Full analysis with all modules",
        "python run_advanced_analysis.py video.mp4 -o results/",
        "",
        "# Specific analysis types",
        "python run_advanced_analysis.py video.mp4 --atmospheric --physics",
        "",
        "# With custom configuration",
        "python run_advanced_analysis.py video.mp4 -c configs/custom_config.yaml"
    ]
    
    for example in examples:
        print(example)

if __name__ == "__main__":
    print("🛸 UAP Video Analysis Suite - Examples")
    print("Choose an example to run:")
    print("1. Basic programmatic analysis")
    print("2. Launch GUI interface")
    print("3. Show command line examples")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            basic_analysis_example()
        elif choice == "2":
            gui_launch_example()
        elif choice == "3":
            command_line_example()
        else:
            print("Invalid choice. Running basic analysis example...")
            basic_analysis_example()
            
    except KeyboardInterrupt:
        print("\n👋 Example interrupted by user")
    except Exception as e:
        print(f"❌ Error running example: {e}")