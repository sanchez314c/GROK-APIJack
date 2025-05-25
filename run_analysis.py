#!/usr/bin/env python3
"""
UAP Analysis Pipeline Runner
===========================

Simple entry point for running UAP video analysis on any video file.
This script provides an easy-to-use interface for the comprehensive analysis pipeline.

Usage:
    python run_analysis.py <video_file>
    python run_analysis.py /path/to/video.mp4
    python run_analysis.py --quick /path/to/video.mov
    python run_analysis.py --config custom_config.yaml /path/to/video.avi
"""

import sys
import argparse
from pathlib import Path
import yaml
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from analyzers.legacy_comprehensive_analyzer import UAPAnalyzer
except ImportError:
    print("Error: Could not import analysis components. Please ensure the src directory is properly set up.")
    sys.exit(1)

def load_config(config_path=None):
    """Load configuration from YAML file or use defaults."""
    default_config = {
        'analysis_types': {
            'motion': True,
            'luminosity': True,
            'spectral': True,
            'depth_3d': False,  # Requires additional dependencies
            'em_noise': True,
            'pulse_detection': True
        },
        'output': {
            'create_enhanced_video': True,
            'create_stabilized_video': True,
            'create_motion_tracking': True,
            'create_em_simulation': True,
            'save_frame_data': True,
            'create_summary_report': True
        },
        'performance': {
            'verbose_output': True,
            'progress_bars': True
        }
    }
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
            # Merge with defaults
            for section, values in custom_config.items():
                if section in default_config:
                    default_config[section].update(values)
                else:
                    default_config[section] = values
    
    return default_config

def create_output_directory(video_path, output_dir=None):
    """Create output directory for analysis results."""
    if output_dir is None:
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"uap_analysis_{video_name}_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    return output_path

def display_banner():
    """Display analysis banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                     UAP VIDEO ANALYSIS PIPELINE                     ║
║                         Organized & Modular                         ║
╚══════════════════════════════════════════════════════════════════════╝
""")

def main():
    """Main entry point for UAP analysis."""
    parser = argparse.ArgumentParser(
        description="UAP Video Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'video_path',
        help='Path to the input video file'
    )
    
    parser.add_argument(
        '-c', '--config',
        help='Path to configuration YAML file',
        default=None
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output directory for analysis results',
        default=None
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick analysis (essential components only)'
    )
    
    parser.add_argument(
        '--frames',
        type=int,
        help='Maximum number of frames to process (default: all)',
        default=None
    )
    
    parser.add_argument(
        '--no-enhanced',
        action='store_true',
        help='Skip enhanced video generation'
    )
    
    parser.add_argument(
        '--no-stabilized',
        action='store_true',
        help='Skip stabilized video generation'
    )
    
    parser.add_argument(
        '--list-outputs',
        action='store_true',
        help='List all possible output files'
    )
    
    args = parser.parse_args()
    
    # Display banner
    display_banner()
    
    if args.list_outputs:
        print("Possible Output Files:")
        print("  Enhanced Videos:")
        print("    - enhanced_video.mp4 (improved clarity and contrast)")
        print("    - stabilized_video.mp4 (camera shake removed)")
        print("    - motion_tracking.mp4 (motion vectors overlay)")
        print("    - em_noise_simulation.mp4 (EM field visualization)")
        print("")
        print("  Analysis Data:")
        print("    - motion_data.npy (motion vectors and energy)")
        print("    - luminosity_data.json (light patterns)")
        print("    - spectral_data.npy (color frequency analysis)")
        print("    - pulse_data.json (rhythmic pattern analysis)")
        print("")
        print("  Visualizations:")
        print("    - motion_energy.png (motion over time)")
        print("    - luminosity_analysis.png (light patterns)")
        print("    - spectral_analysis.png (color evolution)")
        print("    - pulse_pattern_analysis.png (pulse detection)")
        print("")
        print("  Reports:")
        print("    - analysis_summary.json (structured data)")
        print("    - analysis_report.txt (human readable)")
        return 0
    
    # Validate input file
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1
    
    print(f"Analyzing video: {video_path.name}")
    print(f"File size: {video_path.stat().st_size / (1024*1024):.1f} MB")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Modify config based on arguments
        if args.quick:
            config['analysis_types'].update({
                'spectral': False,
                'depth_3d': False,
                'em_noise': False
            })
            config['output'].update({
                'create_stabilized_video': False,
                'create_em_simulation': False
            })
            print("Running in quick mode (motion + luminosity + pulse analysis)")
        
        if args.no_enhanced:
            config['output']['create_enhanced_video'] = False
        
        if args.no_stabilized:
            config['output']['create_stabilized_video'] = False
        
        # Create output directory
        output_dir = create_output_directory(video_path, args.output)
        print(f"Output directory: {output_dir}")
        
        # Initialize analyzer with legacy comprehensive tool
        analyzer = UAPAnalyzer(
            str(video_path),
            str(output_dir),
            verbose=config['performance']['verbose_output']
        )
        
        print(f"\nVideo Properties:")
        print(f"  Resolution: {analyzer.width}x{analyzer.height}")
        print(f"  Frame Rate: {analyzer.fps} fps")
        print(f"  Duration: {analyzer.duration:.2f} seconds")
        print(f"  Total Frames: {analyzer.frame_count}")
        
        # Extract frames
        print("\n" + "="*60)
        print("EXTRACTING FRAMES")
        print("="*60)
        analyzer.extract_frames(extract_all=True, max_frames=args.frames)
        
        # Run analyses based on configuration
        if config['analysis_types']['motion']:
            print("\n" + "="*60)
            print("MOTION ANALYSIS")
            print("="*60)
            analyzer.analyze_motion()
        
        if config['analysis_types']['luminosity']:
            print("\n" + "="*60)
            print("LUMINOSITY ANALYSIS")
            print("="*60)
            analyzer.analyze_luminosity()
        
        if config['analysis_types']['spectral']:
            print("\n" + "="*60)
            print("SPECTRAL ANALYSIS") 
            print("="*60)
            analyzer.generate_spectral_analysis()
        
        # Generate output videos
        if config['output']['create_motion_tracking']:
            print("\n" + "="*60)
            print("CREATING MOTION TRACKING VIDEO")
            print("="*60)
            analyzer.generate_motion_tracking_video()
        
        if config['output']['create_enhanced_video']:
            print("\n" + "="*60)
            print("CREATING ENHANCED VIDEO")
            print("="*60)
            analyzer.generate_enhanced_video()
        
        if config['output']['create_stabilized_video']:
            print("\n" + "="*60)
            print("CREATING STABILIZED VIDEO")
            print("="*60)
            analyzer.create_stabilized_video()
        
        if config['output']['create_em_simulation'] and config['analysis_types']['em_noise']:
            print("\n" + "="*60)
            print("CREATING EM SIMULATION")
            print("="*60)
            analyzer.simulate_em_noise()
        
        # Generate summary report
        if config['output']['create_summary_report']:
            print("\n" + "="*60)
            print("GENERATING SUMMARY REPORT")
            print("="*60)
            analyzer.create_summary_report()
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_dir}")
        print(f"View report: cat {output_dir}/analysis_report.txt")
        print(f"{'='*80}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        logging.exception("Analysis failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())