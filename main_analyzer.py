#!/usr/bin/env python3
"""
UAP Analysis Pipeline - Main Entry Point
========================================

A comprehensive video analysis pipeline for UAP/anomalous aerial phenomena.
Combines multiple analysis techniques including motion tracking, luminosity analysis,
3D reconstruction, EM simulation, and spectral analysis.

Usage:
    python main_analyzer.py <video_path> [--config config.yaml] [--output output_dir]
    
Features:
    - Modular analysis components
    - Configurable pipeline
    - Multiple output formats
    - Automated reporting
    - GPU acceleration support
"""

import argparse
import sys
from pathlib import Path
import yaml
import logging
from datetime import datetime

# Import our modular components
from src.analyzers.motion_analyzer import MotionAnalyzer
from src.analyzers.luminosity_analyzer import LuminosityAnalyzer
from src.analyzers.spectral_analyzer import SpectralAnalyzer
from src.analyzers.depth_analyzer import DepthAnalyzer
from src.analyzers.pulse_analyzer import PulseAnalyzer
from src.processors.video_processor import VideoProcessor
from src.processors.enhancement_processor import EnhancementProcessor
from src.visualizers.report_generator import ReportGenerator
from src.utils.file_manager import FileManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UAPAnalysisPipeline:
    """Main analysis pipeline that coordinates all analysis components."""
    
    def __init__(self, config_path=None):
        """Initialize the pipeline with configuration."""
        self.config = self.load_config(config_path)
        self.file_manager = FileManager()
        self.results = {}
        
    def load_config(self, config_path):
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "analysis_config.yaml"
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_output_directory(self, video_path, output_dir=None):
        """Set up organized output directory structure."""
        if output_dir is None:
            video_name = Path(video_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"uap_analysis_{video_name}_{timestamp}"
        
        output_path = Path(output_dir)
        self.file_manager.create_directory_structure(output_path)
        return output_path
    
    def run_analysis(self, video_path, output_dir=None):
        """Run the complete analysis pipeline."""
        logger.info(f"Starting UAP analysis for: {video_path}")
        
        # Setup output directory
        output_path = self.setup_output_directory(video_path, output_dir)
        logger.info(f"Output directory: {output_path}")
        
        # Initialize video processor
        video_processor = VideoProcessor(video_path, self.config)
        frames, metadata = video_processor.extract_frames()
        
        # Store basic info
        self.results['metadata'] = metadata
        self.results['frame_count'] = len(frames)
        
        # Run analysis components based on configuration
        if self.config['analysis_types']['motion']:
            logger.info("Running motion analysis...")
            motion_analyzer = MotionAnalyzer(self.config)
            self.results['motion'] = motion_analyzer.analyze(frames, metadata)
        
        if self.config['analysis_types']['luminosity']:
            logger.info("Running luminosity analysis...")
            luminosity_analyzer = LuminosityAnalyzer(self.config)
            self.results['luminosity'] = luminosity_analyzer.analyze(frames, metadata)
        
        if self.config['analysis_types']['spectral']:
            logger.info("Running spectral analysis...")
            spectral_analyzer = SpectralAnalyzer(self.config)
            self.results['spectral'] = spectral_analyzer.analyze(frames, metadata)
        
        if self.config['analysis_types']['depth_3d']:
            logger.info("Running 3D depth analysis...")
            depth_analyzer = DepthAnalyzer(self.config)
            self.results['depth'] = depth_analyzer.analyze(frames, metadata)
        
        if self.config['analysis_types']['pulse_detection']:
            logger.info("Running pulse pattern analysis...")
            pulse_analyzer = PulseAnalyzer(self.config)
            self.results['pulses'] = pulse_analyzer.analyze(
                self.results.get('luminosity', {}), metadata
            )
        
        # Generate enhanced outputs
        if self.config['output']['create_enhanced_video']:
            logger.info("Creating enhanced video...")
            enhancer = EnhancementProcessor(self.config)
            enhanced_video_path = enhancer.create_enhanced_video(
                frames, output_path / "enhanced_video.mp4"
            )
            self.results['enhanced_video'] = enhanced_video_path
        
        # Generate comprehensive report
        logger.info("Generating analysis report...")
        report_generator = ReportGenerator(self.config)
        report_path = report_generator.generate_report(
            self.results, output_path
        )
        
        logger.info(f"Analysis complete! Results saved to: {output_path}")
        return output_path, self.results

def display_banner():
    """Display the UAP analyzer banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                     UAP ANALYSIS PIPELINE v2.0                      ║
    ║              Comprehensive Video Analysis for Aerial Phenomena       ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main entry point for the UAP analysis pipeline."""
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
        '--list-analyzers',
        action='store_true',
        help='List available analysis components'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick analysis (motion + luminosity only)'
    )
    
    args = parser.parse_args()
    
    # Display banner
    display_banner()
    
    if args.list_analyzers:
        print("Available Analysis Components:")
        print("  - Motion Analysis: Track object movement and trajectories")
        print("  - Luminosity Analysis: Analyze light patterns and intensity")
        print("  - Spectral Analysis: Examine color and frequency characteristics")
        print("  - 3D Depth Analysis: Reconstruct spatial information")
        print("  - Pulse Detection: Identify rhythmic light patterns")
        print("  - EM Simulation: Visualize potential electromagnetic effects")
        print("  - Geometric Analysis: Detect shape and structural patterns")
        return 0
    
    # Validate input file
    if not Path(args.video_path).exists():
        logger.error(f"Video file not found: {args.video_path}")
        return 1
    
    try:
        # Initialize pipeline
        pipeline = UAPAnalysisPipeline(args.config)
        
        # Modify config for quick analysis
        if args.quick:
            pipeline.config['analysis_types'] = {
                'motion': True,
                'luminosity': True,
                'spectral': False,
                'depth_3d': False,
                'em_noise': False,
                'geometric': False,
                'pulse_detection': True
            }
            logger.info("Running quick analysis mode")
        
        # Run analysis
        output_path, results = pipeline.run_analysis(args.video_path, args.output)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Analysis Summary for {Path(args.video_path).name}")
        print(f"{'='*80}")
        print(f"Frames analyzed: {results.get('frame_count', 'N/A')}")
        print(f"Duration: {results.get('metadata', {}).get('duration', 'N/A'):.2f} seconds")
        print(f"Results saved to: {output_path}")
        
        if 'pulses' in results and results['pulses'].get('detected'):
            print(f"Light pulses detected: {results['pulses']['frequency']:.2f} Hz")
        
        if 'motion' in results:
            print(f"Motion events detected: {len(results['motion'].get('significant_events', []))}")
        
        print(f"{'='*80}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())