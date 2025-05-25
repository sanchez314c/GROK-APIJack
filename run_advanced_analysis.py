#!/usr/bin/env python3
"""
Advanced UAP Analysis Runner
============================

Runs the complete advanced analysis pipeline with all the new analyzers.
This script provides access to all analysis capabilities including stereo vision,
environmental correlation, database matching, acoustic analysis, trajectory 
prediction, and multi-spectral analysis.

Usage:
    python run_advanced_analysis.py <video_file> [options]
"""

import argparse
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from analyzers.legacy_comprehensive_analyzer import UAPAnalyzer
    from analyzers import (
        AtmosphericAnalyzer, PhysicsAnalyzer, SignatureAnalyzer,
        MLClassifier, StereoVisionAnalyzer, EnvironmentalAnalyzer,
        DatabaseMatcher, AcousticAnalyzer, TrajectoryPredictor,
        MultiSpectralAnalyzer
    )
except ImportError as e:
    print(f"Error: Could not import analysis components: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)

def load_config(config_path=None):
    """Load configuration file."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "analysis_config.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_output_directory(video_path, output_dir=None):
    """Create output directory for analysis results."""
    if output_dir is None:
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"advanced_uap_analysis_{video_name}_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    return output_path

def display_advanced_banner():
    """Display the advanced analysis banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    ADVANCED UAP ANALYSIS PIPELINE v2.0                  ║
    ║                     Complete Scientific Analysis Suite                   ║
    ║                                                                          ║
    ║  🌍 Environmental Correlation  🔬 Physics Analysis  🎯 Pattern Matching  ║
    ║  🎵 Acoustic Analysis          📐 3D Reconstruction  🌈 Multi-Spectral   ║
    ║  🧠 ML Classification          🚀 Trajectory Prediction                  ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def run_advanced_analysis(video_path, config, output_path):
    """Run the complete advanced analysis pipeline."""
    print("🚀 Initializing Advanced Analysis Pipeline...")
    
    # Initialize core analyzer for video processing
    core_analyzer = UAPAnalyzer(
        str(video_path), 
        str(output_path),
        verbose=config['performance']['verbose_output']
    )
    
    print(f"📹 Video Properties:")
    print(f"    Resolution: {core_analyzer.width}x{core_analyzer.height}")
    print(f"    Frame Rate: {core_analyzer.fps} fps")
    print(f"    Duration: {core_analyzer.duration:.2f} seconds")
    print(f"    Total Frames: {core_analyzer.frame_count}")
    
    # Extract frames for analysis
    print("\n📊 EXTRACTING FRAMES...")
    frames = core_analyzer.extract_frames(extract_all=True)
    metadata = {
        'fps': core_analyzer.fps,
        'width': core_analyzer.width,
        'height': core_analyzer.height,
        'duration': core_analyzer.duration,
        'frame_count': core_analyzer.frame_count
    }
    
    analysis_results = {}
    
    # Core analyses
    print("\n🎯 CORE MOTION ANALYSIS...")
    if config['analysis_types']['motion']:
        motion_results = core_analyzer.analyze_motion()
        analysis_results['motion'] = motion_results
    
    print("\n💡 CORE LUMINOSITY ANALYSIS...")
    if config['analysis_types']['luminosity']:
        luminosity_results = core_analyzer.analyze_luminosity()
        analysis_results['luminosity'] = luminosity_results
    
    # Advanced analyses
    print("\n🌪️ ATMOSPHERIC ANALYSIS...")
    if config['analysis_types']['atmospheric']:
        atmospheric_analyzer = AtmosphericAnalyzer(config)
        analysis_results['atmospheric'] = atmospheric_analyzer.analyze(frames, metadata)
    
    print("\n⚗️ PHYSICS ANALYSIS...")
    if config['analysis_types']['physics']:
        physics_analyzer = PhysicsAnalyzer(config)
        analysis_results['physics'] = physics_analyzer.analyze(frames, metadata)
    
    print("\n🔬 SIGNATURE ANALYSIS...")
    if config['analysis_types']['signature']:
        signature_analyzer = SignatureAnalyzer(config)
        analysis_results['signature'] = signature_analyzer.analyze(frames, metadata)
    
    print("\n🧠 MACHINE LEARNING CLASSIFICATION...")
    if config['analysis_types']['ml_classification']:
        ml_classifier = MLClassifier(config)
        analysis_results['ml_classification'] = ml_classifier.analyze(frames, metadata)
    
    print("\n📐 STEREO VISION & 3D ANALYSIS...")
    if config['analysis_types']['stereo_vision']:
        stereo_analyzer = StereoVisionAnalyzer(config)
        analysis_results['stereo_vision'] = stereo_analyzer.analyze(frames, metadata)
    
    print("\n🌍 ENVIRONMENTAL CORRELATION...")
    if config['analysis_types']['environmental']:
        env_analyzer = EnvironmentalAnalyzer(config)
        analysis_results['environmental'] = env_analyzer.analyze(frames, metadata)
    
    print("\n📚 DATABASE PATTERN MATCHING...")
    if config['analysis_types']['database_matching']:
        db_matcher = DatabaseMatcher(config)
        analysis_results['database_matching'] = db_matcher.analyze(frames, metadata, analysis_results)
    
    print("\n🎵 ACOUSTIC ANALYSIS...")
    if config['analysis_types']['acoustic']:
        acoustic_analyzer = AcousticAnalyzer(config)
        analysis_results['acoustic'] = acoustic_analyzer.analyze(frames, metadata)
    
    print("\n🚀 TRAJECTORY PREDICTION...")
    if config['analysis_types']['trajectory_prediction']:
        trajectory_predictor = TrajectoryPredictor(config)
        analysis_results['trajectory_prediction'] = trajectory_predictor.analyze(frames, metadata, analysis_results)
    
    print("\n🌈 MULTI-SPECTRAL ANALYSIS...")
    if config['analysis_types']['multispectral']:
        multispectral_analyzer = MultiSpectralAnalyzer(config)
        analysis_results['multispectral'] = multispectral_analyzer.analyze(frames, metadata)
    
    # Generate output videos and reports
    print("\n🎬 GENERATING OUTPUT VIDEOS...")
    if config['output']['create_enhanced_video']:
        core_analyzer.generate_enhanced_video()
    
    if config['output']['create_motion_tracking']:
        core_analyzer.generate_motion_tracking_video()
    
    if config['output']['create_stabilized_video']:
        core_analyzer.create_stabilized_video()
    
    print("\n📋 GENERATING COMPREHENSIVE REPORT...")
    core_analyzer.create_summary_report()
    
    # Generate advanced analysis report
    generate_advanced_report(analysis_results, output_path, metadata)
    
    return analysis_results

def generate_advanced_report(analysis_results, output_path, metadata):
    """Generate comprehensive advanced analysis report."""
    report_path = output_path / "advanced_analysis_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Advanced UAP Analysis Report\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 📹 Video Properties\n\n")
        f.write(f"- **Resolution:** {metadata['width']}x{metadata['height']}\n")
        f.write(f"- **Frame Rate:** {metadata['fps']} fps\n")
        f.write(f"- **Duration:** {metadata['duration']:.2f} seconds\n")
        f.write(f"- **Total Frames:** {metadata['frame_count']}\n\n")
        
        # Atmospheric Analysis
        if 'atmospheric' in analysis_results:
            f.write("## 🌪️ Atmospheric Analysis\n\n")
            atm_data = analysis_results['atmospheric']
            if 'atmospheric_conditions' in atm_data:
                atm_cond = atm_data['atmospheric_conditions']
                f.write(f"- **Atmospheric Quality Score:** {atm_cond.get('atmospheric_quality_score', 'N/A')}\n")
                f.write(f"- **Average Clarity:** {atm_cond.get('average_clarity', 'N/A'):.3f}\n")
                f.write(f"- **Atmospheric Stability:** {atm_cond.get('atmospheric_stability', 'N/A')}\n")
                f.write(f"- **Inversion Layers Detected:** {atm_cond.get('inversion_layers_detected', False)}\n\n")
        
        # Physics Analysis  
        if 'physics' in analysis_results:
            f.write("## ⚗️ Physics Analysis\n\n")
            phys_data = analysis_results['physics']
            if 'anomaly_score' in phys_data:
                f.write(f"- **Physics Anomaly Score:** {phys_data['anomaly_score']:.3f}\n")
            if 'trajectory_analysis' in phys_data:
                traj = phys_data['trajectory_analysis']
                f.write(f"- **Path Efficiency:** {traj.get('path_efficiency', 'N/A'):.3f}\n")
                f.write(f"- **Direction Changes:** {len(traj.get('direction_changes', []))}\n")
            if 'gravitational_analysis' in phys_data:
                grav = phys_data['gravitational_analysis']
                f.write(f"- **Anti-Gravity Events:** {grav.get('anti_gravity_count', 0)}\n\n")
        
        # Database Matching
        if 'database_matching' in analysis_results:
            f.write("## 📚 Pattern Matching Results\n\n")
            db_data = analysis_results['database_matching']
            if 'overall_classification' in db_data:
                classification = db_data['overall_classification']
                primary = classification.get('primary_classification')
                if primary:
                    f.write(f"- **Primary Classification:** {primary['category']} - {primary['specific_type']}\n")
                    f.write(f"- **Confidence:** {primary['confidence']:.3f}\n")
                    f.write(f"- **Probability:** {primary['probability']:.3f}\n\n")
        
        # Trajectory Prediction
        if 'trajectory_prediction' in analysis_results:
            f.write("## 🚀 Trajectory Analysis\n\n")
            traj_data = analysis_results['trajectory_prediction']
            if 'behavior_classification' in traj_data:
                behavior = traj_data['behavior_classification']
                f.write(f"- **Movement Classification:** {behavior.get('classification', 'N/A')}\n")
                f.write(f"- **Classification Confidence:** {behavior.get('confidence', 'N/A'):.3f}\n")
            if 'anomaly_detection' in traj_data:
                anomalies = traj_data['anomaly_detection']
                f.write(f"- **Trajectory Anomalies:** {anomalies.get('anomaly_count', 0)}\n")
                f.write(f"- **Anomaly Severity:** {anomalies.get('anomaly_severity', 'N/A')}\n\n")
        
        # Multi-spectral Analysis
        if 'multispectral' in analysis_results:
            f.write("## 🌈 Multi-Spectral Analysis\n\n")
            ms_data = analysis_results['multispectral']
            if 'color_temperature_analysis' in ms_data:
                ct_data = ms_data['color_temperature_analysis']
                f.write(f"- **Average Color Temperature:** {ct_data.get('average_color_temperature', 'N/A'):.0f}K\n")
                f.write(f"- **Temperature Classification:** {ct_data.get('temperature_classification', 'N/A')}\n")
            if 'spectral_anomaly_detection' in ms_data:
                sa_data = ms_data['spectral_anomaly_detection']
                f.write(f"- **Spectral Anomalies:** {sa_data.get('total_anomaly_count', 0)}\n\n")
        
        f.write("## 📊 Analysis Summary\n\n")
        f.write("This comprehensive analysis examined the footage using multiple advanced techniques ")
        f.write("including atmospheric physics, environmental correlation, pattern matching against ")
        f.write("known phenomena, and multi-spectral characteristics.\n\n")
        
        f.write("For detailed technical data, see the accompanying JSON files in the analysis directory.\n\n")

def main():
    """Main entry point for advanced analysis."""
    parser = argparse.ArgumentParser(
        description="Advanced UAP Video Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('-c', '--config', help='Path to configuration YAML file', default=None)
    parser.add_argument('-o', '--output', help='Output directory for analysis results', default=None)
    parser.add_argument('--quick', action='store_true', help='Run quick analysis (core components only)')
    parser.add_argument('--atmospheric', action='store_true', help='Run atmospheric analysis only')
    parser.add_argument('--physics', action='store_true', help='Run physics analysis only')
    parser.add_argument('--stereo', action='store_true', help='Run stereo vision analysis only')
    parser.add_argument('--environmental', action='store_true', help='Run environmental analysis only')
    parser.add_argument('--database', action='store_true', help='Run database matching only')
    parser.add_argument('--acoustic', action='store_true', help='Run acoustic analysis only')
    parser.add_argument('--trajectory', action='store_true', help='Run trajectory prediction only')
    parser.add_argument('--multispectral', action='store_true', help='Run multi-spectral analysis only')
    
    args = parser.parse_args()
    
    # Display banner
    display_advanced_banner()
    
    # Validate input file
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return 1
    
    print(f"📁 Analyzing video: {video_path.name}")
    print(f"📏 File size: {video_path.stat().st_size / (1024*1024):.1f} MB")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Handle specific analysis requests
        if args.quick:
            # Disable advanced analyses for quick mode
            for key in config['analysis_types']:
                if key not in ['motion', 'luminosity', 'pulse_detection']:
                    config['analysis_types'][key] = False
            print("⚡ Running in quick mode (core analyses only)")
        
        # Handle individual analysis requests
        individual_analyses = ['atmospheric', 'physics', 'stereo', 'environmental', 
                             'database', 'acoustic', 'trajectory', 'multispectral']
        
        if any(getattr(args, analysis) for analysis in individual_analyses):
            # Disable all analyses first
            for key in config['analysis_types']:
                config['analysis_types'][key] = False
            
            # Enable only requested analyses
            if args.atmospheric:
                config['analysis_types']['atmospheric'] = True
                config['analysis_types']['motion'] = True  # Required for atmospheric
            if args.physics:
                config['analysis_types']['physics'] = True
                config['analysis_types']['motion'] = True  # Required for physics
            if args.stereo:
                config['analysis_types']['stereo_vision'] = True
            if args.environmental:
                config['analysis_types']['environmental'] = True
            if args.database:
                config['analysis_types']['database_matching'] = True
                config['analysis_types']['motion'] = True  # Required for matching
            if args.acoustic:
                config['analysis_types']['acoustic'] = True
            if args.trajectory:
                config['analysis_types']['trajectory_prediction'] = True
                config['analysis_types']['motion'] = True  # Required for trajectory
            if args.multispectral:
                config['analysis_types']['multispectral'] = True
        
        # Create output directory
        output_path = create_output_directory(video_path, args.output)
        print(f"📂 Output directory: {output_path}")
        
        # Run analysis
        results = run_advanced_analysis(video_path, config, output_path)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"🎉 ADVANCED ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"📂 Results saved to: {output_path}")
        print(f"📋 Advanced report: {output_path}/advanced_analysis_report.md")
        print(f"📊 Technical data: {output_path}/analysis/")
        
        # Count completed analyses
        completed_analyses = sum(1 for key, enabled in config['analysis_types'].items() if enabled)
        print(f"🔬 Completed {completed_analyses} different analysis types")
        
        print(f"{'='*80}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())