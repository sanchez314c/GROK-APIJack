# 🛸 UAP Video Analyzer GUI

A professional tkinter-based graphical interface for the UAP Video Analysis System.

## Features

### 📹 **Video Input**
- Easy file browser for video selection
- Support for multiple video formats (MP4, AVI, MOV, MKV, WMV, FLV)
- Configurable output directory selection

### 🔬 **Analysis Configuration**
- **Quick Mode**: Fast analysis with core components only
- **Advanced Mode**: Full scientific analysis suite including:
  - 🌪️ Atmospheric Analysis (heat distortion, air displacement)
  - 🔬 Physics Analysis (G-force, energy conservation, anomaly detection)
  - 📐 Stereo Vision Analysis (3D reconstruction, depth analysis)
  - 🌍 Environmental Correlation (weather, atmospheric conditions)

### ⚡ **Real-time Progress Tracking**
- Live status updates during analysis
- Detailed timestamped log output
- Progress indication with start/stop controls
- Real-time command line output display

### 📊 **Results Management**
- Automatic results folder opening
- Integration with system file explorer
- Success/error notifications
- Easy access to generated analysis files

## Usage

### Starting the GUI
```bash
# Launch the GUI application
python uap_gui.py
```

### Basic Workflow
1. **Select Video**: Click "Browse..." to choose your UAP video file
2. **Choose Output**: Set where results should be saved (default: `results/gui_analysis`)
3. **Configure Analysis**: 
   - Enable "Quick Mode" for faster processing
   - Or select specific advanced analyses
4. **Start Analysis**: Click "🚀 Start Analysis"
5. **Monitor Progress**: Watch real-time log updates
6. **View Results**: Click "📂 Open Results" when complete

### Analysis Modes

#### Quick Mode ⚡
- **Purpose**: Fast preliminary analysis
- **Components**: Core motion analysis, basic object detection
- **Time**: ~2-5 minutes for typical video
- **Best for**: Initial screening, rapid assessment

#### Advanced Mode 🔬
- **Purpose**: Comprehensive scientific analysis
- **Components**: All 10+ specialized analyzers
- **Time**: ~15-30 minutes for typical video
- **Best for**: Detailed investigation, research documentation

## GUI Components

### Main Interface
```
┌─────────────────────────────────────────┐
│     🛸 UAP Video Analyzer v2.0         │
│      Advanced Scientific Analysis       │
├─────────────────────────────────────────┤
│ 📹 Video Input                          │
│   Video File: [________________] Browse │
│   Output Dir: [________________] Browse │
├─────────────────────────────────────────┤
│ 🔬 Analysis Options                     │
│   ☑ ⚡ Quick Mode                       │
│   ┌─ Advanced Analyses ─────────────┐   │
│   │ ☐ 🌪️ Atmospheric Analysis      │   │
│   │ ☐ 🔬 Physics Analysis          │   │
│   │ ☐ 📐 Stereo Vision Analysis    │   │
│   │ ☐ 🌍 Environmental Correlation │   │
│   └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
│     [🚀 Start] [⏹️ Stop] [📂 Results]    │
├─────────────────────────────────────────┤
│ ⚡ Analysis Progress                     │
│   Status: Ready to analyze video        │
│   [████████████████████████████████]    │
│   Analysis Log:                         │
│   ┌─────────────────────────────────┐   │
│   │ [12:34:56] Starting analysis... │   │
│   │ [12:34:57] Extracting frames...│   │
│   │ [12:35:10] Motion analysis...  │   │
│   │                                 │   │
│   └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### Key Features

#### Smart UI Behavior
- **Contextual Controls**: Buttons enable/disable based on state
- **Progress Feedback**: Real-time status updates and log streaming
- **Error Handling**: Clear error messages and recovery options
- **Cross-platform**: Works on macOS, Windows, and Linux

#### Integration
- **Seamless Backend**: Direct integration with command-line analysis tools
- **File Management**: Automatic result folder creation and access
- **Configuration**: Uses existing YAML configuration system

## Technical Details

### Dependencies
- **tkinter**: Built-in Python GUI framework
- **threading**: Background analysis execution
- **subprocess**: Command-line tool integration

### Architecture
```
uap_gui.py
├── UAPAnalyzerGUI (main class)
├── Video input handling
├── Analysis configuration
├── Progress monitoring
├── Results management
└── Cross-platform file operations
```

### Output Structure
```
results/gui_analysis/
├── analysis_summary.json
├── motion_analysis/
├── physics_results/
├── atmospheric_data/
├── processed_frames/
└── visualizations/
```

## Troubleshooting

### Common Issues

**GUI won't start**
- Ensure Python has tkinter installed (usually built-in)
- Check that all dependencies are available

**Analysis fails**
- Verify video file exists and is readable
- Check output directory permissions
- Review log output for specific errors

**No results generated**
- Ensure sufficient disk space
- Check that backend analysis tools are properly installed
- Verify video format is supported

### Error Messages
- **"Please select a video file"**: Choose a video using Browse button
- **"Video file does not exist"**: File may have been moved or deleted
- **"Analysis failed"**: Check log for detailed error information

## Advanced Usage

### Custom Configuration
The GUI uses the same configuration system as the command-line tools:
- Default config: `configs/analysis_config.yaml`
- Modify settings for specialized analysis requirements

### Batch Processing
For multiple videos, use the command-line interface:
```bash
# Process multiple videos
for video in *.mp4; do
    python run_advanced_analysis.py "$video" -o "results/batch_$(basename "$video")"
done
```

### Integration with External Tools
Results are saved in standard formats for integration with:
- Scientific analysis software
- Video editing tools
- 3D visualization applications
- Statistical analysis packages

---

## 🚀 **Quick Start**

1. Launch GUI: `python uap_gui.py`
2. Select your UAP video file
3. Choose Quick Mode for fast results
4. Click "Start Analysis"
5. Open results folder when complete

**The GUI provides a user-friendly interface to the powerful UAP analysis pipeline, making advanced scientific video analysis accessible to researchers, investigators, and enthusiasts.**