#!/usr/bin/env python3
"""
UAP Video Analysis GUI
Professional tkinter interface for advanced UAP video analysis
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import json
import os
import sys
from pathlib import Path
import cv2
from PIL import Image, ImageTk
import yaml
import time
from datetime import datetime

class UAP_AnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("UAP Video Analyzer v2.0 - Advanced Scientific Analysis")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Variables
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="results/gui_analysis")
        self.config_path = tk.StringVar(value="configs/analysis_config.yaml")
        self.analysis_running = False
        self.current_process = None
        
        # Analysis options
        self.analysis_options = {
            'atmospheric': tk.BooleanVar(value=True),
            'physics': tk.BooleanVar(value=True),
            'stereo': tk.BooleanVar(value=True),
            'environmental': tk.BooleanVar(value=True),
            'database': tk.BooleanVar(value=True),
            'acoustic': tk.BooleanVar(value=True),
            'trajectory': tk.BooleanVar(value=True),
            'multispectral': tk.BooleanVar(value=True),
            'quick_mode': tk.BooleanVar(value=False)
        }
        
        self.setup_styles()
        self.create_widgets()
        self.setup_layout()
        
    def setup_styles(self):
        """Configure custom styles for the GUI"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', 
                       font=('Arial', 16, 'bold'),
                       background='#2b2b2b',
                       foreground='#ffffff')
        
        style.configure('Header.TLabel',
                       font=('Arial', 12, 'bold'),
                       background='#2b2b2b',
                       foreground='#4CAF50')
        
        style.configure('Custom.TButton',
                       font=('Arial', 10, 'bold'))
        
        style.configure('Analysis.TCheckbutton',
                       font=('Arial', 10),
                       background='#2b2b2b',
                       foreground='#ffffff')
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(self.main_frame, 
                               text="🛸 UAP Video Analyzer - Advanced Scientific Analysis Suite",
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_input_tab()
        self.create_analysis_tab()
        self.create_progress_tab()
        self.create_results_tab()
        
    def create_input_tab(self):
        """Create video input and preview tab"""
        input_frame = ttk.Frame(self.notebook)
        self.notebook.add(input_frame, text="📹 Video Input")
        
        # Video selection section
        video_section = ttk.LabelFrame(input_frame, text="Video Selection", padding=10)
        video_section.pack(fill=tk.X, padx=10, pady=5)
        
        # Video path
        ttk.Label(video_section, text="Video File:").pack(anchor=tk.W)
        path_frame = ttk.Frame(video_section)
        path_frame.pack(fill=tk.X, pady=5)
        
        self.video_entry = ttk.Entry(path_frame, textvariable=self.video_path, width=60)
        self.video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(path_frame, text="Browse", 
                               command=self.browse_video_file,
                               style='Custom.TButton')
        browse_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Video preview section
        preview_section = ttk.LabelFrame(input_frame, text="Video Preview", padding=10)
        preview_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Video preview canvas
        self.preview_canvas = tk.Canvas(preview_section, 
                                       bg='black', 
                                       width=640, 
                                       height=360)
        self.preview_canvas.pack(pady=10)
        
        # Video info
        self.video_info = scrolledtext.ScrolledText(preview_section, 
                                                   height=8, 
                                                   width=80,
                                                   wrap=tk.WORD)
        self.video_info.pack(fill=tk.X, pady=5)
        
    def create_analysis_tab(self):
        """Create analysis configuration tab"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="🔬 Analysis Config")
        
        # Configuration section
        config_section = ttk.LabelFrame(analysis_frame, text="Configuration", padding=10)
        config_section.pack(fill=tk.X, padx=10, pady=5)
        
        # Config file path
        ttk.Label(config_section, text="Configuration File:").pack(anchor=tk.W)
        config_frame = ttk.Frame(config_section)
        config_frame.pack(fill=tk.X, pady=5)
        
        config_entry = ttk.Entry(config_frame, textvariable=self.config_path, width=60)
        config_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        config_browse_btn = ttk.Button(config_frame, text="Browse",
                                      command=self.browse_config_file)
        config_browse_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Output directory
        ttk.Label(config_section, text="Output Directory:").pack(anchor=tk.W, pady=(10, 0))
        output_frame = ttk.Frame(config_section)
        output_frame.pack(fill=tk.X, pady=5)
        
        output_entry = ttk.Entry(output_frame, textvariable=self.output_dir, width=60)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        output_browse_btn = ttk.Button(output_frame, text="Browse",
                                      command=self.browse_output_dir)
        output_browse_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Analysis options
        options_section = ttk.LabelFrame(analysis_frame, text="Analysis Modules", padding=10)
        options_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create two columns for checkboxes
        options_left = ttk.Frame(options_section)
        options_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        options_right = ttk.Frame(options_section)
        options_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Analysis module checkboxes
        analysis_modules = [
            ('atmospheric', '🌪️ Atmospheric Analysis', 'Heat distortion, air displacement detection'),
            ('physics', '🔬 Physics Analysis', 'G-force, energy conservation, anomaly detection'),
            ('stereo', '📐 Stereo Vision Analysis', '3D reconstruction and depth analysis'),
            ('environmental', '🌍 Environmental Correlation', 'Weather, atmospheric, temporal analysis'),
            ('database', '🎯 Database Matching', 'Pattern matching against known phenomena'),
            ('acoustic', '🎵 Acoustic Analysis', 'Audio signature and sonic boom detection'),
            ('trajectory', '🚀 Trajectory Prediction', 'Physics-based movement prediction'),
            ('multispectral', '🌈 Multi-Spectral Analysis', 'Thermal, IR, UV spectrum analysis')
        ]
        
        for i, (key, title, desc) in enumerate(analysis_modules):
            parent = options_left if i < 4 else options_right
            
            cb = ttk.Checkbutton(parent, 
                               text=title,
                               variable=self.analysis_options[key],
                               style='Analysis.TCheckbutton')
            cb.pack(anchor=tk.W, pady=2)
            
            desc_label = ttk.Label(parent, text=f"   {desc}", 
                                 font=('Arial', 8), 
                                 foreground='#888888')
            desc_label.pack(anchor=tk.W, padx=(20, 0))
        
        # Quick mode option
        separator = ttk.Separator(options_section, orient='horizontal')
        separator.pack(fill=tk.X, pady=10)
        
        quick_cb = ttk.Checkbutton(options_section,
                                  text="⚡ Quick Mode (Core analyses only)",
                                  variable=self.analysis_options['quick_mode'],
                                  style='Analysis.TCheckbutton')
        quick_cb.pack(anchor=tk.W)
        
    def create_progress_tab(self):
        """Create analysis progress and control tab"""
        progress_frame = ttk.Frame(self.notebook)
        self.notebook.add(progress_frame, text="⚡ Analysis Progress")
        
        # Control section
        control_section = ttk.LabelFrame(progress_frame, text="Analysis Control", padding=10)
        control_section.pack(fill=tk.X, padx=10, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(control_section)
        button_frame.pack()
        
        self.start_btn = ttk.Button(button_frame, 
                                   text="🚀 Start Analysis",
                                   command=self.start_analysis,
                                   style='Custom.TButton')
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame,
                                  text="⏹️ Stop Analysis", 
                                  command=self.stop_analysis,
                                  state=tk.DISABLED,
                                  style='Custom.TButton')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress section
        progress_section = ttk.LabelFrame(progress_frame, text="Progress", padding=10)
        progress_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_section,
                                           variable=self.progress_var,
                                           maximum=100,
                                           length=400)
        self.progress_bar.pack(pady=5)
        
        # Status label
        self.status_label = ttk.Label(progress_section, 
                                     text="Ready to analyze",
                                     font=('Arial', 10))
        self.status_label.pack(pady=5)
        
        # Output log
        ttk.Label(progress_section, text="Analysis Log:").pack(anchor=tk.W, pady=(10, 0))
        self.log_text = scrolledtext.ScrolledText(progress_section,
                                                 height=15,
                                                 width=80,
                                                 wrap=tk.WORD,
                                                 bg='#1e1e1e',
                                                 fg='#00ff00',
                                                 font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
    def create_results_tab(self):
        """Create results visualization tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📊 Results")
        
        # Results summary
        summary_section = ttk.LabelFrame(results_frame, text="Analysis Summary", padding=10)
        summary_section.pack(fill=tk.X, padx=10, pady=5)
        
        self.results_summary = scrolledtext.ScrolledText(summary_section,
                                                        height=8,
                                                        width=80,
                                                        wrap=tk.WORD)
        self.results_summary.pack(fill=tk.X, pady=5)
        
        # Results actions
        actions_frame = ttk.Frame(summary_section)
        actions_frame.pack(fill=tk.X, pady=5)
        
        open_results_btn = ttk.Button(actions_frame,
                                     text="📂 Open Results Folder",
                                     command=self.open_results_folder)
        open_results_btn.pack(side=tk.LEFT, padx=5)
        
        export_report_btn = ttk.Button(actions_frame,
                                      text="📄 Generate Report",
                                      command=self.generate_report)
        export_report_btn.pack(side=tk.LEFT, padx=5)
        
        # Detailed results
        details_section = ttk.LabelFrame(results_frame, text="Detailed Results", padding=10)
        details_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Results tree
        self.results_tree = ttk.Treeview(details_section, columns=('Value', 'Description'))
        self.results_tree.heading('#0', text='Analysis Component')
        self.results_tree.heading('Value', text='Result')
        self.results_tree.heading('Description', text='Description')
        self.results_tree.pack(fill=tk.BOTH, expand=True)
        
    def setup_layout(self):
        """Final layout adjustments"""
        # Set initial tab
        self.notebook.select(0)
        
    def browse_video_file(self):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.video_path.set(filename)
            self.load_video_preview()
            
    def browse_config_file(self):
        """Browse for configuration file"""
        filename = filedialog.askopenfilename(
            title="Select Configuration File",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if filename:
            self.config_path.set(filename)
            
    def browse_output_dir(self):
        """Browse for output directory"""
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir.set(dirname)
            
    def load_video_preview(self):
        """Load video preview and information"""
        video_path = self.video_path.get()
        if not video_path or not os.path.exists(video_path):
            return
            
        try:
            # Open video with OpenCV
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Get first frame for preview
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize for preview
                preview_width = 640
                preview_height = int((preview_width / width) * height)
                frame_resized = cv2.resize(frame_rgb, (preview_width, preview_height))
                
                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update canvas
                self.preview_canvas.config(width=preview_width, height=preview_height)
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(preview_width//2, preview_height//2, 
                                               image=photo, anchor=tk.CENTER)
                self.preview_canvas.image = photo  # Keep a reference
            
            cap.release()
            
            # Update video info
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            info_text = f"""Video Information:
            
File: {os.path.basename(video_path)}
Size: {file_size:.1f} MB
Resolution: {width}x{height}
Frame Rate: {fps:.2f} fps
Duration: {duration:.2f} seconds
Total Frames: {frame_count}
            
Status: Ready for analysis"""
            
            self.video_info.delete(1.0, tk.END)
            self.video_info.insert(1.0, info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video preview: {str(e)}")
            
    def start_analysis(self):
        """Start the video analysis"""
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file")
            return
            
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", "Video file does not exist")
            return
            
        # Disable start button, enable stop button
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.analysis_running = True
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        self.log("Starting UAP video analysis...")
        
        # Switch to progress tab
        self.notebook.select(2)
        
        # Start analysis in separate thread
        analysis_thread = threading.Thread(target=self.run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        
    def run_analysis(self):
        """Run the analysis process"""
        try:
            # Build command
            cmd = [sys.executable, "run_advanced_analysis.py", self.video_path.get()]
            
            # Add configuration
            if self.config_path.get() and os.path.exists(self.config_path.get()):
                cmd.extend(["-c", self.config_path.get()])
                
            # Add output directory
            if self.output_dir.get():
                cmd.extend(["-o", self.output_dir.get()])
                
            # Add analysis options
            if self.analysis_options['quick_mode'].get():
                cmd.append("--quick")
            else:
                for option, var in self.analysis_options.items():
                    if option != 'quick_mode' and var.get():
                        cmd.append(f"--{option}")
            
            self.log(f"Command: {' '.join(cmd)}")
            
            # Run process
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output line by line
            while True:
                if not self.analysis_running:
                    break
                    
                output = self.current_process.stdout.readline()
                if output == '' and self.current_process.poll() is not None:
                    break
                    
                if output:
                    self.root.after(0, self.log, output.strip())
                    
                    # Update progress based on output patterns
                    if "Extracting frames" in output:
                        self.root.after(0, self.update_status, "Extracting video frames...")
                    elif "Analyzing motion" in output:
                        self.root.after(0, self.update_status, "Analyzing motion patterns...")
                    elif "Physics analysis" in output:
                        self.root.after(0, self.update_status, "Running physics analysis...")
                    elif "Environmental" in output:
                        self.root.after(0, self.update_status, "Environmental correlation...")
                        
            # Wait for process to complete
            return_code = self.current_process.wait()
            
            if return_code == 0:
                self.root.after(0, self.analysis_complete)
            else:
                self.root.after(0, self.analysis_error, f"Analysis failed with code {return_code}")
                
        except Exception as e:
            self.root.after(0, self.analysis_error, str(e))
            
    def stop_analysis(self):
        """Stop the current analysis"""
        if self.current_process:
            self.current_process.terminate()
            
        self.analysis_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_status("Analysis stopped by user")
        self.log("Analysis stopped by user")
        
    def analysis_complete(self):
        """Handle analysis completion"""
        self.analysis_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_var.set(100)
        self.update_status("Analysis completed successfully!")
        self.log("✅ Analysis completed successfully!")
        
        # Switch to results tab
        self.notebook.select(3)
        self.load_results()
        
    def analysis_error(self, error_msg):
        """Handle analysis error"""
        self.analysis_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_status(f"Analysis failed: {error_msg}")
        self.log(f"❌ Error: {error_msg}")
        messagebox.showerror("Analysis Error", f"Analysis failed: {error_msg}")
        
    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
    def update_status(self, status):
        """Update status label"""
        self.status_label.config(text=status)
        
    def load_results(self):
        """Load and display analysis results"""
        output_dir = self.output_dir.get()
        if not os.path.exists(output_dir):
            return
            
        # Look for results files
        results_summary = "Analysis Results Summary:\n\n"
        
        # Check for common result files
        result_files = [
            "analysis_summary.json",
            "motion_analysis.json", 
            "physics_results.json",
            "atmospheric_analysis.json"
        ]
        
        for filename in result_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                results_summary += f"✅ {filename} - Generated\n"
            else:
                results_summary += f"❌ {filename} - Not found\n"
                
        self.results_summary.delete(1.0, tk.END)
        self.results_summary.insert(1.0, results_summary)
        
    def open_results_folder(self):
        """Open results folder in file explorer"""
        output_dir = self.output_dir.get()
        if os.path.exists(output_dir):
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", output_dir])
            elif sys.platform == "win32":  # Windows
                subprocess.run(["explorer", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])
        else:
            messagebox.showwarning("Warning", "Results folder does not exist")
            
    def generate_report(self):
        """Generate analysis report"""
        messagebox.showinfo("Info", "Report generation feature coming soon!")

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = UAP_AnalyzerGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()