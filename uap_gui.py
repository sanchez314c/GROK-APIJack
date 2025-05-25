#!/usr/bin/env python3
"""
UAP Video Analyzer GUI
Clean, professional interface for UAP video analysis
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import os
import sys
from datetime import datetime

class UAPAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🛸 UAP Video Analyzer v2.0")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="results/gui_analysis")
        self.analysis_running = False
        self.current_process = None
        
        # Analysis options
        self.quick_mode = tk.BooleanVar(value=True)
        self.atmospheric = tk.BooleanVar(value=True)
        self.physics = tk.BooleanVar(value=True)
        self.stereo = tk.BooleanVar(value=False)
        self.environmental = tk.BooleanVar(value=False)
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title section
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(title_frame, 
                               text="🛸 UAP Video Analyzer",
                               font=('Arial', 18, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame,
                                  text="Advanced Scientific Analysis Suite",
                                  font=('Arial', 10))
        subtitle_label.pack()
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="📹 Video Input", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Video file selection
        video_frame = ttk.Frame(file_frame)
        video_frame.pack(fill=tk.X)
        
        ttk.Label(video_frame, text="Video File:").pack(anchor=tk.W)
        
        path_frame = ttk.Frame(video_frame)
        path_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.video_entry = ttk.Entry(path_frame, textvariable=self.video_path, width=60)
        self.video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(path_frame, text="Browse...", command=self.browse_video)
        browse_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Output directory
        ttk.Label(video_frame, text="Output Directory:").pack(anchor=tk.W)
        
        output_frame = ttk.Frame(video_frame)
        output_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_dir, width=60)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        output_browse_btn = ttk.Button(output_frame, text="Browse...", command=self.browse_output)
        output_browse_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Analysis options section
        options_frame = ttk.LabelFrame(main_frame, text="🔬 Analysis Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Quick mode
        quick_frame = ttk.Frame(options_frame)
        quick_frame.pack(fill=tk.X, pady=(0, 10))
        
        quick_cb = ttk.Checkbutton(quick_frame, 
                                  text="⚡ Quick Mode (Faster, core analysis only)",
                                  variable=self.quick_mode,
                                  command=self.toggle_quick_mode)
        quick_cb.pack(anchor=tk.W)
        
        # Advanced options
        self.advanced_frame = ttk.LabelFrame(options_frame, text="Advanced Analyses", padding="5")
        self.advanced_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Create two columns for options
        left_col = ttk.Frame(self.advanced_frame)
        left_col.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        
        right_col = ttk.Frame(self.advanced_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.Y, expand=True)
        
        # Left column options
        ttk.Checkbutton(left_col, text="🌪️ Atmospheric Analysis", 
                       variable=self.atmospheric).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(left_col, text="🔬 Physics Analysis", 
                       variable=self.physics).pack(anchor=tk.W, pady=2)
        
        # Right column options  
        ttk.Checkbutton(right_col, text="📐 Stereo Vision Analysis", 
                       variable=self.stereo).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(right_col, text="🌍 Environmental Correlation", 
                       variable=self.environmental).pack(anchor=tk.W, pady=2)
        
        # Disable advanced options if quick mode is on
        self.toggle_quick_mode()
        
        # Control section
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Center the buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack()
        
        self.start_btn = ttk.Button(button_frame, 
                                   text="🚀 Start Analysis",
                                   command=self.start_analysis,
                                   style='Accent.TButton')
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(button_frame,
                                  text="⏹️ Stop Analysis",
                                  command=self.stop_analysis,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        results_btn = ttk.Button(button_frame,
                               text="📂 Open Results",
                               command=self.open_results)
        results_btn.pack(side=tk.LEFT, padx=(20, 0))
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="⚡ Analysis Progress", padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_label = ttk.Label(progress_frame, 
                                     text="Ready to analyze video",
                                     font=('Arial', 10, 'bold'))
        self.status_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame,
                                           variable=self.progress_var,
                                           mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Log area
        ttk.Label(progress_frame, text="Analysis Log:").pack(anchor=tk.W)
        
        self.log_text = scrolledtext.ScrolledText(progress_frame,
                                                 height=12,
                                                 wrap=tk.WORD,
                                                 font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
    def toggle_quick_mode(self):
        """Enable/disable advanced options based on quick mode"""
        if self.quick_mode.get():
            # Disable advanced options
            for child in self.advanced_frame.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, ttk.Checkbutton):
                        widget.configure(state='disabled')
        else:
            # Enable advanced options
            for child in self.advanced_frame.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, ttk.Checkbutton):
                        widget.configure(state='normal')
        
    def browse_video(self):
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
            self.log(f"Selected video: {os.path.basename(filename)}")
            
    def browse_output(self):
        """Browse for output directory"""
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir.set(dirname)
            self.log(f"Output directory: {dirname}")
            
    def start_analysis(self):
        """Start the video analysis"""
        # Validate inputs
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file")
            return
            
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", "Selected video file does not exist")
            return
            
        # Update UI
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self.analysis_running = True
        self.progress_bar.start(10)  # Start indeterminate progress
        
        # Clear log and add start message
        self.log_text.delete(1.0, tk.END)
        self.log("🚀 Starting UAP video analysis...")
        self.log(f"Video: {os.path.basename(self.video_path.get())}")
        self.log(f"Output: {self.output_dir.get()}")
        
        # Start analysis in background thread
        analysis_thread = threading.Thread(target=self.run_analysis, daemon=True)
        analysis_thread.start()
        
    def run_analysis(self):
        """Run the analysis process"""
        try:
            # Build command
            cmd = [sys.executable, "run_advanced_analysis.py", self.video_path.get()]
            
            # Add output directory
            if self.output_dir.get():
                cmd.extend(["-o", self.output_dir.get()])
                
            # Add analysis options
            if self.quick_mode.get():
                cmd.append("--quick")
            else:
                if self.atmospheric.get():
                    cmd.append("--atmospheric")
                if self.physics.get():
                    cmd.append("--physics")
                if self.stereo.get():
                    cmd.append("--stereo")
                if self.environmental.get():
                    cmd.append("--environmental")
            
            self.root.after(0, self.log, f"Command: {' '.join(cmd)}")
            self.root.after(0, self.update_status, "Initializing analysis...")
            
            # Start process
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
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
                    output = output.strip()
                    self.root.after(0, self.log, output)
                    
                    # Update status based on output
                    if "Extracting frames" in output:
                        self.root.after(0, self.update_status, "📹 Extracting video frames...")
                    elif "motion" in output.lower():
                        self.root.after(0, self.update_status, "🎯 Analyzing motion patterns...")
                    elif "physics" in output.lower():
                        self.root.after(0, self.update_status, "🔬 Running physics analysis...")
                    elif "atmospheric" in output.lower():
                        self.root.after(0, self.update_status, "🌪️ Atmospheric analysis...")
                    elif "completed" in output.lower():
                        self.root.after(0, self.update_status, "✅ Analysis completed!")
                        
            # Get return code
            return_code = self.current_process.wait()
            
            if return_code == 0:
                self.root.after(0, self.analysis_complete)
            else:
                self.root.after(0, self.analysis_error, f"Process failed (code {return_code})")
                
        except Exception as e:
            self.root.after(0, self.analysis_error, str(e))
            
    def stop_analysis(self):
        """Stop the current analysis"""
        if self.current_process:
            self.current_process.terminate()
            
        self.analysis_running = False
        self.reset_ui()
        self.update_status("⏹️ Analysis stopped by user")
        self.log("Analysis stopped by user")
        
    def analysis_complete(self):
        """Handle successful analysis completion"""
        self.analysis_running = False
        self.reset_ui()
        self.update_status("✅ Analysis completed successfully!")
        self.log("🎉 Analysis completed successfully!")
        
        # Show completion dialog
        result = messagebox.askyesno(
            "Analysis Complete",
            "Video analysis completed successfully!\n\nWould you like to open the results folder?",
            icon='question'
        )
        
        if result:
            self.open_results()
            
    def analysis_error(self, error_msg):
        """Handle analysis error"""
        self.analysis_running = False
        self.reset_ui()
        self.update_status(f"❌ Analysis failed")
        self.log(f"❌ Error: {error_msg}")
        messagebox.showerror("Analysis Error", f"Analysis failed:\n{error_msg}")
        
    def reset_ui(self):
        """Reset UI to ready state"""
        self.start_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        self.progress_bar.stop()
        
    def log(self, message):
        """Add timestamped message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def update_status(self, status):
        """Update status label"""
        self.status_label.configure(text=status)
        
    def open_results(self):
        """Open results folder in file explorer"""
        output_dir = self.output_dir.get()
        if os.path.exists(output_dir):
            try:
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["open", output_dir])
                elif sys.platform == "win32":  # Windows
                    subprocess.run(["explorer", output_dir])
                else:  # Linux
                    subprocess.run(["xdg-open", output_dir])
                self.log(f"Opened results folder: {output_dir}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open results folder: {e}")
        else:
            messagebox.showwarning("Warning", "Results folder does not exist yet")

def main():
    """Main application entry point"""
    root = tk.Tk()
    
    # Configure style
    style = ttk.Style()
    if "aqua" in style.theme_names():
        style.theme_use("aqua")
    elif "vista" in style.theme_names():
        style.theme_use("vista")
    
    # Create and center the application
    app = UAPAnalyzerGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Set minimum size
    root.minsize(800, 600)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()