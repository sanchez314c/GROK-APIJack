#!/usr/bin/env python3
"""
Stable UAP Video Analyzer GUI
Simplified version to avoid crashes
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import os
import sys
from datetime import datetime

class StableUAPGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("UAP Video Analyzer")
        self.root.geometry("800x600")
        
        # Variables
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="results/gui_analysis")
        self.analysis_running = False
        self.current_process = None
        
        # Analysis options
        self.quick_mode = tk.BooleanVar(value=True)
        self.atmospheric = tk.BooleanVar(value=True)
        self.physics = tk.BooleanVar(value=True)
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create the GUI layout"""
        # Main container with padding
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="UAP Video Analyzer",
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Video Input", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Video file
        ttk.Label(file_frame, text="Video File:").pack(anchor=tk.W)
        
        path_frame = ttk.Frame(file_frame)
        path_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.video_entry = ttk.Entry(path_frame, textvariable=self.video_path, width=60)
        self.video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(path_frame, text="Browse", command=self.browse_video)
        browse_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Output directory
        ttk.Label(file_frame, text="Output Directory:").pack(anchor=tk.W)
        
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_dir, width=60)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        output_browse_btn = ttk.Button(output_frame, text="Browse", command=self.browse_output)
        output_browse_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Analysis options
        options_frame = ttk.LabelFrame(main_frame, text="Analysis Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Quick mode checkbox
        quick_cb = ttk.Checkbutton(options_frame, 
                                  text="Quick Mode (faster analysis)",
                                  variable=self.quick_mode)
        quick_cb.pack(anchor=tk.W, pady=(0, 10))
        
        # Advanced options frame
        adv_frame = ttk.Frame(options_frame)
        adv_frame.pack(fill=tk.X)
        
        ttk.Label(adv_frame, text="Advanced Analyses:").pack(anchor=tk.W)
        
        cb_frame = ttk.Frame(adv_frame)
        cb_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Checkbutton(cb_frame, text="Atmospheric Analysis", 
                       variable=self.atmospheric).pack(side=tk.LEFT)
        ttk.Checkbutton(cb_frame, text="Physics Analysis", 
                       variable=self.physics).pack(side=tk.LEFT, padx=(20, 0))
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=(0, 15))
        
        self.start_btn = ttk.Button(control_frame, 
                                   text="Start Analysis",
                                   command=self.start_analysis)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(control_frame,
                                  text="Stop Analysis",
                                  command=self.stop_analysis,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        results_btn = ttk.Button(control_frame,
                               text="Open Results",
                               command=self.open_results)
        results_btn.pack(side=tk.LEFT, padx=(20, 0))
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Analysis Progress", padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_label = ttk.Label(progress_frame, 
                                     text="Ready to analyze",
                                     font=('Arial', 10, 'bold'))
        self.status_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Log
        ttk.Label(progress_frame, text="Analysis Log:").pack(anchor=tk.W)
        
        self.log_text = scrolledtext.ScrolledText(progress_frame,
                                                 height=15,
                                                 wrap=tk.WORD,
                                                 font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
    def browse_video(self):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
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
            messagebox.showerror("Error", "Video file does not exist")
            return
            
        # Update UI
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self.analysis_running = True
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        self.log("Starting UAP video analysis...")
        self.log(f"Video: {os.path.basename(self.video_path.get())}")
        self.log(f"Output: {self.output_dir.get()}")
        
        # Start analysis thread
        threading.Thread(target=self.run_analysis, daemon=True).start()
        
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
            
            self.root.after(0, self.log, f"Command: {' '.join(cmd)}")
            self.root.after(0, self.update_status, "Initializing...")
            
            # Create output directory
            os.makedirs(self.output_dir.get(), exist_ok=True)
            
            # Start process
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True,
                cwd=os.getcwd()
            )
            
            # Read output
            while True:
                if not self.analysis_running:
                    break
                    
                output = self.current_process.stdout.readline()
                if output == '' and self.current_process.poll() is not None:
                    break
                    
                if output:
                    output = output.strip()
                    if output:  # Only log non-empty lines
                        self.root.after(0, self.log, output)
                        
                        # Update status
                        if "Extracting frames" in output:
                            self.root.after(0, self.update_status, "Extracting frames...")
                        elif "motion" in output.lower():
                            self.root.after(0, self.update_status, "Analyzing motion...")
                        elif "physics" in output.lower():
                            self.root.after(0, self.update_status, "Physics analysis...")
                        elif "atmospheric" in output.lower():
                            self.root.after(0, self.update_status, "Atmospheric analysis...")
                        elif "completed" in output.lower():
                            self.root.after(0, self.update_status, "Analysis completed!")
                            
            # Get result
            return_code = self.current_process.wait()
            
            if return_code == 0:
                self.root.after(0, self.analysis_complete)
            else:
                self.root.after(0, self.analysis_error, f"Process failed (code {return_code})")
                
        except Exception as e:
            self.root.after(0, self.analysis_error, str(e))
            
    def stop_analysis(self):
        """Stop the analysis"""
        if self.current_process:
            self.current_process.terminate()
            
        self.analysis_running = False
        self.reset_ui()
        self.update_status("Analysis stopped")
        self.log("Analysis stopped by user")
        
    def analysis_complete(self):
        """Handle completion"""
        self.analysis_running = False
        self.reset_ui()
        self.update_status("Analysis completed successfully!")
        self.log("Analysis completed successfully!")
        
        # Show completion dialog
        result = messagebox.askyesno(
            "Analysis Complete",
            "Video analysis completed!\n\nOpen results folder?",
            icon='question'
        )
        
        if result:
            self.open_results()
            
    def analysis_error(self, error_msg):
        """Handle error"""
        self.analysis_running = False
        self.reset_ui()
        self.update_status("Analysis failed")
        self.log(f"Error: {error_msg}")
        messagebox.showerror("Analysis Error", f"Analysis failed:\n{error_msg}")
        
    def reset_ui(self):
        """Reset UI state"""
        self.start_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        
    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def update_status(self, status):
        """Update status"""
        self.status_label.configure(text=status)
        
    def open_results(self):
        """Open results folder"""
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
                messagebox.showerror("Error", f"Could not open folder: {e}")
        else:
            messagebox.showwarning("Warning", "Results folder does not exist")

def main():
    """Main function"""
    # Create root window
    root = tk.Tk()
    
    # Create application
    app = StableUAPGUI(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Set minimum size
    root.minsize(700, 500)
    
    # Start application
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("GUI closed by user")
    except Exception as e:
        print(f"GUI error: {e}")

if __name__ == "__main__":
    main()