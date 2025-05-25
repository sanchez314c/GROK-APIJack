#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

def test_gui():
    root = tk.Tk()
    root.title("UAP Analyzer Test")
    root.geometry("400x300")
    
    # Simple test interface
    frame = ttk.Frame(root, padding="20")
    frame.pack(fill=tk.BOTH, expand=True)
    
    ttk.Label(frame, text="🛸 UAP Analyzer GUI Test", 
              font=('Arial', 14, 'bold')).pack(pady=10)
    
    video_path = tk.StringVar()
    
    ttk.Label(frame, text="Video File:").pack(anchor=tk.W)
    
    path_frame = ttk.Frame(frame)
    path_frame.pack(fill=tk.X, pady=5)
    
    entry = ttk.Entry(path_frame, textvariable=video_path, width=40)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def browse():
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.mov *.avi"), ("All files", "*.*")]
        )
        if filename:
            video_path.set(filename)
            messagebox.showinfo("Success", f"Selected: {os.path.basename(filename)}")
    
    ttk.Button(path_frame, text="Browse", command=browse).pack(side=tk.RIGHT, padx=(5,0))
    
    def test_analysis():
        if not video_path.get():
            messagebox.showerror("Error", "Please select a video file first")
            return
        messagebox.showinfo("Test", "GUI is working! Video analysis would start here.")
    
    ttk.Button(frame, text="🚀 Test Analysis", command=test_analysis).pack(pady=20)
    
    ttk.Label(frame, text="If you can see this and interact with the buttons,\nthe GUI system is working correctly!").pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    test_gui()