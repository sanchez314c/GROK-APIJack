#!/usr/bin/env python3
"""
UAP Analyzer GUI Launcher
Handles dependencies and launches the GUI
"""

import sys
import subprocess
import importlib

def check_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        'tkinter',
        'cv2',
        'PIL',
        'yaml',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'tkinter':
                importlib.import_module('tkinter')
            else:
                importlib.import_module(package)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        # Map package names to pip names
        pip_names = {
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'tkinter': 'tk'  # Usually comes with Python
        }
        
        for package in missing_packages:
            pip_name = pip_names.get(package, package)
            if package == 'tkinter':
                print("Note: tkinter should come with Python. Please check your Python installation.")
                continue
            
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
                print(f"✅ Installed {pip_name}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {pip_name}")
                return False
    
    return True

def main():
    """Main launcher function"""
    print("🛸 UAP Video Analyzer GUI Launcher")
    print("=" * 50)
    
    print("Checking dependencies...")
    if not check_dependencies():
        print("❌ Dependency check failed. Please install missing packages manually.")
        return
    
    print("\n🚀 Launching GUI...")
    try:
        # Import and run the GUI
        from uap_analyzer_gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"❌ Failed to import GUI module: {e}")
        print("Make sure uap_analyzer_gui.py is in the same directory.")
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")

if __name__ == "__main__":
    main()