#!/usr/bin/env python3
"""
macOS Application Builder for UAP Video Analysis Suite
Creates a standalone .app bundle that users can run without Python installation
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import tempfile

class MacOSAppBuilder:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.app_name = "UAP Video Analyzer"
        self.bundle_id = "org.uapanalysis.videoanalyzer"
        self.version = "2.0.0"
        self.build_dir = self.project_dir / "build"
        self.dist_dir = self.project_dir / "dist"
        
    def check_dependencies(self):
        """Check if PyInstaller and other build tools are available"""
        print("🔍 Checking build dependencies...")
        
        try:
            import PyInstaller
            print("✅ PyInstaller available")
        except ImportError:
            print("❌ PyInstaller not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])
            
        # Check for additional dependencies
        required_packages = ["pillow", "py2app"]
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"✅ {package} available")
            except ImportError:
                print(f"📦 Installing {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package])
                
    def create_app_icon(self):
        """Create a macOS app icon (.icns file)"""
        print("🎨 Creating app icon...")
        
        icon_dir = self.project_dir / "assets"
        icon_dir.mkdir(exist_ok=True)
        
        # Create a simple icon using Python (you can replace with a custom design)
        icon_script = '''
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

# Create icon image
size = 512
img = PIL.Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = PIL.ImageDraw.Draw(img)

# Background gradient
for i in range(size):
    alpha = int(255 * (1 - i / size))
    color = (20, 20, 40, alpha)
    draw.rectangle([0, i, size, i+1], fill=color)

# UFO shape
center_x, center_y = size // 2, size // 2
ufo_width, ufo_height = size // 3, size // 8
ellipse_bbox = [
    center_x - ufo_width,
    center_y - ufo_height,
    center_x + ufo_width,
    center_y + ufo_height
]
draw.ellipse(ellipse_bbox, fill=(150, 150, 200, 200), outline=(200, 200, 255, 255), width=3)

# Dome
dome_width, dome_height = ufo_width // 2, ufo_height // 2
dome_bbox = [
    center_x - dome_width,
    center_y - ufo_height - dome_height,
    center_x + dome_width,
    center_y - ufo_height + dome_height
]
draw.ellipse(dome_bbox, fill=(100, 100, 150, 180), outline=(150, 150, 200, 255), width=2)

# Lights
for angle in [0, 60, 120, 180, 240, 300]:
    import math
    light_x = center_x + int((ufo_width - 20) * math.cos(math.radians(angle)))
    light_y = center_y + int((ufo_height - 5) * math.sin(math.radians(angle)))
    draw.ellipse([light_x-8, light_y-8, light_x+8, light_y+8], 
                fill=(255, 255, 100, 255))

img.save('assets/icon.png')
print("✅ Icon created at assets/icon.png")
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(icon_script)
            f.flush()
            subprocess.run([sys.executable, f.name], cwd=self.project_dir)
            os.unlink(f.name)
            
        # Convert PNG to ICNS
        png_path = icon_dir / "icon.png"
        icns_path = icon_dir / "icon.icns"
        
        if png_path.exists():
            # Use iconutil to create icns
            iconset_path = icon_dir / "icon.iconset"
            iconset_path.mkdir(exist_ok=True)
            
            # Create multiple sizes for iconset
            sizes = [16, 32, 64, 128, 256, 512]
            for size in sizes:
                for scale in [1, 2]:
                    if size * scale <= 1024:
                        output_name = f"icon_{size}x{size}"
                        if scale == 2:
                            output_name += "@2x"
                        output_name += ".png"
                        
                        subprocess.run([
                            "sips", "-z", str(size * scale), str(size * scale),
                            str(png_path), "--out", str(iconset_path / output_name)
                        ], capture_output=True)
            
            # Convert iconset to icns
            subprocess.run([
                "iconutil", "-c", "icns", str(iconset_path), "-o", str(icns_path)
            ], capture_output=True)
            
            # Clean up
            shutil.rmtree(iconset_path, ignore_errors=True)
            
            if icns_path.exists():
                print(f"✅ App icon created: {icns_path}")
                return icns_path
                
        return None
        
    def create_pyinstaller_spec(self):
        """Create PyInstaller spec file for the application"""
        print("📝 Creating PyInstaller spec file...")
        
        spec_content = f'''
# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

project_dir = Path("{self.project_dir}")

# Collect all analyzer modules
analyzer_modules = []
analyzers_dir = project_dir / "src" / "analyzers"
if analyzers_dir.exists():
    for py_file in analyzers_dir.glob("*.py"):
        if py_file.name != "__init__.py":
            analyzer_modules.append(str(py_file))

# Collect configuration files
config_files = []
configs_dir = project_dir / "configs"
if configs_dir.exists():
    for config_file in configs_dir.glob("*.yaml"):
        config_files.append((str(config_file), "configs"))

# Collect assets
asset_files = []
assets_dir = project_dir / "assets"
if assets_dir.exists():
    for asset_file in assets_dir.iterdir():
        if asset_file.is_file():
            asset_files.append((str(asset_file), "assets"))

a = Analysis(
    ['{self.project_dir / "stable_gui.py"}'],
    pathex=[str(project_dir)],
    binaries=[],
    datas=config_files + asset_files + [
        (str(project_dir / "src"), "src"),
    ],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        'cv2',
        'numpy',
        'scipy',
        'yaml',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'threading',
        'subprocess',
        'multiprocessing',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'matplotlib.tests',
        'numpy.tests',
        'scipy.tests',
        'pytest',
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{self.app_name.replace(" ", "")}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='{self.app_name.replace(" ", "")}',
)

app = BUNDLE(
    coll,
    name='{self.app_name}.app',
    icon='{self.project_dir / "assets" / "icon.icns"}',
    bundle_identifier='{self.bundle_id}',
    version='{self.version}',
    info_plist={{
        'CFBundleName': '{self.app_name}',
        'CFBundleDisplayName': '{self.app_name}',
        'CFBundleGetInfoString': 'Advanced Scientific Analysis Tool for UAP Video Investigation',
        'CFBundleIdentifier': '{self.bundle_id}',
        'CFBundleVersion': '{self.version}',
        'CFBundleShortVersionString': '{self.version}',
        'NSHumanReadableCopyright': 'Copyright © 2024 UAP Analysis Team',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,
        'LSMinimumSystemVersion': '10.15',
        'CFBundleDocumentTypes': [
            {{
                'CFBundleTypeName': 'Video File',
                'CFBundleTypeRole': 'Viewer',
                'LSItemContentTypes': [
                    'public.movie',
                    'com.apple.quicktime-movie',
                    'public.mpeg-4',
                    'public.avi'
                ],
                'LSHandlerRank': 'Alternate'
            }}
        ]
    }},
)
'''
        
        spec_path = self.project_dir / f"{self.app_name.replace(' ', '')}.spec"
        with open(spec_path, 'w') as f:
            f.write(spec_content)
            
        print(f"✅ Spec file created: {spec_path}")
        return spec_path
        
    def build_app(self):
        """Build the macOS application using PyInstaller"""
        print("🔨 Building macOS application...")
        
        # Create icon
        icon_path = self.create_app_icon()
        
        # Create spec file
        spec_path = self.create_pyinstaller_spec()
        
        # Clean previous builds
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        if self.dist_dir.exists():
            shutil.rmtree(self.dist_dir)
            
        # Run PyInstaller
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm",
            str(spec_path)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=self.project_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            app_path = self.dist_dir / f"{self.app_name}.app"
            if app_path.exists():
                print(f"✅ App built successfully: {app_path}")
                return app_path
            else:
                print("❌ App bundle not found after build")
                return None
        else:
            print(f"❌ Build failed:")
            print(result.stdout)
            print(result.stderr)
            return None
            
    def optimize_app(self, app_path):
        """Optimize the built application"""
        print("⚡ Optimizing application...")
        
        # Remove unnecessary files
        frameworks_dir = app_path / "Contents" / "Frameworks"
        if frameworks_dir.exists():
            # Remove test files and documentation
            for item in frameworks_dir.rglob("*test*"):
                if item.is_file() or item.is_dir():
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    except:
                        pass
        
        # Get app size
        size_result = subprocess.run(["du", "-sh", str(app_path)], capture_output=True, text=True)
        if size_result.returncode == 0:
            size = size_result.stdout.split()[0]
            print(f"📦 App size: {size}")
            
    def create_dmg(self, app_path):
        """Create a DMG installer for the application"""
        print("💿 Creating DMG installer...")
        
        dmg_name = f"{self.app_name.replace(' ', '_')}_v{self.version}.dmg"
        dmg_path = self.dist_dir / dmg_name
        
        # Remove existing DMG
        if dmg_path.exists():
            dmg_path.unlink()
            
        # Create temporary DMG directory
        temp_dmg_dir = self.dist_dir / "dmg_temp"
        if temp_dmg_dir.exists():
            shutil.rmtree(temp_dmg_dir)
        temp_dmg_dir.mkdir()
        
        # Copy app to temp directory
        temp_app_path = temp_dmg_dir / f"{self.app_name}.app"
        shutil.copytree(app_path, temp_app_path)
        
        # Create Applications symlink
        applications_link = temp_dmg_dir / "Applications"
        applications_link.symlink_to("/Applications")
        
        # Create DMG
        cmd = [
            "hdiutil", "create",
            "-volname", self.app_name,
            "-srcfolder", str(temp_dmg_dir),
            "-ov", "-format", "UDZO",
            str(dmg_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temp directory
        shutil.rmtree(temp_dmg_dir)
        
        if result.returncode == 0 and dmg_path.exists():
            print(f"✅ DMG created: {dmg_path}")
            return dmg_path
        else:
            print(f"❌ DMG creation failed: {result.stderr}")
            return None
            
    def create_installer_package(self, app_path):
        """Create a .pkg installer package"""
        print("📦 Creating installer package...")
        
        pkg_name = f"{self.app_name.replace(' ', '_')}_v{self.version}.pkg"
        pkg_path = self.dist_dir / pkg_name
        
        # Create package
        cmd = [
            "pkgbuild",
            "--root", str(app_path.parent),
            "--identifier", self.bundle_id,
            "--version", self.version,
            "--install-location", "/Applications",
            str(pkg_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and pkg_path.exists():
            print(f"✅ PKG created: {pkg_path}")
            return pkg_path
        else:
            print(f"❌ PKG creation failed: {result.stderr}")
            return None
            
    def build(self):
        """Main build process"""
        print(f"🛸 Building {self.app_name} for macOS...")
        print("=" * 60)
        
        # Check dependencies
        self.check_dependencies()
        
        # Build app
        app_path = self.build_app()
        if not app_path:
            return False
            
        # Optimize
        self.optimize_app(app_path)
        
        # Create installers
        dmg_path = self.create_dmg(app_path)
        pkg_path = self.create_installer_package(app_path)
        
        # Summary
        print("\n🎉 Build Complete!")
        print("=" * 60)
        print(f"📱 Application: {app_path}")
        if dmg_path:
            print(f"💿 DMG Installer: {dmg_path}")
        if pkg_path:
            print(f"📦 PKG Installer: {pkg_path}")
            
        print(f"\n🚀 Ready for distribution!")
        print(f"Users can now download and run {self.app_name} without installing Python!")
        
        return True

if __name__ == "__main__":
    builder = MacOSAppBuilder()
    success = builder.build()
    sys.exit(0 if success else 1)