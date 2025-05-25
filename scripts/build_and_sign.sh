#!/bin/bash
# Advanced macOS App Builder with Code Signing and Notarization
# Builds a production-ready, signed, and notarized macOS application

set -e  # Exit on any error

# Configuration
APP_NAME="UAP Video Analyzer"
BUNDLE_ID="org.uapanalysis.videoanalyzer"
VERSION="2.0.0"
DEVELOPER_ID="Developer ID Application: Your Name (TEAM_ID)"
NOTARIZATION_EMAIL="your-email@example.com"

echo "🛸 Building and Signing UAP Video Analyzer for macOS"
echo "=================================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script must be run on macOS"
    exit 1
fi

# Check for Xcode Command Line Tools
if ! command -v codesign &> /dev/null; then
    echo "❌ Xcode Command Line Tools not found. Please install with:"
    echo "   xcode-select --install"
    exit 1
fi

# Install build dependencies
echo "📦 Installing build dependencies..."
pip install -r build_requirements.txt

# Build the application
echo "🔨 Building application..."
python build_macos_app.py

# Check if app was built successfully
APP_PATH="dist/${APP_NAME}.app"
if [ ! -d "$APP_PATH" ]; then
    echo "❌ Application build failed - app bundle not found"
    exit 1
fi

echo "✅ Application built successfully"

# Code signing (optional - requires Apple Developer account)
if [ ! -z "$DEVELOPER_ID" ] && [ "$DEVELOPER_ID" != "Developer ID Application: Your Name (TEAM_ID)" ]; then
    echo "✍️  Signing application..."
    
    # Sign all binaries and frameworks
    find "$APP_PATH" -type f \( -name "*.dylib" -o -name "*.so" \) -exec codesign --force --verify --verbose --sign "$DEVELOPER_ID" {} \;
    
    # Sign the main executable
    codesign --force --verify --verbose --sign "$DEVELOPER_ID" "$APP_PATH/Contents/MacOS/"*
    
    # Sign the app bundle
    codesign --force --verify --verbose --sign "$DEVELOPER_ID" --entitlements scripts/entitlements.plist "$APP_PATH"
    
    echo "✅ Application signed successfully"
    
    # Verify signature
    codesign --verify --verbose=2 "$APP_PATH"
    spctl --assess --verbose=2 "$APP_PATH"
    
else
    echo "⚠️  Skipping code signing (no Developer ID configured)"
fi

# Create DMG
echo "💿 Creating DMG installer..."
DMG_NAME="${APP_NAME// /_}_v${VERSION}.dmg"
DMG_PATH="dist/$DMG_NAME"

# Remove existing DMG
[ -f "$DMG_PATH" ] && rm "$DMG_PATH"

# Create temporary DMG directory
TEMP_DMG_DIR="dist/dmg_temp"
[ -d "$TEMP_DMG_DIR" ] && rm -rf "$TEMP_DMG_DIR"
mkdir -p "$TEMP_DMG_DIR"

# Copy app and create Applications symlink
cp -R "$APP_PATH" "$TEMP_DMG_DIR/"
ln -s "/Applications" "$TEMP_DMG_DIR/Applications"

# Create DMG
hdiutil create -volname "$APP_NAME" -srcfolder "$TEMP_DMG_DIR" -ov -format UDZO "$DMG_PATH"

# Clean up
rm -rf "$TEMP_DMG_DIR"

echo "✅ DMG created: $DMG_PATH"

# Sign DMG (if code signing is enabled)
if [ ! -z "$DEVELOPER_ID" ] && [ "$DEVELOPER_ID" != "Developer ID Application: Your Name (TEAM_ID)" ]; then
    echo "✍️  Signing DMG..."
    codesign --force --sign "$DEVELOPER_ID" "$DMG_PATH"
    echo "✅ DMG signed successfully"
fi

# Notarization (optional - requires Apple Developer account and app-specific password)
if [ ! -z "$NOTARIZATION_EMAIL" ] && [ "$NOTARIZATION_EMAIL" != "your-email@example.com" ]; then
    echo "📋 Submitting for notarization..."
    
    # Submit for notarization
    xcrun altool --notarize-app \
        --primary-bundle-id "$BUNDLE_ID" \
        --username "$NOTARIZATION_EMAIL" \
        --password "@keychain:AC_PASSWORD" \
        --file "$DMG_PATH"
    
    echo "✅ Submitted for notarization (check email for status)"
    echo "ℹ️  After approval, staple the notarization:"
    echo "   xcrun stapler staple '$DMG_PATH'"
else
    echo "⚠️  Skipping notarization (no Apple ID configured)"
fi

# Create GitHub release assets
echo "📋 Preparing GitHub release assets..."
RELEASE_DIR="dist/github_release"
mkdir -p "$RELEASE_DIR"

# Copy DMG to release directory
cp "$DMG_PATH" "$RELEASE_DIR/"

# Create checksums
cd "$RELEASE_DIR"
shasum -a 256 *.dmg > checksums.txt
cd - > /dev/null

# Create release notes
cat > "$RELEASE_DIR/release_notes.md" << EOF
# UAP Video Analyzer v${VERSION} - macOS Release

## 🛸 What's New
- Complete UAP video analysis suite
- Professional GUI interface
- 10+ scientific analysis modules
- Hardware acceleration (Metal MPS)
- Cross-platform compatibility

## 📦 Installation
1. Download \`${DMG_NAME}\`
2. Open the DMG file
3. Drag "UAP Video Analyzer" to your Applications folder
4. Launch from Applications or Spotlight

## 🔐 Security
This application is $([ ! -z "$DEVELOPER_ID" ] && [ "$DEVELOPER_ID" != "Developer ID Application: Your Name (TEAM_ID)" ] && echo "code signed and " || echo "")built for macOS 10.15+

## 📊 System Requirements
- macOS 10.15 (Catalina) or later
- 8GB RAM (16GB recommended)
- 10GB free disk space
- Optional: Apple Silicon or dedicated GPU for acceleration

## 🔗 Links
- [Source Code](https://github.com/yourusername/UAPAnalysis)
- [Documentation](https://github.com/yourusername/UAPAnalysis/wiki)
- [Issues](https://github.com/yourusername/UAPAnalysis/issues)

## 📊 File Information
\`\`\`
$(ls -lh ${DMG_NAME})
$(cat checksums.txt)
\`\`\`
EOF

echo "✅ GitHub release assets prepared in: $RELEASE_DIR"

# Display final summary
echo ""
echo "🎉 Build Complete!"
echo "=================="
echo "📱 App Bundle: $APP_PATH"
echo "💿 DMG Installer: $DMG_PATH"
echo "📦 Release Assets: $RELEASE_DIR"
echo ""
echo "📏 File Sizes:"
du -sh "$APP_PATH" "$DMG_PATH"
echo ""

if [ ! -z "$DEVELOPER_ID" ] && [ "$DEVELOPER_ID" != "Developer ID Application: Your Name (TEAM_ID)" ]; then
    echo "✅ Application is code signed and ready for distribution"
else
    echo "⚠️  Application is unsigned - users may see security warnings"
    echo "   Consider setting up code signing for production releases"
fi

echo ""
echo "🚀 Ready for GitHub release!"
echo "Upload the DMG file and release notes to GitHub Releases"