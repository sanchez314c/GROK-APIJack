#!/bin/bash
# GitHub Repository Setup Script for UAP Analysis Suite

echo "🛸 Setting up UAP Analysis Suite for GitHub..."
echo "=============================================="

# Initialize git repository
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "📦 Git repository already exists"
fi

# Create .gitkeep files for empty directories
echo "📁 Creating directory structure..."
mkdir -p data/raw data/processed results tests docs
touch data/raw/.gitkeep
touch data/processed/.gitkeep  
touch results/.gitkeep
touch tests/.gitkeep
touch docs/.gitkeep
echo "✅ Directory structure created"

# Set up git configuration (you can modify these)
echo "⚙️ Setting up git configuration..."
git config user.name "UAP Research Team"
git config user.email "research@uapanalysis.org"
echo "✅ Git configuration set"

# Add files to git
echo "📝 Adding files to git..."
git add .
echo "✅ Files added to staging"

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "🛸 Initial commit: UAP Video Analysis Suite

- Complete analysis pipeline with 10+ scientific modules
- Professional GUI interface with tkinter
- Command-line tools for batch processing
- Hardware acceleration (Metal MPS, CUDA, OpenCL)
- Comprehensive documentation and examples
- Cross-platform compatibility (macOS, Linux, Windows)

Features:
- Atmospheric analysis (heat distortion, air displacement)
- Physics analysis (G-force, energy conservation, anomaly detection)  
- Environmental correlation (weather, atmospheric conditions)
- Stereo vision and 3D reconstruction
- Pattern matching and database comparison
- Acoustic analysis and sonic boom detection
- Trajectory prediction with ML
- Multi-spectral analysis (thermal, IR, UV)
- Motion tracking and enhancement
- Professional scientific reporting

🚀 Ready for UAP video investigation and research!

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

echo "✅ Initial commit created"

# Display repository status
echo ""
echo "📊 Repository Status:"
echo "===================="
git status --short
echo ""
git log --oneline -1
echo ""

# Instructions for GitHub
echo "🌐 GitHub Setup Instructions:"
echo "============================="
echo "1. Create a new repository on GitHub named 'UAPAnalysis'"
echo "2. Run these commands to connect to GitHub:"
echo ""
echo "   git remote add origin https://github.com/yourusername/UAPAnalysis.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Optional: Set up GitHub Pages for documentation"
echo "4. Optional: Enable GitHub Actions for CI/CD"
echo ""

# Display project statistics
echo "📈 Project Statistics:"
echo "===================="
echo "Total files: $(find . -type f | wc -l)"
echo "Python files: $(find . -name "*.py" | wc -l)"
echo "Documentation files: $(find . -name "*.md" | wc -l)"
echo "Configuration files: $(find . -name "*.yaml" -o -name "*.yml" | wc -l)"
echo ""

# Final message
echo "🎉 GitHub repository setup complete!"
echo ""
echo "Next steps:"
echo "- Review and customize the README.md"
echo "- Add your GitHub username to links in README.md"
echo "- Test the installation scripts"
echo "- Upload to GitHub using the commands above"
echo "- Add sample videos to test the analysis"
echo ""
echo "🛸 Ready to share your UAP analysis tools with the world!"