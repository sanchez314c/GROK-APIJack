#!/usr/bin/env python3
"""
Setup script for UAP Video Analysis Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name="uap-video-analysis",
    version="2.0.0",
    author="UAP Research Community",
    author_email="research@uap-analysis.org",
    description="Comprehensive video analysis pipeline for UAP/aerial phenomena research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/uap-video-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Multimedia :: Video :: Display",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "pytest-cov>=2.12.0",
        ],
        "advanced": [
            "torch>=1.9.0",
            "transformers>=4.20.0", 
            "open3d>=0.15.0",
            "scikit-image>=0.18.0",
        ],
        "gpu": [
            "torch>=1.9.0+cu118",
            "torchvision>=0.10.0+cu118",
        ]
    },
    entry_points={
        "console_scripts": [
            "uap-analyze=run_analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "*.md"],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/uap-video-analysis/issues",
        "Source": "https://github.com/your-username/uap-video-analysis",
        "Documentation": "https://github.com/your-username/uap-video-analysis/wiki",
    },
)