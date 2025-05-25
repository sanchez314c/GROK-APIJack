#!/usr/bin/env python3

import os
import re
from pathlib import Path

def analyze_script_content(file_path):
    """Analyze script content to determine its primary characteristics"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    features = {
        'size': len(content),
        'has_playwright': 'playwright' in content.lower(),
        'has_curl': 'curl' in content.lower() and 'subprocess' in content.lower(),
        'has_httpx': 'httpx' in content.lower(),
        'has_requests': 'requests' in content.lower(),
        'has_backoff': 'backoff' in content.lower(),
        'has_cookie_refresh': 'refresh_cookies' in content.lower(),
        'has_auto_install': 'install_package' in content.lower() or 'install_dependencies' in content.lower(),
        'is_minimal': len(content) < 5000,
        'is_large': len(content) > 40000,
        'version_info': None
    }
    
    # Look for version comments
    version_match = re.search(r'# Version: (.+)', content)
    if version_match:
        features['version_info'] = version_match.group(1).strip()
    
    # Look for author/purpose comments
    purpose_match = re.search(r'# Purpose: (.+)', content)
    author_match = re.search(r'# Author: (.+)', content)
    
    if purpose_match:
        features['purpose'] = purpose_match.group(1).strip()
    if author_match:
        features['author'] = author_match.group(1).strip()
    
    return features

def determine_script_name(script_num, features):
    """Determine the best name for a script based on its features"""
    
    # Start with base name
    name_parts = ["grok"]
    
    # Add approach indicator
    if features['has_playwright'] and features['has_curl']:
        if features['has_cookie_refresh']:
            name_parts.append("ultimate_curl_playwright")
        else:
            name_parts.append("curl_playwright")
    elif features['has_curl']:
        if features['has_cookie_refresh']:
            name_parts.append("curl_auto_refresh")
        else:
            name_parts.append("curl_basic")
    elif features['has_httpx'] and features['has_requests']:
        name_parts.append("httpx_requests_hybrid")
    elif features['has_httpx']:
        if features['has_backoff']:
            name_parts.append("httpx_backoff")
        else:
            name_parts.append("httpx_basic")
    elif features['has_requests']:
        name_parts.append("requests_basic")
    else:
        name_parts.append("basic")
    
    # Add feature indicators
    if features['has_auto_install']:
        name_parts.append("auto_deps")
    
    if features['is_minimal']:
        name_parts.append("minimal")
    elif features['is_large']:
        name_parts.append("full")
    
    # Add version if available
    if features.get('version_info'):
        version = features['version_info'].lower()
        if 'updated' in version or 'enhanced' in version:
            name_parts.append("v2")
        elif 'ultimate' in version or 'final' in version:
            name_parts.append("ultimate")
    
    return "_".join(name_parts) + ".py"

def main():
    scripts_dir = Path("/Volumes/mpRAID/Development/Github/GROk3-APIJack/ALL_GROK_SCRIPTS")
    
    # Get all script files
    script_files = sorted(scripts_dir.glob("grok_script_*.py"))
    
    rename_map = {}
    
    for script_file in script_files:
        script_num = script_file.stem.split('_')[-1]
        features = analyze_script_content(script_file)
        new_name = determine_script_name(script_num, features)
        
        # Handle name conflicts
        counter = 1
        original_name = new_name
        while (scripts_dir / new_name).exists() or new_name in rename_map.values():
            name_base = original_name.replace('.py', '')
            new_name = f"{name_base}_v{counter}.py"
            counter += 1
        
        rename_map[script_file.name] = new_name
        
        print(f"{script_file.name} -> {new_name}")
        print(f"  Size: {features['size']} bytes")
        print(f"  Features: Playwright={features['has_playwright']}, Curl={features['has_curl']}, "
              f"HttpX={features['has_httpx']}, Requests={features['has_requests']}")
        print(f"  Auto-refresh={features['has_cookie_refresh']}, Auto-install={features['has_auto_install']}")
        print()
    
    # Perform renames
    print("Performing renames...")
    for old_name, new_name in rename_map.items():
        old_path = scripts_dir / old_name
        new_path = scripts_dir / new_name
        old_path.rename(new_path)
        print(f"Renamed: {old_name} -> {new_name}")
    
    # Update analysis file
    analysis_file = scripts_dir / "SCRIPT_ANALYSIS.md"
    with open(analysis_file, 'r') as f:
        content = f.read()
    
    # Replace old names with new names in analysis
    for old_name, new_name in rename_map.items():
        old_script_num = old_name.split('_')[-1].replace('.py', '')
        content = content.replace(f"grok_script_{old_script_num}.py", new_name)
    
    with open(analysis_file, 'w') as f:
        f.write(content)
    
    # Create a summary file
    summary_file = scripts_dir / "RENAMED_SCRIPTS_SUMMARY.md"
    with open(summary_file, 'w') as f:
        f.write("# Grok Scripts - Renamed and Categorized\n\n")
        f.write(f"Total scripts: {len(rename_map)}\n\n")
        
        # Group by type
        curl_scripts = []
        httpx_scripts = []
        hybrid_scripts = []
        other_scripts = []
        
        for old_name, new_name in rename_map.items():
            if 'curl' in new_name:
                curl_scripts.append(new_name)
            elif 'httpx' in new_name:
                httpx_scripts.append(new_name)
            elif 'hybrid' in new_name:
                hybrid_scripts.append(new_name)
            else:
                other_scripts.append(new_name)
        
        f.write("## Curl-based Scripts\n")
        for script in sorted(curl_scripts):
            f.write(f"- {script}\n")
        
        f.write("\n## HttpX-based Scripts\n")
        for script in sorted(httpx_scripts):
            f.write(f"- {script}\n")
        
        f.write("\n## Hybrid Scripts\n")
        for script in sorted(hybrid_scripts):
            f.write(f"- {script}\n")
        
        f.write("\n## Other Scripts\n")
        for script in sorted(other_scripts):
            f.write(f"- {script}\n")
        
        f.write("\n## Recommended Scripts\n\n")
        f.write("### Most Advanced (Full Features)\n")
        for script in sorted(rename_map.values()):
            if 'ultimate' in script or 'full' in script:
                f.write(f"- **{script}** - Complete implementation with all features\n")
        
        f.write("\n### Production Ready (Stable)\n")
        for script in sorted(rename_map.values()):
            if 'auto_refresh' in script and 'full' in script:
                f.write(f"- **{script}** - Stable with automatic cookie refresh\n")
        
        f.write("\n### Development/Testing (Minimal)\n")
        for script in sorted(rename_map.values()):
            if 'minimal' in script or 'basic' in script:
                f.write(f"- **{script}** - Lightweight for testing\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"Updated analysis at: {analysis_file}")

if __name__ == "__main__":
    main()