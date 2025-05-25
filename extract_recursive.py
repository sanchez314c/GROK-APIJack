#!/usr/bin/env python3

import os
import re
import hashlib
from pathlib import Path

def extract_scripts_recursive(base_dir):
    """Recursively extract all Python scripts from markdown files"""
    scripts = []
    
    # Find all .md files recursively
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir)
                print(f"Processing {rel_path}...")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Multiple patterns to catch different formats
                    patterns = [
                        r'```python\s*\n(#!/usr/bin/env python3.*?)```',
                        r'cat\s*<<\s*[\'"]?EOF[\'"]?\s*>\s*["\']?[^"\']*\.py["\']?\s*\n(#!/usr/bin/env python3.*?)EOF',
                        r'```\s*\n(#!/usr/bin/env python3.*?)```',
                        r'```(?:bash)?\s*\n.*?cat.*?\.py.*?\n(#!/usr/bin/env python3.*?)```'
                    ]
                    
                    for pattern_idx, pattern in enumerate(patterns):
                        matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
                        
                        for match_idx, match in enumerate(matches):
                            if match.strip().startswith('#!/usr/bin/env python3'):
                                clean_content = match.strip()
                                if clean_content.endswith('EOF'):
                                    clean_content = clean_content[:-3].strip()
                                
                                # Generate hash for deduplication
                                content_hash = hashlib.md5(clean_content.encode()).hexdigest()
                                
                                scripts.append({
                                    'file': rel_path,
                                    'pattern': pattern_idx + 1,
                                    'match': match_idx + 1,
                                    'content': clean_content,
                                    'hash': content_hash,
                                    'size': len(clean_content)
                                })
                                
                except Exception as e:
                    print(f"Error processing {rel_path}: {e}")
    
    return scripts

def main():
    base_dir = "/Volumes/mpRAID/Development/Github/GROk3-APIJack"
    
    # Extract all scripts recursively
    all_scripts = extract_scripts_recursive(base_dir)
    print(f"\nTotal scripts found: {len(all_scripts)}")
    
    # Remove exact duplicates by hash
    seen_hashes = set()
    unique_scripts = []
    
    for script in all_scripts:
        if script['hash'] not in seen_hashes:
            seen_hashes.add(script['hash'])
            unique_scripts.append(script)
    
    print(f"Unique scripts after removing exact duplicates: {len(unique_scripts)}")
    
    # Create unified output directory
    output_dir = Path(base_dir) / "ALL_GROK_SCRIPTS"
    output_dir.mkdir(exist_ok=True)
    
    # Clear existing files
    for existing_file in output_dir.glob("*.py"):
        existing_file.unlink()
    
    # Save unique scripts
    for i, script in enumerate(unique_scripts, 1):
        filename = f"grok_script_{i:03d}.py"
        output_file = output_dir / filename
        
        with open(output_file, 'w') as f:
            f.write(script['content'])
        
        print(f"Saved: {filename} (from {script['file']}) - {script['size']} bytes")
    
    # Create analysis file
    analysis_file = output_dir / "SCRIPT_ANALYSIS.md"
    with open(analysis_file, 'w') as f:
        f.write("# All Grok Scripts Analysis\n\n")
        f.write(f"Total unique scripts: {len(unique_scripts)}\n\n")
        
        for i, script in enumerate(unique_scripts, 1):
            f.write(f"## Script {i:03d}\n")
            f.write(f"- Source: {script['file']}\n")
            f.write(f"- Size: {script['size']} bytes\n")
            f.write(f"- Hash: {script['hash'][:16]}...\n")
            
            # Analyze features
            content = script['content'].lower()
            features = []
            
            if 'playwright' in content:
                features.append("Playwright automation")
            if 'curl' in content:
                features.append("curl subprocess")
            if 'httpx' in content:
                features.append("httpx HTTP client")
            if 'requests' in content:
                features.append("requests HTTP client")
            if 'fastapi' in content:
                features.append("FastAPI server")
            if 'backoff' in content:
                features.append("Retry with backoff")
            if 'conversation_map' in content:
                features.append("Conversation mapping")
            if 'cookie' in content:
                features.append("Cookie handling")
            if 'refresh_cookies' in content:
                features.append("Cookie refresh")
            if 'stream' in content:
                features.append("Streaming support")
            if 'async def' in content:
                features.append("Async/await")
            
            if features:
                f.write(f"- Features: {', '.join(features)}\n")
            f.write("\n")
    
    print(f"\nAnalysis saved to: {analysis_file}")
    print(f"All scripts saved to: {output_dir}")

if __name__ == "__main__":
    main()