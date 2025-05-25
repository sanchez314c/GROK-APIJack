#!/usr/bin/env python3

import os
import re
from pathlib import Path

def extract_scripts_from_file(file_path):
    """Extract all Python scripts from a markdown file using multiple patterns"""
    scripts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Multiple patterns to catch different formats
    patterns = [
        # Standard markdown python blocks
        r'```python\s*\n(#!/usr/bin/env python3.*?)```',
        # Bash commands with cat creating python files  
        r'cat\s*<<\s*[\'"]?EOF[\'"]?\s*>\s*["\']?[^"\']*\.py["\']?\s*\n(#!/usr/bin/env python3.*?)EOF',
        # Direct python code blocks without language specification
        r'```\s*\n(#!/usr/bin/env python3.*?)```',
        # Python blocks within code fences
        r'```(?:bash)?\s*\n.*?cat.*?\.py.*?\n(#!/usr/bin/env python3.*?)```'
    ]
    
    for pattern_idx, pattern in enumerate(patterns):
        matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
        
        for match_idx, match in enumerate(matches):
            if match.strip().startswith('#!/usr/bin/env python3'):
                # Clean up the content
                clean_content = match.strip()
                # Remove any trailing EOF or similar markers
                if clean_content.endswith('EOF'):
                    clean_content = clean_content[:-3].strip()
                
                scripts.append({
                    'file': os.path.basename(file_path),
                    'pattern': pattern_idx + 1,
                    'match': match_idx + 1,
                    'content': clean_content
                })
    
    return scripts

def find_line_numbers(file_path):
    """Find line numbers where Python scripts start"""
    line_numbers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        if '#!/usr/bin/env python3' in line:
            line_numbers.append(i)
    
    return line_numbers

def main():
    base_dir = Path("/Volumes/mpRAID/Development/Github/GROk3-APIJack")
    
    # Files to extract from
    files_to_process = [
        "chat-6dff0ed7-1421-4e96-ac09-232ab33d3e94.md",
        "chat-d658a23a-66bd-492a-b0e3-8bf8d1f58b03.md", 
        "chat-fee94528-01d6-4e85-bdb3-2ec469865117.md",
        "grok-chat.md",
        "grok-chat (1).md",
        "grok-chat (2).md", 
        "grok-chat (3).md",
        "grok-chat (4).md",
        "grok-chat (5).md",
        "grok-chat (6).md",
        "grok-chat (7).md",
        "grok-chat (8).md"
    ]
    
    all_scripts = []
    
    for file_name in files_to_process:
        file_path = base_dir / file_name
        if file_path.exists():
            print(f"Processing {file_name}...")
            
            # Find line numbers for reference
            line_numbers = find_line_numbers(file_path)
            print(f"  Python script headers found at lines: {line_numbers}")
            
            scripts = extract_scripts_from_file(file_path)
            print(f"  Extracted {len(scripts)} complete scripts")
            all_scripts.extend(scripts)
        else:
            print(f"File not found: {file_name}")
    
    # Save each script
    output_dir = base_dir / "recovered_scripts_complete"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nTotal scripts found: {len(all_scripts)}")
    
    # Remove duplicates based on content similarity
    unique_scripts = []
    for script in all_scripts:
        is_duplicate = False
        for existing in unique_scripts:
            # Check if scripts are similar (allowing for minor variations)
            if len(script['content']) == len(existing['content']):
                # If same length, check similarity
                similarity = len(set(script['content'].split()) & set(existing['content'].split())) / len(set(script['content'].split()) | set(existing['content'].split()))
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_scripts.append(script)
    
    print(f"Unique scripts after deduplication: {len(unique_scripts)}")
    
    for i, script in enumerate(unique_scripts):
        output_file = output_dir / f"grok_script_{i+1:02d}_{script['file']}_p{script['pattern']}_m{script['match']}.py"
        with open(output_file, 'w') as f:
            f.write(script['content'])
        print(f"Saved: {output_file}")
    
    # Create detailed summary
    summary_file = output_dir / "COMPLETE_EXTRACTION_SUMMARY.md"
    with open(summary_file, 'w') as f:
        f.write("# Complete Grok Scripts Recovery Summary\n\n")
        f.write(f"Total scripts extracted: {len(all_scripts)}\n")
        f.write(f"Unique scripts after deduplication: {len(unique_scripts)}\n\n")
        
        f.write("## Unique Scripts\n\n")
        for i, script in enumerate(unique_scripts):
            f.write(f"### Script {i+1:02d}\n")
            f.write(f"- Source: {script['file']}\n")
            f.write(f"- Pattern: {script['pattern']}, Match: {script['match']}\n")
            f.write(f"- File: grok_script_{i+1:02d}_{script['file']}_p{script['pattern']}_m{script['match']}.py\n")
            f.write(f"- Length: {len(script['content'])} characters\n")
            f.write(f"- Lines: {len(script['content'].split())}\n\n")
            
            # Extract key features
            content = script['content']
            features = []
            if 'playwright' in content.lower():
                features.append("Playwright automation")
            if 'curl' in content.lower():
                features.append("curl subprocess")
            if 'httpx' in content.lower():
                features.append("httpx HTTP client")
            if 'fastapi' in content.lower():
                features.append("FastAPI server")
            if 'backoff' in content.lower():
                features.append("Retry with backoff")
            if 'conversation_map' in content.lower():
                features.append("Conversation mapping")
            if 'cookie' in content.lower():
                features.append("Cookie handling")
            
            if features:
                f.write(f"- Key features: {', '.join(features)}\n\n")
    
    print(f"\nDetailed summary saved to: {summary_file}")

if __name__ == "__main__":
    main()