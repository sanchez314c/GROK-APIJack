#!/usr/bin/env python3

import os
import re
from pathlib import Path

def extract_scripts_from_file(file_path):
    """Extract all Python scripts from a markdown file"""
    scripts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all Python code blocks
    pattern = r'```(?:python|bash)?\s*\n(#!/usr/bin/env python3.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for i, match in enumerate(matches):
        if match.strip().startswith('#!/usr/bin/env python3'):
            scripts.append({
                'file': os.path.basename(file_path),
                'script_num': i + 1,
                'content': match.strip()
            })
    
    return scripts

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
            scripts = extract_scripts_from_file(file_path)
            print(f"  Found {len(scripts)} scripts")
            all_scripts.extend(scripts)
        else:
            print(f"File not found: {file_name}")
    
    # Save each script
    output_dir = base_dir / "recovered_scripts"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nTotal scripts found: {len(all_scripts)}")
    
    for i, script in enumerate(all_scripts):
        output_file = output_dir / f"script_{i+1:02d}_{script['file']}_{script['script_num']}.py"
        with open(output_file, 'w') as f:
            f.write(script['content'])
        print(f"Saved: {output_file}")
    
    # Create summary
    summary_file = output_dir / "EXTRACTION_SUMMARY.md"
    with open(summary_file, 'w') as f:
        f.write("# Recovered Grok Scripts Summary\n\n")
        f.write(f"Total scripts extracted: {len(all_scripts)}\n\n")
        
        for i, script in enumerate(all_scripts):
            f.write(f"## Script {i+1:02d}\n")
            f.write(f"- Source: {script['file']}\n")
            f.write(f"- Script number in file: {script['script_num']}\n")
            f.write(f"- File: script_{i+1:02d}_{script['file']}_{script['script_num']}.py\n")
            f.write(f"- Length: {len(script['content'])} characters\n\n")
    
    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    main()