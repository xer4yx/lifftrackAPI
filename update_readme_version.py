#!/usr/bin/env python
"""
Simple script to update the version in README.md from version.py
"""
import re

def update_readme_version():
    # Get version from version.py
    with open('version.py', 'r', encoding='utf-8') as f:
        version_content = f.read()
    
    version_match = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', version_content)
    if not version_match:
        print("Warning: Could not find version in version.py")
        return
    
    version = version_match.group(1)
    
    # Update README.md
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()
    except UnicodeDecodeError:
        # Try with different encodings if utf-8 fails
        with open('README.md', 'r', encoding='latin-1') as f:
            readme_content = f.read()
    
    updated_readme = re.sub(
        r'(\*\*Version:\*\*\s*`)v[^`]*(`.*)',
        f'\\1v{version}\\2',
        readme_content
    )
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(updated_readme)
    
    print(f"Updated README.md with version v{version}")

if __name__ == "__main__":
    update_readme_version() 