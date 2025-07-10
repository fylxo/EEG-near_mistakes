#!/usr/bin/env python3
"""
Code Backup Script

This script creates a timestamped backup of important code files before major refactoring.
It includes only source code, configuration, and documentation - excluding data and results.

Usage:
    python backup_code.py [--output-dir BACKUP_DIR] [--description "Optional description"]
"""

import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path
import json


def create_code_backup(output_dir=None, description=None, verbose=True):
    """
    Create a timestamped backup of important code files.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory to store backups (default: ./backups)
    description : str, optional
        Optional description for this backup
    verbose : bool
        Whether to print progress information
    """
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set default backup directory
    if output_dir is None:
        output_dir = "backups"
    
    # Create backup directory with timestamp
    backup_name = f"code_backup_{timestamp}"
    backup_path = os.path.join(output_dir, backup_name)
    
    if verbose:
        print(f"🔄 Creating code backup: {backup_name}")
        print(f"📁 Backup location: {backup_path}")
    
    # Create backup directory
    os.makedirs(backup_path, exist_ok=True)
    
    # Define what to backup (relative to project root)
    backup_targets = [
        # Core source code
        {
            'source': 'src/core',
            'description': 'Core analysis modules'
        },
        {
            'source': 'src/eeg_analysis_package', 
            'description': 'EEG analysis utilities'
        },
        
        # Configuration files
        {
            'source': 'data/config',
            'description': 'Configuration files and mappings'
        },
        
        # Documentation
        {
            'source': 'docs',
            'description': 'Documentation files',
            'optional': True  # Won't fail if missing
        },
        
        # Tests (important ones only)
        {
            'source': 'tests',
            'description': 'Test files',
            'optional': True
        },
        
        # Scripts directory if exists
        {
            'source': 'scripts',
            'description': 'Utility scripts',
            'optional': True
        },
        
        # Individual important files
        {
            'source': 'interactive_theta_analysis.ipynb',
            'description': 'Interactive analysis notebook',
            'optional': True
        },
    ]
    
    # Track what was backed up
    backup_manifest = {
        'timestamp': timestamp,
        'description': description or f"Code backup created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        'backed_up_items': [],
        'skipped_items': [],
        'total_files': 0,
        'total_size_mb': 0
    }
    
    # Perform backup
    for target in backup_targets:
        source_path = target['source']
        item_description = target['description']
        is_optional = target.get('optional', False)
        
        if verbose:
            print(f"  📋 Processing: {source_path} ({item_description})")
        
        # Check if source exists
        if not os.path.exists(source_path):
            if is_optional:
                # Silently skip optional files that don't exist
                backup_manifest['skipped_items'].append({
                    'path': source_path,
                    'reason': 'Not found (optional)',
                    'description': item_description
                })
                continue
            else:
                if verbose:
                    print(f"    ❌ ERROR: Required path {source_path} not found!")
                backup_manifest['skipped_items'].append({
                    'path': source_path,
                    'reason': 'Not found (required)',
                    'description': item_description
                })
                continue
        
        # Create destination path
        dest_path = os.path.join(backup_path, source_path)
        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)
        
        try:
            if os.path.isfile(source_path):
                # Copy single file
                shutil.copy2(source_path, dest_path)
                file_size = os.path.getsize(source_path)
                
                backup_manifest['backed_up_items'].append({
                    'path': source_path,
                    'type': 'file',
                    'size_bytes': file_size,
                    'description': item_description
                })
                backup_manifest['total_files'] += 1
                backup_manifest['total_size_mb'] += file_size / (1024 * 1024)
                
                if verbose:
                    print(f"    ✅ Copied file: {source_path}")
                    
            elif os.path.isdir(source_path):
                # Copy directory tree
                shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                
                # Count files and calculate size
                file_count = 0
                total_size = 0
                for root, dirs, files in os.walk(dest_path):
                    file_count += len(files)
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
                
                backup_manifest['backed_up_items'].append({
                    'path': source_path,
                    'type': 'directory',
                    'file_count': file_count,
                    'size_bytes': total_size,
                    'description': item_description
                })
                backup_manifest['total_files'] += file_count
                backup_manifest['total_size_mb'] += total_size / (1024 * 1024)
                
                if verbose:
                    print(f"    ✅ Copied directory: {source_path} ({file_count} files)")
                    
        except Exception as e:
            if verbose:
                print(f"    ❌ ERROR copying {source_path}: {e}")
            backup_manifest['skipped_items'].append({
                'path': source_path,
                'reason': f'Copy error: {str(e)}',
                'description': item_description
            })
    
    # Save backup manifest
    manifest_path = os.path.join(backup_path, 'backup_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(backup_manifest, f, indent=2)
    
    # Create backup README
    readme_content = f"""# Code Backup: {backup_name}

**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Description:** {backup_manifest['description']}

## Backup Contents

### Statistics
- **Total files:** {backup_manifest['total_files']}
- **Total size:** {backup_manifest['total_size_mb']:.2f} MB

### Backed up items:
"""
    
    for item in backup_manifest['backed_up_items']:
        if item['type'] == 'file':
            readme_content += f"- 📄 `{item['path']}` - {item['description']}\n"
        else:
            readme_content += f"- 📁 `{item['path']}/` - {item['description']} ({item['file_count']} files)\n"
    
    if backup_manifest['skipped_items']:
        readme_content += "\n### Skipped items:\n"
        for item in backup_manifest['skipped_items']:
            readme_content += f"- ⚠️ `{item['path']}` - {item['reason']}\n"
    
    readme_content += f"""
## Usage

This backup contains the code state before major refactoring.
To restore any file:

```bash
# Copy specific file back
cp {backup_path}/src/core/nm_theta_cross_rats.py src/core/

# Copy entire directory back  
cp -r {backup_path}/src/core/ src/
```

## Backup Manifest

See `backup_manifest.json` for detailed information about this backup.
"""
    
    readme_path = os.path.join(backup_path, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # Final summary
    if verbose:
        print(f"\n✅ Backup completed successfully!")
        print(f"📊 Summary:")
        print(f"   • {backup_manifest['total_files']} files backed up")
        print(f"   • {backup_manifest['total_size_mb']:.2f} MB total size")
        print(f"   • {len(backup_manifest['backed_up_items'])} items backed up")
        print(f"   • {len(backup_manifest['skipped_items'])} items skipped")
        print(f"📁 Backup location: {os.path.abspath(backup_path)}")
        print(f"📋 Manifest: {os.path.abspath(manifest_path)}")
        print(f"📖 README: {os.path.abspath(readme_path)}")
    
    return backup_path, backup_manifest


def list_backups(backup_dir="backups"):
    """
    List all existing backups with their information.
    """
    if not os.path.exists(backup_dir):
        print(f"No backup directory found at: {backup_dir}")
        return
    
    backups = []
    for item in os.listdir(backup_dir):
        item_path = os.path.join(backup_dir, item)
        if os.path.isdir(item_path) and item.startswith('code_backup_'):
            manifest_path = os.path.join(item_path, 'backup_manifest.json')
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    backups.append((item, manifest))
                except:
                    backups.append((item, None))
    
    if not backups:
        print(f"No backups found in: {backup_dir}")
        return
    
    print(f"📁 Found {len(backups)} backups in {backup_dir}:")
    print()
    
    for backup_name, manifest in sorted(backups, reverse=True):
        print(f"🔖 {backup_name}")
        if manifest:
            timestamp_str = datetime.fromisoformat(manifest['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"   📅 Created: {timestamp_str}")
            print(f"   📝 Description: {manifest['description']}")
            print(f"   📊 Files: {manifest['total_files']}, Size: {manifest['total_size_mb']:.2f} MB")
        else:
            print(f"   ⚠️ No manifest found")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Create timestamped backups of important code files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create backup with default settings
  python backup_code.py
  
  # Create backup with custom description
  python backup_code.py --description "Before implementing configuration system"
  
  # Create backup in custom directory
  python backup_code.py --output-dir /path/to/backups
  
  # List existing backups
  python backup_code.py --list
        """
    )
    
    parser.add_argument('--output-dir', default='backups',
                       help='Directory to store backups (default: ./backups)')
    parser.add_argument('--description', 
                       help='Optional description for this backup')
    parser.add_argument('--list', action='store_true',
                       help='List existing backups instead of creating new one')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    if args.list:
        list_backups(args.output_dir)
    else:
        create_code_backup(
            output_dir=args.output_dir,
            description=args.description,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()