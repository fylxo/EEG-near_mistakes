#!/usr/bin/env python3
"""
Result Cleanup Script

This script helps manage disk space by cleaning up analysis result folders.
Use with caution - deleted files cannot be recovered.

Usage:
    python scripts/cleanup_results.py --help
    python scripts/cleanup_results.py --list
    python scripts/cleanup_results.py --clean-legacy
    python scripts/cleanup_results.py --clean-single-sessions
    python scripts/cleanup_results.py --clean-all
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def get_directory_size(path: str) -> int:
    """Get total size of directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def find_result_folders() -> Dict[str, List[str]]:
    """Find all result folders categorized by type."""
    root_dir = Path(".")
    
    folders = {
        'legacy_multi_session': [],
        'legacy_single_session': [],
        'legacy_roi_analysis': [],
        'organized_results': [],
        'other_results': []
    }
    
    # Find legacy folders in root directory
    for item in root_dir.iterdir():
        if item.is_dir():
            name = item.name
            if name.startswith('nm_multi_session_'):
                folders['legacy_multi_session'].append(str(item))
            elif name.startswith('single_session_'):
                folders['legacy_single_session'].append(str(item))
            elif name in ['nm_roi_theta_results', 'nm_theta_results']:
                folders['legacy_roi_analysis'].append(str(item))
    
    # Find organized results
    results_dir = root_dir / 'results'
    if results_dir.exists():
        for item in results_dir.rglob('*'):
            if item.is_dir():
                folders['organized_results'].append(str(item))
    
    return folders


def list_result_folders():
    """List all result folders with sizes."""
    print("🔍 Scanning for result folders...\n")
    
    folders = find_result_folders()
    total_size = 0
    
    for category, folder_list in folders.items():
        if not folder_list:
            continue
            
        print(f"📁 {category.replace('_', ' ').title()}:")
        category_size = 0
        
        for folder in sorted(folder_list):
            if os.path.exists(folder):
                size = get_directory_size(folder)
                category_size += size
                print(f"   {folder:<50} {format_size(size):>8}")
        
        if category_size > 0:
            print(f"   {'Subtotal:':<50} {format_size(category_size):>8}")
        print()
        
        total_size += category_size
    
    print(f"🗂️  Total size of all result folders: {format_size(total_size)}")


def confirm_deletion(folder_type: str, folders: List[str]) -> bool:
    """Ask user to confirm deletion."""
    if not folders:
        print(f"No {folder_type} folders found.")
        return False
    
    print(f"\n⚠️  The following {folder_type} folders will be PERMANENTLY DELETED:")
    total_size = 0
    
    for folder in folders:
        if os.path.exists(folder):
            size = get_directory_size(folder)
            total_size += size
            print(f"   {folder} ({format_size(size)})")
    
    print(f"\nTotal space to be freed: {format_size(total_size)}")
    
    response = input(f"\nAre you sure you want to delete these {len(folders)} folders? [y/N]: ")
    return response.lower() in ['y', 'yes']


def clean_folders(folders: List[str], folder_type: str):
    """Clean specified folders."""
    if not folders:
        print(f"No {folder_type} folders to clean.")
        return
    
    if not confirm_deletion(folder_type, folders):
        print("Cleanup cancelled.")
        return
    
    deleted_count = 0
    freed_space = 0
    
    for folder in folders:
        if os.path.exists(folder):
            try:
                size = get_directory_size(folder)
                shutil.rmtree(folder)
                deleted_count += 1
                freed_space += size
                print(f"✅ Deleted: {folder}")
            except Exception as e:
                print(f"❌ Failed to delete {folder}: {e}")
    
    print(f"\n🎉 Cleanup complete!")
    print(f"   Deleted: {deleted_count} folders")
    print(f"   Freed space: {format_size(freed_space)}")


def clean_legacy_folders():
    """Clean all legacy result folders."""
    print("🧹 Cleaning legacy result folders...")
    
    folders = find_result_folders()
    all_legacy = (
        folders['legacy_multi_session'] + 
        folders['legacy_single_session'] + 
        folders['legacy_roi_analysis']
    )
    
    clean_folders(all_legacy, "legacy")


def clean_single_sessions():
    """Clean single session result folders."""
    print("🧹 Cleaning single session result folders...")
    
    folders = find_result_folders()
    clean_folders(folders['legacy_single_session'], "single session")


def clean_multi_sessions():
    """Clean multi-session result folders."""
    print("🧹 Cleaning multi-session result folders...")
    
    folders = find_result_folders()
    clean_folders(folders['legacy_multi_session'], "multi-session")


def clean_roi_analysis():
    """Clean ROI analysis result folders."""
    print("🧹 Cleaning ROI analysis result folders...")
    
    folders = find_result_folders()
    clean_folders(folders['legacy_roi_analysis'], "ROI analysis")


def move_legacy_to_organized():
    """Move legacy folders to organized results structure."""
    print("📦 Moving legacy folders to organized structure...")
    
    folders = find_result_folders()
    
    # Create organized structure if it doesn't exist
    os.makedirs('results/multi_session', exist_ok=True)
    os.makedirs('results/single_sessions', exist_ok=True)
    os.makedirs('results/roi_analysis', exist_ok=True)
    
    moved_count = 0
    
    # Move multi-session folders
    for folder in folders['legacy_multi_session']:
        if os.path.exists(folder):
            folder_name = os.path.basename(folder)
            if 'nm_multi_session_' in folder_name:
                # Extract rat ID from folder name
                parts = folder_name.split('_')
                if 'rat' in parts:
                    rat_idx = parts.index('rat') + 1
                    if rat_idx < len(parts):
                        rat_id = parts[rat_idx]
                        new_name = f"rat_{rat_id}_legacy"
                        new_path = f"results/multi_session/{new_name}"
                        
                        try:
                            shutil.move(folder, new_path)
                            print(f"✅ Moved: {folder} → {new_path}")
                            moved_count += 1
                        except Exception as e:
                            print(f"❌ Failed to move {folder}: {e}")
    
    # Move single session folders
    for folder in folders['legacy_single_session']:
        if os.path.exists(folder):
            folder_name = os.path.basename(folder)
            new_path = f"results/single_sessions/{folder_name}"
            
            try:
                shutil.move(folder, new_path)
                print(f"✅ Moved: {folder} → {new_path}")
                moved_count += 1
            except Exception as e:
                print(f"❌ Failed to move {folder}: {e}")
    
    # Move ROI analysis folders
    for folder in folders['legacy_roi_analysis']:
        if os.path.exists(folder):
            folder_name = os.path.basename(folder)
            new_path = f"results/roi_analysis/{folder_name}"
            
            try:
                shutil.move(folder, new_path)
                print(f"✅ Moved: {folder} → {new_path}")
                moved_count += 1
            except Exception as e:
                print(f"❌ Failed to move {folder}: {e}")
    
    print(f"\n📦 Migration complete! Moved {moved_count} folders.")


def main():
    parser = argparse.ArgumentParser(description='Cleanup analysis result folders')
    parser.add_argument('--list', '-l', action='store_true', 
                       help='List all result folders with sizes')
    parser.add_argument('--clean-legacy', action='store_true',
                       help='Clean all legacy result folders')
    parser.add_argument('--clean-single-sessions', action='store_true',
                       help='Clean single session result folders')
    parser.add_argument('--clean-multi-sessions', action='store_true',
                       help='Clean multi-session result folders')
    parser.add_argument('--clean-roi-analysis', action='store_true',
                       help='Clean ROI analysis result folders')
    parser.add_argument('--clean-all', action='store_true',
                       help='Clean ALL result folders (use with extreme caution)')
    parser.add_argument('--move-legacy', action='store_true',
                       help='Move legacy folders to organized structure')
    
    args = parser.parse_args()
    
    if args.list:
        list_result_folders()
    elif args.clean_legacy:
        clean_legacy_folders()
    elif args.clean_single_sessions:
        clean_single_sessions()
    elif args.clean_multi_sessions:
        clean_multi_sessions()
    elif args.clean_roi_analysis:
        clean_roi_analysis()
    elif args.move_legacy:
        move_legacy_to_organized()
    elif args.clean_all:
        print("⚠️  WARNING: This will delete ALL result folders!")
        folders = find_result_folders()
        all_folders = []
        for folder_list in folders.values():
            all_folders.extend(folder_list)
        clean_folders(all_folders, "ALL")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()