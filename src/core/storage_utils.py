#!/usr/bin/env python3
"""
Storage utilities for managing EEG analysis results across different drives.

This module provides functions to manage storage locations and move results
between drives based on available space.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Storage locations with their typical free space
STORAGE_LOCATIONS = {
    'c_drive': {
        'path': '/mnt/c/Users/flavi/Desktop/eeg-near_mistakes/results',
        'description': 'C: drive (limited space)',
        'recommended_for': 'small results, temporary files'
    },
    'd_drive': {
        'path': '/mnt/d/nm_theta_results',
        'description': 'D: drive (471 GB available)',
        'recommended_for': 'large analysis results, long-term storage'
    },
    'scratch': {
        'path': '/tmp/nm_theta_scratch',
        'description': 'Temporary scratch space',
        'recommended_for': 'temporary processing files'
    }
}

def get_disk_usage(path: str) -> Tuple[int, int, int]:
    """
    Get disk usage statistics for a given path.
    
    Parameters:
    -----------
    path : str
        Path to check disk usage
        
    Returns:
    --------
    Tuple[int, int, int]
        (total, used, free) space in bytes
    """
    try:
        statvfs = os.statvfs(path)
        total = statvfs.f_frsize * statvfs.f_blocks
        free = statvfs.f_frsize * statvfs.f_bavail
        used = total - free
        return total, used, free
    except Exception as e:
        print(f"Error getting disk usage for {path}: {e}")
        return 0, 0, 0

def format_bytes(bytes_value: int) -> str:
    """Format bytes as human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def check_storage_locations() -> Dict:
    """
    Check available space in all configured storage locations.
    
    Returns:
    --------
    Dict
        Dictionary with storage info for each location
    """
    storage_info = {}
    
    for location_name, location_config in STORAGE_LOCATIONS.items():
        path = location_config['path']
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        total, used, free = get_disk_usage(path)
        
        storage_info[location_name] = {
            'path': path,
            'description': location_config['description'],
            'recommended_for': location_config['recommended_for'],
            'total': total,
            'used': used,
            'free': free,
            'total_human': format_bytes(total),
            'used_human': format_bytes(used),
            'free_human': format_bytes(free),
            'exists': os.path.exists(path),
            'writable': os.access(path, os.W_OK) if os.path.exists(path) else False
        }
    
    return storage_info

def recommend_storage_location(estimated_size_gb: float = 10.0) -> str:
    """
    Recommend the best storage location based on estimated file size.
    
    Parameters:
    -----------
    estimated_size_gb : float
        Estimated size of files to store in GB
        
    Returns:
    --------
    str
        Recommended storage location name
    """
    storage_info = check_storage_locations()
    estimated_size_bytes = estimated_size_gb * 1024**3
    
    # Find locations with enough free space
    suitable_locations = []
    for location_name, info in storage_info.items():
        if info['free'] > estimated_size_bytes * 1.2:  # 20% buffer
            suitable_locations.append((location_name, info))
    
    if not suitable_locations:
        print(f"⚠️  Warning: No storage location has enough space for {estimated_size_gb} GB")
        return 'd_drive'  # Default to D: drive
    
    # Prefer D: drive for large files, C: drive for small files
    if estimated_size_gb > 5.0:
        for location_name, info in suitable_locations:
            if location_name == 'd_drive':
                return location_name
    else:
        for location_name, info in suitable_locations:
            if location_name == 'c_drive':
                return location_name
    
    # Return the first suitable location
    return suitable_locations[0][0]

def get_storage_path(location_name: str = None, estimated_size_gb: float = 10.0) -> str:
    """
    Get the storage path for a given location or recommended location.
    
    Parameters:
    -----------
    location_name : str, optional
        Name of storage location, None for automatic recommendation
    estimated_size_gb : float
        Estimated size of files to store in GB
        
    Returns:
    --------
    str
        Full path to storage location
    """
    if location_name is None:
        location_name = recommend_storage_location(estimated_size_gb)
    
    if location_name not in STORAGE_LOCATIONS:
        raise ValueError(f"Unknown storage location: {location_name}. "
                        f"Available: {list(STORAGE_LOCATIONS.keys())}")
    
    path = STORAGE_LOCATIONS[location_name]['path']
    os.makedirs(path, exist_ok=True)
    return path

def move_results(source_path: str, destination_location: str = 'd_drive') -> str:
    """
    Move analysis results from one location to another.
    
    Parameters:
    -----------
    source_path : str
        Path to source directory/file
    destination_location : str
        Name of destination storage location
        
    Returns:
    --------
    str
        Path to moved results
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    
    dest_base = get_storage_path(destination_location)
    source_name = os.path.basename(source_path)
    dest_path = os.path.join(dest_base, source_name)
    
    print(f"Moving {source_path} to {dest_path}")
    
    if os.path.isdir(source_path):
        shutil.move(source_path, dest_path)
    else:
        shutil.move(source_path, dest_path)
    
    print(f"✓ Results moved to: {dest_path}")
    return dest_path

def print_storage_summary():
    """Print a summary of all storage locations and their status."""
    storage_info = check_storage_locations()
    
    print("\n📁 Storage Locations Summary")
    print("=" * 60)
    
    for location_name, info in storage_info.items():
        status = "✓" if info['exists'] and info['writable'] else "❌"
        print(f"{status} {location_name.upper()}")
        print(f"    Path: {info['path']}")
        print(f"    Description: {info['description']}")
        print(f"    Free space: {info['free_human']}")
        print(f"    Recommended for: {info['recommended_for']}")
        print()

def create_storage_index(results_path: str):
    """
    Create an index file to track where results are stored.
    
    Parameters:
    -----------
    results_path : str
        Path where results are stored
    """
    index_file = os.path.join(results_path, 'storage_index.json')
    
    index_data = {
        'storage_location': results_path,
        'created_at': os.path.getctime(results_path) if os.path.exists(results_path) else None,
        'last_modified': os.path.getmtime(results_path) if os.path.exists(results_path) else None,
        'storage_info': check_storage_locations()
    }
    
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Storage utilities for EEG analysis results')
    parser.add_argument('--check', action='store_true', help='Check storage locations')
    parser.add_argument('--recommend', type=float, help='Recommend location for given size (GB)')
    parser.add_argument('--move', nargs=2, metavar=('SOURCE', 'DEST_LOCATION'), 
                       help='Move files from source to destination location')
    
    args = parser.parse_args()
    
    if args.check:
        print_storage_summary()
    elif args.recommend is not None:
        location = recommend_storage_location(args.recommend)
        path = get_storage_path(location)
        print(f"Recommended location for {args.recommend} GB: {location}")
        print(f"Path: {path}")
    elif args.move:
        source, dest_location = args.move
        dest_path = move_results(source, dest_location)
        print(f"Moved to: {dest_path}")
    else:
        print_storage_summary()