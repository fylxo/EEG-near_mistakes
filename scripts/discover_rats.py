#!/usr/bin/env python3
"""
Rat Discovery Script for SLURM Array Jobs

This script discovers available rat IDs from the EEG data file and creates
the necessary configuration files for SLURM array job submission.
"""

import os
import sys
import pickle
import json
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def discover_rats(pkl_path: str) -> List[str]:
    """
    Discover all available rat IDs from the EEG data file.
    
    Parameters:
    -----------
    pkl_path : str
        Path to the main EEG data file
        
    Returns:
    --------
    List[str]
        List of available rat IDs
    """
    print(f"Loading data from: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    
    rat_ids = set()
    for session_data in all_data:
        rat_id = session_data.get('rat_id')
        if rat_id is not None:
            rat_ids.add(str(rat_id))
    
    rat_ids_list = sorted(list(rat_ids))
    print(f"Found {len(rat_ids_list)} rats: {rat_ids_list}")
    
    return rat_ids_list

def create_rat_config(rat_ids: List[str], output_file: str = 'rat_config.json'):
    """
    Create a configuration file with rat IDs and array job parameters.
    
    Parameters:
    -----------
    rat_ids : List[str]
        List of rat IDs
    output_file : str
        Path to save the configuration file
    """
    config = {
        'rat_ids': rat_ids,
        'n_rats': len(rat_ids),
        'array_range': f"1-{len(rat_ids)}",
        'rat_id_map': {str(i+1): rat_id for i, rat_id in enumerate(rat_ids)}
    }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {output_file}")
    return config

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover rats and create SLURM array job configuration')
    parser.add_argument('--pkl_path', required=True, help='Path to the main EEG data file')
    parser.add_argument('--output', default='rat_config.json', help='Output configuration file')
    parser.add_argument('--exclude_9442', action='store_true', help='Exclude rat 9442 (compatibility issues)')
    
    args = parser.parse_args()
    
    # Discover rats
    rat_ids = discover_rats(args.pkl_path)
    
    # Optionally exclude rat 9442
    if args.exclude_9442 and '9442' in rat_ids:
        rat_ids.remove('9442')
        print(f"Excluded rat 9442. Remaining: {len(rat_ids)} rats")
    
    # Create configuration
    config = create_rat_config(rat_ids, args.output)
    
    print(f"\nArray job configuration:")
    print(f"  Array range: {config['array_range']}")
    print(f"  Total jobs: {config['n_rats']}")
    print(f"  Rat ID mapping: {config['rat_id_map']}")

if __name__ == "__main__":
    main()