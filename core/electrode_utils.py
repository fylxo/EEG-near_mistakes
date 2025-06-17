import pandas as pd
import numpy as np

# ROI mapping based on the brain region information
# This maps brain region names to their corresponding electrode numbers
ROI_MAP = {
    'frontal': [2, 29, 31, 32],
    'motor': [1, 3, 5, 26, 28, 30],
    'ss': [4, 6, 22, 27],  # somatosensory
    'hippocampus': [7, 11, 15, 16, 17, 18],
    'visual': [8, 9, 10, 12, 13, 14, 19, 20, 21, 23, 24, 25]
}

def load_electrode_mappings(csv_file='consistent_electrode_mappings.csv'):
    """
    Load electrode mappings from CSV file into a pandas DataFrame.
    
    Args:
        csv_file (str): Path to the CSV file with electrode mappings
        
    Returns:
        pd.DataFrame: DataFrame with rat_id as index and electrode mappings as columns
    """
    df = pd.read_csv(csv_file, index_col='rat_id')
    return df

def get_channels(rat_id, roi_or_channels, mapping_df=None, roi_map=None):
    """
    Get channel indices for a given rat_id and ROI or custom channel numbers.

    Args:
        rat_id (str or int): Rat identifier
        roi_or_channels: 
            - If str: a ROI name, e.g. 'frontal'
            - If list of int: channel numbers (1-based, as in roi_map) or direct indices (0-based)
        mapping_df (pd.DataFrame, optional): DataFrame with electrode mappings. 
                                           If None, loads from default CSV file.
        roi_map (dict, optional): ROI to electrode number mapping. If None, uses default ROI_MAP.
        
    Returns:
        List of channel indices (0-based) for use with EEG data.
    """
    # Load default data if not provided
    if mapping_df is None:
        mapping_df = load_electrode_mappings()
    if roi_map is None:
        roi_map = ROI_MAP
    
    # Handle both string and integer rat IDs
    if rat_id not in mapping_df.index:
        # Try converting to int if it's a string
        if isinstance(rat_id, str) and rat_id.isdigit():
            rat_id = int(rat_id)
        # Try converting to string if it's an int
        elif isinstance(rat_id, int):
            rat_id_str = str(rat_id)
            if rat_id_str in mapping_df.index:
                rat_id = rat_id_str
        
        # If still not found, raise error
        if rat_id not in mapping_df.index:
            available_rats = list(mapping_df.index)
            raise ValueError(f"Rat ID {rat_id} not found in mapping! Available rats: {available_rats}")

    mapping_row = mapping_df.loc[rat_id].values
    # Remove NaN values and convert to int
    mapping_row = mapping_row[~pd.isna(mapping_row)].astype(int)
    
    if isinstance(roi_or_channels, str):
        # ROI name
        roi_name = roi_or_channels.lower()
        if roi_name not in roi_map:
            raise ValueError(f"ROI '{roi_name}' not in roi_map keys: {list(roi_map.keys())}")
        roi_numbers = set(roi_map[roi_name])
        indices = [i for i, val in enumerate(mapping_row) if val in roi_numbers]
        return indices
    elif isinstance(roi_or_channels, (list, tuple, set)):
        # Custom list of numbers (could be channel numbers or indices)
        if all(isinstance(x, int) for x in roi_or_channels):
            # Are these actual mapping numbers (e.g. 1-32), or 0-based indices?
            if all((1 <= x <= 32) for x in roi_or_channels):
                # Assume channel numbers, map to indices
                indices = [i for i, val in enumerate(mapping_row) if val in roi_or_channels]
                return indices
            elif all((0 <= x < len(mapping_row)) for x in roi_or_channels):
                # Direct indices
                return list(roi_or_channels)
            else:
                raise ValueError(f"List must be all 0-{len(mapping_row)-1} (indices) or 1-32 (channel numbers)")
        else:
            raise ValueError("Custom channel list must be integers")
    else:
        raise ValueError("roi_or_channels must be a string or a list/tuple/set of ints")

def get_roi_info(rat_id, mapping_df=None, roi_map=None):
    """
    Get information about which channels belong to which ROIs for a specific rat.
    
    Args:
        rat_id (str or int): Rat identifier
        mapping_df (pd.DataFrame, optional): DataFrame with electrode mappings
        roi_map (dict, optional): ROI to electrode number mapping
        
    Returns:
        dict: ROI names as keys, channel indices as values
    """
    if mapping_df is None:
        mapping_df = load_electrode_mappings()
    if roi_map is None:
        roi_map = ROI_MAP
    
    roi_info = {}
    for roi_name in roi_map.keys():
        try:
            indices = get_channels(rat_id, roi_name, mapping_df, roi_map)
            if indices:  # Only include ROIs that have channels
                roi_info[roi_name] = indices
        except ValueError:
            continue  # Skip ROIs that don't have channels for this rat
    
    return roi_info

if __name__ == "__main__":
    # Example usage
    try:
        # Load electrode mappings
        mapping_df = load_electrode_mappings()
        print(f"Loaded electrode mappings for {len(mapping_df)} rats")
        print(f"Available rats: {list(mapping_df.index)}")
        
        # Example with a specific rat
        rat_id = 10501  # Use integer, not string
        if rat_id in mapping_df.index:
            print(f"\nExample usage for rat {rat_id}:")
            
            # Show the actual electrode mapping for this rat
            actual_mapping = mapping_df.loc[rat_id].values
            actual_mapping = actual_mapping[~pd.isna(actual_mapping)].astype(int)
            print(f"Actual electrode mapping: {actual_mapping}")
            
            # Test with electrode numbers that actually exist in the data
            print(f"\nTesting with actual electrode numbers:")
            
            # Custom channel numbers (using numbers that exist in the mapping)
            custom_channels = [10, 11, 12]  # These are in the mapping
            try:
                custom_indices = get_channels(rat_id, custom_channels, mapping_df)
                print(f"Custom channels {custom_channels} -> indices: {custom_indices}")
            except Exception as e:
                print(f"Custom channels error: {e}")
            
            # Direct indices
            direct_indices = [0, 4, 9]
            try:
                result_indices = get_channels(rat_id, direct_indices, mapping_df)
                print(f"Direct indices {direct_indices} -> {result_indices}")
            except Exception as e:
                print(f"Direct indices error: {e}")
            
            # Try ROI mapping with correct brain regions
            print(f"\nTesting ROI mapping:")
            roi_info = get_roi_info(rat_id, mapping_df)
            if roi_info:
                print("ROI information:", roi_info)
                
                # Test individual ROIs
                for roi_name in ['frontal', 'motor', 'visual', 'hippocampus']:
                    try:
                        indices = get_channels(rat_id, roi_name, mapping_df)
                        print(f"{roi_name.capitalize()} channels: {indices}")
                    except Exception as e:
                        print(f"{roi_name.capitalize()} error: {e}")
            else:
                print("No ROIs found")
        
    except FileNotFoundError:
        print("Electrode mappings CSV file not found. Run roi_mapping.py first.")
    except Exception as e:
        print(f"Error: {e}") 