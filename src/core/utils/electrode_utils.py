import pandas as pd
import numpy as np

# ROI mapping based on the brain region information
# This maps brain region names to their corresponding electrode numbers
# Updated ROI definitions as of latest specification
ROI_MAP = {
    'mPFC': [8, 6, 9, 11],  # medial Prefrontal Cortex
    'motor': [7, 10, 5, 12, 3, 14],  # Motor cortex
    'somatomotor': [4, 13, 2, 15, 23, 24, 1, 16, 25, 26],  # Somatomotor cortex
    'visual': [20, 21, 22, 27, 28, 29, 17, 18, 19, 30, 31, 32],  # Visual cortex
    
    # Keep legacy names for backward compatibility
    'frontal': [8, 6, 9, 11],  # Alias for mPFC
    'ss': [4, 13, 2, 15, 23, 24, 1, 16, 25, 26],  # Alias for somatomotor (somatosensory)
}

def load_electrode_mappings(csv_file='data/config/consistent_electrode_mappings.csv'):
    """
    Load electrode mappings from CSV file into a pandas DataFrame.
    
    Args:
        csv_file (str): Path to the CSV file with electrode mappings
        
    Returns:
        pd.DataFrame: DataFrame with rat_id as index and electrode mappings as columns
    """
    df = pd.read_csv(csv_file, index_col='rat_id')
    return df


def get_channel_indices_from_electrodes(rat_id, electrode_numbers, mapping_df):
    """
    Given a rat_id, a list of electrode numbers (e.g., [2, 29, 31, 32]),
    and the mapping_df, return the 0-based channel indices for the EEG data.
    """
    # Try to match either str or int representation of rat_id
    str_id = str(rat_id)
    int_id = int(rat_id) if isinstance(rat_id, (str, int)) and str(rat_id).isdigit() else None

    if rat_id in mapping_df.index:
        row = mapping_df.loc[rat_id].values
    elif str_id in mapping_df.index:
        row = mapping_df.loc[str_id].values
    elif int_id is not None and int_id in mapping_df.index:
        row = mapping_df.loc[int_id].values
    else:
        raise ValueError(f"Rat ID {rat_id} not found in mapping DataFrame index. Available: {list(mapping_df.index)}")

    row = row[~pd.isna(row)].astype(int)
    indices = [i for i, val in enumerate(row) if val in electrode_numbers]
    return indices


def get_channels(rat_id, roi_or_channels, mapping_df=None, roi_map=None):
    if mapping_df is None:
        mapping_df = load_electrode_mappings()
    if roi_map is None:
        roi_map = ROI_MAP

    # Handle ROI name (string)
    if isinstance(roi_or_channels, str):
        if roi_or_channels not in roi_map:
            raise ValueError(f"ROI '{roi_or_channels}' not in ROI_MAP")
        electrode_numbers = roi_map[roi_or_channels]
        return get_channel_indices_from_electrodes(rat_id, electrode_numbers, mapping_df)
    # Handle explicit electrode numbers
    elif isinstance(roi_or_channels, (list, tuple, set)):
        # If all entries are int and in 1-32, treat as electrode numbers (NOT channel indices!)
        if all(isinstance(x, int) and 1 <= x <= 32 for x in roi_or_channels):
            return get_channel_indices_from_electrodes(rat_id, roi_or_channels, mapping_df)
        else:
            raise ValueError("Custom channel list must be electrode numbers (1-32)")
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