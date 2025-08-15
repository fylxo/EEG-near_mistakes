import h5py
import numpy as np
import csv

# File paths - Use project root relative paths
import os
import sys

# Get project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

INPUT_FILE = os.path.join(project_root, 'data', 'config', 'electrodes_placement.mat')
OUTPUT_FILE = os.path.join(project_root, 'data', 'config', 'consistent_electrode_mappings.csv')

def ascii_to_str(arr):
    """Convert HDF5 ASCII/char/bytes arrays to string."""
    if isinstance(arr, bytes):
        return arr.decode('utf-8')
    if isinstance(arr, np.ndarray):
        if arr.dtype.char == 'S':
            return b''.join(arr.flatten()).decode('utf-8')
        try:
            # If it's uint16 or uint8 char array
            return ''.join(chr(c) for c in arr.flatten() if c != 0)
        except Exception:
            return str(arr)
    return str(arr)

def extract_cell(f, cell):
    """Dereference and extract the actual value."""
    if isinstance(cell, h5py.Reference):
        value = f[cell][()]
        return extract_cell(f, value)
    elif isinstance(cell, np.ndarray):
        if cell.dtype == h5py.ref_dtype:
            # Handle array of references
            if cell.size == 1:
                # Single reference
                return extract_cell(f, cell.item())
            else:
                # Multiple references - dereference each
                result = []
                for ref in cell.flatten():
                    dereferenced = extract_cell(f, ref)
                    if isinstance(dereferenced, np.ndarray):
                        result.extend(dereferenced.flatten())
                    else:
                        result.append(dereferenced)
                return np.array(result)
        else:
            return cell
    else:
        return cell

def process_mapping(mapping_raw):
    """Process mapping data to extract numeric electrode mapping."""
    try:
        # If it's already a numpy array with numbers, use it
        if isinstance(mapping_raw, np.ndarray) and np.issubdtype(mapping_raw.dtype, np.number):
            return mapping_raw.flatten()
        
        # If it's a list or other sequence, try to convert
        if hasattr(mapping_raw, '__iter__') and not isinstance(mapping_raw, (str, bytes)):
            try:
                # Try to convert to numeric array
                arr = np.array(mapping_raw, dtype=float)
                return arr.flatten()
            except (ValueError, TypeError):
                # If conversion fails, try element by element
                numeric_values = []
                for item in mapping_raw:
                    if isinstance(item, (int, float, np.integer, np.floating)):
                        numeric_values.append(float(item))
                    elif isinstance(item, np.ndarray) and item.size == 1:
                        numeric_values.append(float(item.item()))
                if numeric_values:
                    return np.array(numeric_values)
        
        return None
    except Exception as e:
        print(f"Error processing mapping: {e}")
        return None

if __name__ == "__main__":
    # Open .mat file
    with h5py.File(INPUT_FILE, 'r') as f:
        eeg_channels = f['eeg_channels']
        nrows, ncols = eeg_channels.shape
        print(f"Data structure: {nrows} rows x {ncols} columns")
        
        mapping_dict = {}
        electrode_counts = {}

        # Structure is 4 rows x 306 columns
        # Row 0: Rat IDs
        # Row 1: Session IDs  
        # Row 2: Electrode mappings (32x1 double)
        # Row 3: Brain region strings (32x1 cell)
        
        for col in range(ncols):  # Iterate through columns (sessions)
            # Extract data for this session
            rat_id_raw = extract_cell(f, eeg_channels[0, col])      # Row 0: Rat ID
            session_id_raw = extract_cell(f, eeg_channels[1, col])  # Row 1: Session ID  
            mapping_raw = extract_cell(f, eeg_channels[2, col])     # Row 2: 32x1 electrode mapping
            # eeg_channels[3, col] would be the brain region strings (ignored)

            # Convert rat and session IDs to string
            rat_id = ascii_to_str(rat_id_raw)
            session_id = ascii_to_str(session_id_raw)
            
            # Process mapping - should be 32x1 double
            mapping = process_mapping(mapping_raw)
            
            if mapping is not None and mapping.size == 32:
                mapping_list = mapping.astype(int).tolist()
                if rat_id not in mapping_dict:
                    mapping_dict[rat_id] = {}
                mapping_dict[rat_id][session_id] = mapping_list
                if rat_id not in electrode_counts:
                    electrode_counts[rat_id] = set()
                electrode_counts[rat_id].add(mapping.size)
            else:
                size_info = f"{mapping.size}" if mapping is not None else "unknown"
                print(f"Skipping {rat_id} {session_id}: mapping is not 32 electrodes (found {size_info})")

    # Report electrode count variations
    print(f"\n=== SUMMARY ===")
    print(f"Total sessions processed: {ncols}")
    print(f"Rats with consistent 32-electrode mappings: {len(mapping_dict)}")

    print(f"\nRat IDs found: {sorted(mapping_dict.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))}")

    print(f"\nElectrode count per rat:")
    for rat_id in sorted(mapping_dict.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
        sessions_count = len(mapping_dict[rat_id])
        print(f"  Rat {rat_id}: {sessions_count} sessions with 32 electrodes")

    # Consistency check
    final_mapping = {}
    inconsistent_rats = []
    variable_electrode_rats = []

    for rat_id, sessions in mapping_dict.items():
        mappings = list(sessions.values())
        first_mapping = mappings[0]
        
        # Check if all mappings have the same length
        same_length = all(len(m) == len(first_mapping) for m in mappings)
        if not same_length:
            variable_electrode_rats.append(rat_id)
            print(f"Rat {rat_id} has variable electrode counts across sessions.")
            continue
        
        # Check if all mappings are identical
        all_equal = all(np.array_equal(first_mapping, m) for m in mappings)
        if all_equal:
            final_mapping[rat_id] = first_mapping
        else:
            print(f"Rat {rat_id} has inconsistent mappings across sessions.")
            inconsistent_rats.append(rat_id)

    # Determine maximum number of electrodes for CSV header
    max_electrodes = max([len(mapping) for mapping in final_mapping.values()]) if final_mapping else 32

    # Save as CSV
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['rat_id'] + [f'ch_{i}' for i in range(max_electrodes)]
        writer.writerow(header)
        
        for rat_id, mapping in final_mapping.items():
            # Pad mapping with NaN if it has fewer electrodes than max
            padded_mapping = mapping + [np.nan] * (max_electrodes - len(mapping))
            writer.writerow([rat_id] + padded_mapping)

    print(f"\nSaved {len(final_mapping)} consistent rat mappings to '{OUTPUT_FILE}'")
    print(f"Maximum electrodes per rat: {max_electrodes}")

    if inconsistent_rats:
        print("Rats with inconsistent mappings:", inconsistent_rats)
    if variable_electrode_rats:
        print("Rats with variable electrode counts:", variable_electrode_rats)
