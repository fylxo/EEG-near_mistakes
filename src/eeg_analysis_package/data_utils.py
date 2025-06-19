import numpy as np
import h5py
import os
import glob
import pickle
from tqdm import tqdm

def ascii_to_str(arr):
    if isinstance(arr, bytes):
        return arr.decode('utf-8')
    if isinstance(arr, np.ndarray):
        # Flatten and cast to uint8 (just in case)
        arr = arr.flatten()
        try:
            return ''.join(chr(c) for c in arr if c != 0)
        except Exception:
            # Fallback for char arrays stored as bytes
            if arr.dtype.char == 'S':
                return b''.join(arr).decode('utf-8')
    return str(arr)

def read_mat_file(file_path):
    with h5py.File(file_path, 'r') as f:
        data_eeg = f['Data_eeg']
        def get_cell(col):
            ref = data_eeg[col, 0]
            arr = f[ref]
            data = arr[()]
            if isinstance(data, bytes):
                return data.decode('utf-8')
            elif isinstance(data, np.ndarray) and data.dtype.char == 'S':
                return b''.join(data).decode('utf-8')
            else:
                return np.array(data)
        # Extract columns
        rat_id = ascii_to_str(get_cell(0))
        session_date = ascii_to_str(get_cell(1))
        eeg = get_cell(2).T  # Transpose!
        eeg_time = get_cell(3).flatten().reshape(1, -1)  # 1D array
        velocity_trace = get_cell(4).flatten()  # 1D array
        velocity_time = get_cell(5).flatten()
        nm_peak_times = get_cell(6).flatten()
        nm_sizes = get_cell(7).flatten()
        iti_peak_times = get_cell(8).flatten()
        iti_sizes = get_cell(9).flatten()
        # Pack result
        return {
            'file_path': file_path,
            'rat_id': rat_id,
            'session_date': session_date,
            'eeg': eeg,
            'eeg_time': eeg_time,
            'velocity_trace': velocity_trace,
            'velocity_time': velocity_time,
            'nm_peak_times': nm_peak_times,
            'nm_sizes': nm_sizes,
            'iti_peak_times': iti_peak_times,
            'iti_sizes': iti_sizes
        }

def debug_print(data):
    np.set_printoptions(suppress=False, precision=4, formatter={'float_kind':'{:0.4e}'.format})
    print(f"Rat ID: {data['rat_id']}")
    print(f"Session Date: {data['session_date']}")
    print(f"EEG shape: {data['eeg'].shape}")
    print(f"EEG time shape: {data['eeg_time'].shape}")
    print(f"Velocity time shape: {data['velocity_time'].shape}")
    print(f"Velocity trace shape: {data['velocity_trace'].shape}")
    print(f"NM peak times shape: {data['nm_peak_times'].shape}")
    print(f"NM sizes shape: {data['nm_sizes'].shape}")
    print(f"ITI peak times shape: {data['iti_peak_times'].shape}")
    print(f"ITI sizes shape: {data['iti_sizes'].shape}")
    print("EEG (ch 1, first 5):", data['eeg'][0, :5])  # channel 1, first 5 times
    print("EEG time (first 5):", data['eeg_time'][:5])
    print("Velocity trace (first 5):")
    print(data['velocity_trace'][:5])
    print("NM sizes (first 10):", data['nm_sizes'][:10])
    print("ITI sizes (first 10):", data['iti_sizes'][:10])

def batch_process_files(mat_dir):
    mat_files = sorted(glob.glob(os.path.join(mat_dir, '*.mat')))
    all_data = []

    for file_path in tqdm(mat_files, desc='Loading MAT files'):
        try:
            data = read_mat_file(file_path)
            all_data.append(data)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    return all_data

def save_data(all_data, filename='all_eeg_data.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(all_data, f)
        
def load_data(filename='all_eeg_data.pkl'):
    with open(filename, 'rb') as f:
        all_data = pickle.load(f)
    return all_data