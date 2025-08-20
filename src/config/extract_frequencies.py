"""
Frequency Data Extraction Utility

Extracts frequency data from MATLAB files and saves in text format.
"""

import scipy.io
import numpy as np
from pathlib import Path


def extract_frequencies_from_mat(mat_file_path, output_dir=None):
    """
    Extract frequency data from MATLAB file and save as text files.
    
    Parameters
    ----------
    mat_file_path : str
        Path to the MATLAB .mat file containing frequency data
    output_dir : str, optional
        Output directory for text files (default: current directory)
        
    Returns
    -------
    dict
        Dictionary containing extracted frequency arrays
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Load MATLAB data
    mat_data = scipy.io.loadmat(mat_file_path)
    
    if 'frq' in mat_data:
        frequencies = np.squeeze(mat_data['frq'])
        
        # Save full frequency array
        full_output = output_dir / 'frequencies.txt'
        np.savetxt(full_output, frequencies)
        
        # Save downsampled frequency array (every other frequency)
        downsampled = frequencies[::2]
        downsampled_output = output_dir / 'frequencies_128.txt'
        np.savetxt(downsampled_output, downsampled, fmt='%.2f')
        
        return {
            'full': frequencies,
            'downsampled': downsampled,
            'files_created': [str(full_output), str(downsampled_output)]
        }
    else:
        raise ValueError("No 'frq' variable found in MATLAB file")


if __name__ == "__main__":
    # Extract frequencies from default location
    mat_file = 'src/core/Frequency.mat'
    try:
        result = extract_frequencies_from_mat(mat_file)
        print(f"Extracted {len(result['full'])} frequencies")
        print(f"Created files: {result['files_created']}")
    except FileNotFoundError:
        print(f"MATLAB file not found: {mat_file}")
    except Exception as e:
        print(f"Error extracting frequencies: {e}")