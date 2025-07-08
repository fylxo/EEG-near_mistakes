import scipy.io
import numpy as np

mat_data = scipy.io.loadmat('src/core/Frequency.mat')

if 'frq' in mat_data:
    frequencies = np.squeeze(mat_data['frq'])
    print(f"Shape of frequencies array: {frequencies.shape}")
    print(f"Total number of frequencies: {len(frequencies)}")
    print(f"First 10 frequencies: {frequencies[:10]}")
    np.savetxt('frequencies.txt', frequencies)
    print("Frequencies saved to frequencies.txt")
    # Save every other frequency to a new file, formatted to 2 decimal places
    frequencies_128 = frequencies[::2]
    np.savetxt('frequencies_128.txt', frequencies_128, fmt='%.2f')
    print("Every other frequency saved to frequencies_128.txt with 2 decimal places")
else:
    print("No 'frq' variable found in Frequency.mat")