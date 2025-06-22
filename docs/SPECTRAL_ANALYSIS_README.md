# Spectral Analysis Pipeline for EEG Near-Mistake Events

This pipeline computes Morlet spectrograms for entire EEG sessions, extracts time-frequency windows around Near-Mistake (NM) and ITI control events, and analyzes event-related spectral changes.

## Overview

The pipeline follows these steps:
1. **Compute full-session Morlet spectrograms** for all 32 channels
2. **Extract time-frequency windows** around NM and ITI events (-2s to +2s by default)
3. **Organize windows** by event type (NM vs ITI) and size (1, 2, 3)
4. **Average spectrograms** within each group
5. **Extract theta power time-courses** and generate plots
6. **Save all intermediate results** for downstream statistical analysis

## Key Features

- ✅ **Memory efficient**: Computes spectrogram on full session, then extracts windows
- ✅ **Edge case handling**: Safely handles events near recording start/end
- ✅ **Modular design**: Each step can be run independently
- ✅ **Comprehensive output**: Both figures and data arrays for statistics
- ✅ **Batch processing**: Process single sessions or entire dataset
- ✅ **ROI support**: Analyze specific channel regions or all channels

## Files Created

### Core Pipeline
- `spectral_analysis_pipeline.py` - Main pipeline functions
- `test_spectral_pipeline.py` - Test and validation scripts
- `batch_spectral_analysis.py` - Batch processing for multiple sessions

### Input Data
- `all_eeg_data.pkl` - Dataset containing ~300 sessions with:
  - `eeg`: (32, n_timepoints) EEG data
  - `eeg_time`: (1, n_timepoints) time vector
  - `nm_peak_times`: (n_nm,) NM event times
  - `nm_sizes`: (n_nm,) NM event sizes (1, 2, or 3)
  - `iti_peak_times`: (n_iti,) ITI event times
  - `iti_sizes`: (n_iti,) ITI event sizes (1, 2, or 3)

## Quick Start

### 1. Test the Pipeline

```python
# Run quick validation test
python test_spectral_pipeline.py

# Or in Python:
from test_spectral_pipeline import quick_test, full_test
quick_test()  # Fast validation
full_test()   # Complete test with plots
```

### 2. Process a Single Session

```python
from spectral_analysis_pipeline import load_session_data, process_single_session

# Load a session
session_data = load_session_data('all_eeg_data.pkl', session_index=0)

# Create session identifier
session_id = f"{session_data['rat_id']}_{session_data['session_date']}"

# Process with default parameters
results = process_single_session(
    session_data=session_data,
    session_id=session_id,
    save_path='results',
    plot_results=True
)
```

### 3. Batch Processing

```python
from batch_spectral_analysis import test_batch_processing, process_all_sessions

# Test on first few sessions
test_batch_processing()

# Process entire dataset (will take several hours)
batch_results = process_all_sessions(
    output_dir='all_sessions_results',
    max_sessions=None  # Set to number for subset
)
```

## Parameters

### Core Analysis Parameters
- `sfreq`: Sampling frequency (default: 200.0 Hz)
- `freqs`: Frequency vector (default: 4-50 Hz)
- `n_cycles`: Morlet wavelet cycles (default: 7)
- `window_duration`: Event window duration (default: 4.0s, i.e., ±2s)
- `theta_band`: Theta frequency band (default: 4-8 Hz)

### Channel Selection
- `channels`: List of channel indices to analyze (default: all 32)
- `roi_channels`: Subset for ROI analysis (default: all selected channels)

### Output Control
- `save_path`: Directory to save results (default: None)
- `plot_results`: Generate plots (default: True)

## Output Structure

### Results Dictionary
```python
results = {
    'session_id': str,
    'freqs': np.ndarray,           # Frequency vector
    'times': np.ndarray,           # Full session time vector
    'channels': List[int],         # Processed channel indices
    'roi_channels': List[int],     # ROI channel indices
    'tfr_full': np.ndarray,        # Full session TFR (channels, freqs, times)
    'group_averages': Dict,        # Average spectrograms per group
    'theta_data': Dict,            # Theta power time-courses
    'parameters': Dict             # Analysis parameters
}
```

### Group Averages Structure
```python
group_averages = {
    'NM': {
        1: {'avg_spectrogram': ndarray, 'window_times': ndarray, 'n_events': int},
        2: {'avg_spectrogram': ndarray, 'window_times': ndarray, 'n_events': int},
        3: {'avg_spectrogram': ndarray, 'window_times': ndarray, 'n_events': int}
    },
    'ITI': {
        1: {'avg_spectrogram': ndarray, 'window_times': ndarray, 'n_events': int},
        2: {'avg_spectrogram': ndarray, 'window_times': ndarray, 'n_events': int},
        3: {'avg_spectrogram': ndarray, 'window_times': ndarray, 'n_events': int}
    }
}
```

### Generated Plots
1. **Average Spectrograms**: Event-aligned spectrograms for each group and channel
2. **Theta Time-courses**: Theta power evolution around events
3. **ROI Averages**: Spatial averages across selected channels

## Example Workflows

### Basic Single Session Analysis
```python
from spectral_analysis_pipeline import *

# Load and process
session_data = load_session_data('all_eeg_data.pkl', 0)
results = process_single_session(session_data, 'test_session')

# Access results
nm_size1_spec = results['group_averages']['NM'][1]['avg_spectrogram']
theta_timecourse = results['theta_data']['NM'][1]['theta_power']
```

### Custom Analysis
```python
# Custom frequency range (focus on theta/alpha)
freqs = np.arange(3, 15, 0.5)

# Subset of channels (e.g., frontal regions)
channels = [0, 1, 2, 3, 8, 9, 10, 11]  # Example frontal channels
roi_channels = [0, 1, 2, 3]  # Subset for ROI

# Longer time window
results = process_single_session(
    session_data=session_data,
    session_id='custom_analysis',
    freqs=freqs,
    channels=channels,
    roi_channels=roi_channels,
    window_duration=6.0,  # ±3 seconds
    theta_band=(3, 7),    # Custom theta band
    save_path='custom_results'
)
```

### Batch Processing with Progress
```python
from batch_spectral_analysis import *

# Get dataset info
info = get_dataset_info()
print(f"Dataset has {info['n_sessions']} sessions")

# Process subset for testing
batch_results = process_session_batch(
    pkl_path='all_eeg_data.pkl',
    session_indices=[0, 1, 2, 3, 4],  # First 5 sessions
    output_dir='test_batch',
    channels=list(range(16)),  # First 16 channels for speed
    plot_results=False  # Skip plots for batch processing
)

# Aggregate results
aggregated = aggregate_batch_results(batch_results)
```

## Expected Processing Times

- **Single session (32 channels)**: ~5-10 minutes
- **Single session (8 channels)**: ~2-3 minutes  
- **Full dataset (300 sessions, 32 channels)**: ~25-50 hours
- **Quick test**: ~30 seconds

## Memory Requirements

- **Single session processing**: ~2-4 GB RAM
- **Batch processing**: ~4-8 GB RAM (loads dataset once)
- **Full dataset in memory**: ~13 GB (not recommended)

## Troubleshooting

### Memory Issues
```python
# Use fewer channels
channels = list(range(8))  # First 8 channels

# Use coarser frequency resolution
freqs = np.arange(4, 50, 2)  # Every 2 Hz instead of 1 Hz

# Process in smaller batches
batch_results = process_session_batch(
    session_indices=list(range(10)),  # Smaller batches
    save_individual=True  # Save each session separately
)
```

### No Events in Some Groups
Some sessions may have no events of certain sizes. The pipeline handles this gracefully:
```python
# Check which groups have data
for event_type in results['group_averages']:
    for size in results['group_averages'][event_type]:
        n_events = results['group_averages'][event_type][size]['n_events']
        print(f"{event_type} size {size}: {n_events} events")
```

### Edge Cases Near Recording Boundaries
Events too close to recording start/end are automatically excluded with warnings:
```python
# Check valid events
nm_valid = results['nm_windows']['valid_events']  # Indices of valid NM events
iti_valid = results['iti_windows']['valid_events']  # Indices of valid ITI events
```

## Next Steps for Statistical Analysis

The pipeline saves all intermediate results for downstream statistical analysis:

1. **Load saved results**: `pickle.load(open('results/analysis_results.pkl', 'rb'))`
2. **Extract power values**: Use `group_averages` for spectrograms, `theta_data` for time-courses
3. **Statistical comparisons**: Compare NM vs ITI for each size group
4. **Time-frequency analysis**: Analyze specific frequency bands and time windows
5. **Channel-wise analysis**: Compare across brain regions

## Dependencies

- `numpy`
- `matplotlib`
- `mne` (for Morlet wavelets)
- `pickle` (built-in)
- Existing `eeg_analysis_package`

## Contact

This pipeline was created for EEG near-mistake event analysis. Modify parameters as needed for your specific research questions.