# Complete Technical Analysis of nm_theta_cross_rats.py

## Overview

This document provides a comprehensive technical analysis of the `nm_theta_cross_rats.py` script, which implements cross-rats aggregation for Near-Mistake (NM) theta analysis using pre-event baseline normalization.

## Table of Contents

1. [Core Purpose & Design Philosophy](#core-purpose--design-philosophy)
2. [Normalization Strategy](#normalization-strategy)
3. [Function Patching System](#function-patching-system)
4. [Advanced Baseline Normalization Logic](#advanced-baseline-normalization-logic)
5. [Frequency Management](#frequency-management)
6. [Rat 9442 Special Handling](#rat-9442-special-handling)
7. [Rat Discovery & Validation](#rat-discovery--validation)
8. [Multi-Session Processing](#multi-session-processing)
9. [Cross-Rats Aggregation](#cross-rats-aggregation)
10. [Advanced Visualization](#advanced-visualization)
11. [Interactive Features](#interactive-features)
12. [Comprehensive Error Handling & Resilience](#comprehensive-error-handling--resilience)
13. [Configuration Integration](#configuration-integration)
14. [Main Execution Modes](#main-execution-modes)
15. [Key Strengths](#key-strengths)
16. [Pipeline Validation Perspective](#pipeline-validation-perspective)

---

## 1. Core Purpose & Design Philosophy

The script implements a **cross-rats aggregation pipeline** for Near-Mistake (NM) theta analysis using **pre-event baseline normalization**. This is the main analysis method that:

- **Processes multiple rats' EEG data simultaneously**: Enables population-level statistical analysis
- **Uses -1.0 to -0.5 seconds before each NM event as the baseline**: Provides consistent normalization reference
- **Aggregates results across rats**: Finds population-level theta power differences between NM types
- **Implements proper z-score normalization**: Uses cross-event statistics rather than single-event variance

### Design Philosophy

The script follows several key design principles:

1. **Methodological rigor**: Implements established EEG analysis best practices
2. **Robustness**: Handles failures gracefully with comprehensive error recovery
3. **Flexibility**: Supports multiple frequency specifications and ROI definitions
4. **Transparency**: Provides detailed logging and metadata tracking
5. **Memory efficiency**: Manages memory usage through cleanup and garbage collection

---

## 2. Normalization Strategy (Lines 3-16, 129-225)

**Pre-event baseline normalization** is the key methodological differentiator from global normalization approaches:

### Baseline Period Definition
- **Time window**: -1.0 to -0.5 seconds before each NM event
- **Rationale**: This period represents "neutral" brain activity before the cognitive event
- **Advantage**: Removes inter-subject and inter-session baseline differences

### Z-Score Calculation Method
```python
z_score = (power - baseline_mean) / baseline_std
```

Where:
- `power`: Spectral power at each time-frequency point
- `baseline_mean`: Mean power across all events' baseline periods
- `baseline_std`: Standard deviation across all events' baseline periods

### Key Advantages Over Global Normalization

1. **Event-specific reference**: Each event is normalized relative to its own pre-event state
2. **Removes oscillatory variance**: Averages baseline periods to focus on power changes
3. **Preserves temporal dynamics**: Maintains time-locked responses to events
4. **Population consistency**: Enables meaningful cross-subject comparisons

---

## 3. Function Patching System (Lines 52-127)

The script uses a **sophisticated patching mechanism** to override core analysis functions temporarily:

### Main Patching Function: `run_analysis_baseline()`

```python
def run_analysis_baseline(*args, **kwargs):
    # Patch core functions to use baseline normalization
    patch_core_functions_for_baseline()
    
    # Force baseline normalization flag
    kwargs['force_baseline_normalization'] = True
    
    try:
        # Call original function with patched core functions
        return original_run_analysis(*args, **kwargs)
    finally:
        # Restore original functions
        restore_original_functions()
```

### Patched Functions

1. **`compute_baseline_statistics_wrapper()`** (Lines 129-138):
   - Returns dummy global statistics
   - Real baseline computation happens in normalization step
   - Maintains interface compatibility

2. **`normalize_windows_baseline_wrapper()`** (Lines 141-225):
   - Implements proper baseline z-score normalization
   - Core function for the entire normalization process
   - Handles missing baseline periods gracefully

3. **`extract_nm_event_windows_baseline_wrapper()`** (Lines 227-243):
   - Adds time vectors for baseline period identification
   - Ensures window_times are available for normalization
   - Maintains original extraction functionality

### Patching Strategy Benefits

- **Non-invasive**: Doesn't modify original code files
- **Reversible**: Automatically restores original functions
- **Compatible**: Works with existing analysis pipeline
- **Testable**: Can be easily validated against original methods

---

## 4. Advanced Baseline Normalization Logic (Lines 141-225)

The `normalize_windows_baseline_wrapper()` function implements the core normalization algorithm:

### Step 1: Baseline Period Extraction (Lines 168-189)

```python
# Extract baseline period (-1.0 to -0.5 seconds)
baseline_mask = (window_times >= -1.0) & (window_times <= -0.5)

baseline_averages = []  # Will be (n_events, n_freqs)
for event_idx in range(n_events):
    event_window = windows[event_idx]  # (n_freqs, n_times)
    baseline_data = event_window[:, baseline_mask]  # (n_freqs, baseline_times)
    baseline_avg = np.mean(baseline_data, axis=1)  # (n_freqs,) - single value per freq
    baseline_averages.append(baseline_avg)
```

**Key aspects**:
- **Time masking**: Identifies exact baseline period in each event window
- **Temporal averaging**: Reduces oscillatory variance within baseline periods
- **Frequency preservation**: Maintains spectral resolution across frequency bands

### Step 2: Cross-Event Statistics Computation (Lines 192-196)

```python
# Compute statistics across events (proper z-score denominator)
baseline_averages = np.array(baseline_averages)  # (n_events, n_freqs)
baseline_mean_across_events = np.mean(baseline_averages, axis=0)  # (n_freqs,)
baseline_std_across_events = np.std(baseline_averages, axis=0)     # (n_freqs,)
baseline_std_across_events = np.maximum(baseline_std_across_events, 1e-12)  # Avoid division by zero
```

**Statistical rationale**:
- **Cross-event mean**: Represents typical baseline power for each frequency
- **Cross-event standard deviation**: Captures natural variability in baseline power
- **Numerical stability**: Prevents division by zero with minimum threshold

### Step 3: Z-Score Normalization Application (Lines 208-210)

```python
# Apply z-score normalization to each event
for event_idx in range(n_events):
    event_window = windows[event_idx]  # (n_freqs, n_times)
    # Z-score: (event - baseline_mean) / baseline_std_across_events
    normalized_event = (event_window - mean_expanded) / std_expanded
    normalized_events.append(normalized_event)
```

**Implementation details**:
- **Broadcasting**: Expands baseline statistics to match event window dimensions
- **Event-wise processing**: Normalizes each event using population statistics
- **Preservation**: Maintains original data structure and metadata

### Error Handling and Fallbacks

The function includes several fallback mechanisms:

1. **Missing baseline periods**: Falls back to global normalization
2. **Dimension mismatches**: Creates fallback time vectors
3. **Zero variance**: Applies minimum threshold to prevent division errors
4. **Invalid data**: Graceful handling with warning messages

---

## 5. Frequency Management (Lines 339-444)

The script implements a **dual frequency system** supporting both file-based and generated frequencies:

### File-Based Frequency Loading

```python
def load_frequencies_from_file(freq_file_path: str) -> np.ndarray:
    frequencies = []
    with open(freq_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    freq = float(line)
                    frequencies.append(freq)
                except ValueError:
                    print(f"Warning: Skipping invalid frequency value: {line}")
    return np.array(frequencies)
```

### Enhanced Frequency Filtering

The `get_frequencies()` function implements sophisticated frequency selection:

1. **Main frequency extraction**:
   ```python
   main_mask = (all_frequencies >= freq_min) & (all_frequencies <= freq_max)
   main_frequencies = all_frequencies[main_mask]
   ```

2. **Edge frequency inclusion**:
   ```python
   # Include frequency just below freq_min
   below_mask = all_frequencies < freq_min
   if np.any(below_mask):
       closest_below_idx = np.where(below_mask)[0][-1]
       closest_below = all_frequencies[closest_below_idx]
       extended_frequencies = np.concatenate([[closest_below], extended_frequencies])
   ```

3. **Coverage optimization**:
   - Includes frequencies slightly outside requested range
   - Ensures proper spectroscopic coverage
   - Reports effective vs requested frequency ranges

### Generated Frequency Option

For cases without frequency files:
```python
frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
```

**Benefits of dual system**:
- **Precision**: File-based frequencies allow exact frequency control
- **Flexibility**: Generated frequencies work without pre-computed frequency lists
- **Compatibility**: Supports both experimental and exploratory analysis modes

---

## 6. Rat 9442 Special Handling (Lines 447-753)

Rat 9442 requires special handling due to **mixed electrode configurations** across sessions:

### Session Type Classification

```python
RAT_9442_32_CHANNEL_SESSIONS = ['070419', '080419', '090419', '190419']
RAT_9442_20_CHANNEL_ELECTRODES = [10, 11, 12, 13, 14, 15, 16, 19, 1, 24, 25, 29, 2, 3, 4, 5, 6, 7, 8, 9]
```

### Mapping Strategy

```python
def get_rat_9442_mapping_for_session(session_id: str, mapping_df: pd.DataFrame) -> str:
    if session_id in RAT_9442_32_CHANNEL_SESSIONS:
        return '9151'  # Use rat 9151's mapping for 32-channel sessions
    else:
        return '9442'  # Use rat 9442's mapping for 20-channel sessions
```

### Compatibility Checking

The `check_rat_9442_compatibility()` function:

1. **Electrode validation**:
   ```python
   requested_electrodes_set = set(requested_electrodes)
   available_electrodes_set = set(available_electrodes)
   missing_electrodes = requested_electrodes_set - available_electrodes_set
   ```

2. **Automatic exclusion**:
   - Removes rat 9442 if requested electrodes not available
   - Provides detailed compatibility reporting
   - Prevents analysis failures due to missing electrodes

### Benefits of Special Handling

- **Data integrity**: Ensures only valid electrode combinations are analyzed
- **Automatic validation**: Prevents runtime errors from electrode mismatches
- **Transparent reporting**: Clear communication of inclusion/exclusion decisions

---

## 7. Rat Discovery & Validation (Lines 578-663)

The `discover_rat_ids()` function implements comprehensive rat discovery and validation:

### Dataset Scanning

```python
with open(pkl_path, 'rb') as f:
    all_data = pickle.load(f)

rat_ids = set()
for session_data in all_data:
    rat_id = session_data.get('rat_id')
    if rat_id is not None:
        rat_ids.add(str(rat_id))
```

### Compatibility Assessment

For each discovered rat, the function:

1. **Checks electrode availability**: Verifies requested electrodes exist
2. **Validates ROI compatibility**: Ensures ROI mapping is possible
3. **Applies exclusion rules**: Removes incompatible rats automatically

### Reporting and Transparency

```python
print(f"📊 Final rat selection:")
print(f"  Total rats found: {len(rat_ids_list) + len(excluded_rats)}")
print(f"  Rats to process: {len(rat_ids_list)}")
if excluded_rats:
    print(f"  Excluded rats: {excluded_rats}")
```

**Benefits**:
- **Automatic discovery**: No manual rat list maintenance required
- **Intelligent filtering**: Removes problematic rats before analysis
- **Complete transparency**: Full reporting of inclusion/exclusion decisions

---

## 8. Multi-Session Processing (Lines 755-880)

The `process_single_rat_multi_session()` function handles individual rat processing:

### Analysis Pipeline Integration

```python
if method == 'mne':
    results = run_analysis_baseline(
        mode='multi',
        method='basic',
        parallel_type=None,
        pkl_path=pkl_path,
        roi=roi_channels,
        # ... other parameters
    )
```

### Error Handling and Resilience

```python
try:
    # Process rat with baseline normalization
    return rat_id, results
except Exception as e:
    if verbose:
        print(f"❌ Error processing rat {rat_id}: {str(e)}")
        import traceback
        traceback.print_exc()
    return rat_id, None
```

### Resource Management

```python
# Cleanup intermediate session files if requested
if cleanup_intermediate_files:
    cleanup_session_folders(rat_save_path, verbose=verbose)

# Force garbage collection
gc.collect()
```

**Key features**:
- **Built-in resilience**: Handles session-level failures automatically
- **Memory management**: Cleans up intermediate files and forces garbage collection
- **Detailed logging**: Provides comprehensive progress reporting

---

## 9. Cross-Rats Aggregation (Lines 882-1058)

The `aggregate_cross_rats_results()` function implements the final cross-rats aggregation:

### Data Collection Phase (Lines 938-960)

```python
# Collect spectrograms from all rats
for rat_id, results in valid_results.items():
    for nm_size, window_data in results['averaged_windows'].items():
        spectrograms = aggregated_windows[nm_size]['spectrograms']
        spectrograms.append(window_data['avg_spectrogram'])
        
        aggregated_windows[nm_size]['total_events'].append(window_data['n_events'])
        aggregated_windows[nm_size]['n_sessions'].append(window_data['n_sessions'])
        aggregated_windows[nm_size]['rat_ids'].append(rat_id)
```

### Cross-Rats Averaging (Lines 964-987)

```python
for nm_size, data in aggregated_windows.items():
    spectrograms = np.array(data['spectrograms'])  # Shape: (n_rats, n_freqs, n_times)
    
    # Average across rats
    avg_spectrogram = np.mean(spectrograms, axis=0)
    
    # Apply mild smoothing to reduce horizontal line artifacts
    from scipy.ndimage import gaussian_filter1d
    avg_spectrogram = gaussian_filter1d(avg_spectrogram, sigma=0.8, axis=0)
```

### Artifact Reduction Strategy

The aggregation includes **mild Gaussian smoothing** along the frequency axis:
- **Purpose**: Reduces horizontal line artifacts from outlier frequencies
- **Parameters**: `sigma=0.8` provides gentle smoothing
- **Preservation**: Maintains overall spectral structure while reducing artifacts

### Comprehensive Metadata Tracking

```python
final_aggregated_windows[nm_size] = {
    'avg_spectrogram': avg_spectrogram,
    'window_times': first_result['averaged_windows'][nm_size]['window_times'],
    'total_events_per_rat': data['total_events'],
    'n_sessions_per_rat': data['n_sessions'],
    'rat_ids': data['rat_ids'],
    'n_rats': spectrograms.shape[0],
    'total_events_all_rats': sum(data['total_events']),
    'total_sessions_all_rats': sum(data['n_sessions'])
}
```

**Benefits**:
- **Statistical power**: Combines data from multiple rats for population-level analysis
- **Artifact reduction**: Smoothing reduces noise while preserving signal
- **Complete tracking**: Maintains full provenance and statistics

---

## 10. Advanced Visualization (Lines 1095-1323)

The `create_cross_rats_visualizations()` function implements professional-quality visualization:

### Color Mapping Strategy

```python
# Calculate color limits from all spectrograms
all_spectrograms = []
for nm_size in nm_sizes:
    window_data = results['averaged_windows'][nm_size]
    all_spectrograms.append(window_data['avg_spectrogram'])

vmin, vmax = calculate_color_limits(all_spectrograms)
```

The `calculate_color_limits()` function:
1. **Concatenates all data**: Creates unified data range
2. **Applies percentile clipping**: Removes extreme outliers
3. **Ensures symmetry**: Creates balanced color scale around zero
4. **Rounds values**: Provides clean color bar labels

### Professional Color Palette

The script uses the **Parula colormap** (Lines 246-336):
- **Origin**: MATLAB's default colormap, scientifically designed
- **Properties**: Perceptually uniform, colorblind-friendly
- **Implementation**: Custom ListedColormap with 256 color values

### Log-Frequency Axis Implementation

```python
# Use log-frequency spacing for y-axis
log_frequencies = np.log10(frequencies)

# Set y-axis ticks to show frequencies on log scale
ax.set_yticks(log_frequencies)

# Create intelligent frequency labeling
freq_labels = [''] * len(frequencies)
freq_labels[0] = f'{frequencies[0]:.1f}'  # Always show first
freq_labels[-1] = f'{frequencies[-1]:.1f}'  # Always show last

# Show 5 intermediate frequencies with minimum distance constraint
if len(frequencies) > 6:
    n_intermediate = 5
    indices = np.linspace(1, len(frequencies)-2, n_intermediate, dtype=int)
    min_distance = max(1, len(frequencies) // 5)
    
    for idx in indices:
        if idx >= min_distance and idx <= len(frequencies) - min_distance - 1:
            freq_labels[idx] = f'{frequencies[idx]:.1f}'
```

### Dimension Mismatch Handling

```python
if actual_samples != expected_samples:
    if actual_samples > expected_samples:
        # Trim excess samples symmetrically
        excess = actual_samples - expected_samples
        start_trim = excess // 2
        end_trim = excess - start_trim
        avg_spectrogram = avg_spectrogram[:, start_trim:actual_samples-end_trim]
    else:
        # Pad with edge values
        padding = expected_samples - actual_samples
        avg_spectrogram = np.pad(avg_spectrogram, ((0, 0), (0, padding)), mode='edge')
```

### Artifact Reduction in Visualization

```python
# Use improved shading to reduce artifacts from irregular data
im = ax.pcolormesh(window_times, log_frequencies, avg_spectrogram,
                  shading='gouraud', cmap=PARULA_COLORMAP, vmin=vmin, vmax=vmax)
```

**`shading='gouraud'` benefits**:
- **Smooth interpolation**: Reduces pixelation artifacts
- **Better visualization**: Creates more professional-looking plots
- **Handles irregular data**: Works well with uneven frequency spacing

---

## 11. Interactive Features (Lines 1536-1549)

Integration with the `interactive_spectrogram.py` module:

```python
# Create interactive spectrograms
try:
    interactive_figs = add_interactive_to_results(
        results=aggregated_results,
        save_path=save_path
    )
    if verbose:
        print("✓ Interactive spectrograms created! Open HTML files in browser to hover and see values.")
except Exception as e:
    if verbose:
        print(f"⚠️  Could not create interactive spectrograms: {e}")
```

**Interactive features include**:
- **Hover tooltips**: Show exact frequency, time, and z-score values
- **Zoom and pan**: Navigate large spectrograms easily
- **HTML export**: Creates standalone HTML files for sharing
- **Browser compatibility**: Works in any modern web browser

---

## 12. Comprehensive Error Handling & Resilience

The script implements **multiple layers of error handling**:

### Session-Level Resilience
- **Built-in retry logic**: Automatically retries failed sessions
- **Memory failure recovery**: Handles out-of-memory conditions
- **Graceful degradation**: Continues with available sessions

### Rat-Level Error Handling
```python
try:
    rat_id_str, results = process_single_rat_multi_session(...)
    rat_results[rat_id_str] = results
    
    if results is None:
        failed_rats.append(rat_id_str)
        error_details[rat_id_str] = "Processing failed - see logs above"
    else:
        successful_rats.append(rat_id_str)
        
except Exception as e:
    # Handle unexpected errors
    failed_rats.append(rat_id)
    error_details[rat_id] = str(e)
```

### Analysis-Level Recovery
- **Partial results handling**: Continues analysis with available rats
- **Detailed error reporting**: Provides specific failure information
- **Clean resource management**: Ensures proper cleanup even on failure

### Memory Management Strategy
```python
# Force garbage collection after each rat
gc.collect()

# Cleanup intermediate files
if cleanup_intermediate_files:
    cleanup_session_folders(rat_save_path, verbose=verbose)

# Clean up memory at dataset scanning
del all_data
gc.collect()
```

---

## 13. Configuration Integration (Lines 36, 1380-1402)

Seamless integration with the configuration system:

### Configuration Import
```python
from config import AnalysisConfig, DataConfig, PlottingConfig
```

### Default Parameter Application
```python
# Apply configuration defaults for None values
if pkl_path is None:
    pkl_path = DataConfig.get_data_file_path(DataConfig.MAIN_EEG_DATA_FILE)
if freq_min is None:
    freq_min = AnalysisConfig.THETA_MIN_FREQ
if freq_max is None:
    freq_max = AnalysisConfig.THETA_MAX_FREQ
# ... continue for all parameters
```

**Configuration benefits**:
- **Centralized settings**: All defaults in one location
- **Easy customization**: Override only needed parameters
- **Consistency**: Same defaults across all analysis scripts
- **Maintainability**: Single point of configuration updates

---

## 14. Main Execution Modes (Lines 1585-1710)

The script supports **dual execution modes**:

### Command-Line Mode
```python
def main():
    parser = argparse.ArgumentParser(
        description='Cross-Rats NM Theta Analysis - Pre-Event Normalization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze frontal ROI across all rats
  python nm_theta_cross_rats.py --roi frontal --freq_min 3 --freq_max 8
        """
    )
    # ... argument parsing and execution
```

### IDE/Interactive Mode
```python
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line usage
        main()
    else:
        # IDE usage with direct function calls
        results = run_cross_rats_analysis(
            roi="6",
            pkl_path=data_path,
            freq_min=3.0,
            freq_max=7.0,
            freq_file_path="data/config/frequencies.txt",
            window_duration=2.0,
            save_path="results/cross_rats",
            cleanup_intermediate_files=True,
            verbose=True
        )
```

**Current IDE Configuration**:
- **ROI**: Channel 6 (specific electrode analysis)
- **Frequency range**: 3-7 Hz (focused theta band)
- **Frequency source**: File-based for precise control
- **Window duration**: 2 seconds around events
- **Cleanup**: Enabled for disk space management
- **Verbosity**: Full detailed output

---

## 15. Key Strengths

The implementation demonstrates several key strengths:

### 1. Methodological Rigor
- **Established EEG practices**: Follows neuroscientific analysis standards
- **Proper z-score implementation**: Uses appropriate statistical normalization
- **Baseline period selection**: Uses validated pre-event normalization window

### 2. Robust Architecture
- **Error recovery**: Multiple layers of failure handling
- **Memory management**: Efficient resource usage and cleanup
- **Automatic validation**: Built-in compatibility checking

### 3. Professional Visualization
- **Scientific color mapping**: Perceptually uniform, publication-ready plots
- **Intelligent axis labeling**: Clear, uncluttered frequency displays
- **Artifact reduction**: Smooth visualization of complex data

### 4. Comprehensive Documentation
- **Detailed logging**: Complete progress and error reporting
- **Metadata tracking**: Full provenance of all processing steps
- **Configuration integration**: Centralized, maintainable settings

### 5. Flexibility and Extensibility
- **Multiple frequency sources**: File-based and generated options
- **ROI specification**: Supports various electrode selection methods
- **Dual execution modes**: Command-line and interactive usage

---

## 16. Pipeline Validation Perspective

**Why this pipeline should produce correct results**:

### Methodological Foundation
- **Pre-event baseline normalization**: Established method in EEG analysis literature
- **Cross-event statistics**: Proper z-score denominator calculation
- **Population aggregation**: Standard approach for group-level analysis

### Technical Implementation
- **Proper data handling**: Careful dimension management and error checking
- **Artifact mitigation**: Multiple smoothing and filtering steps
- **Statistical rigor**: Appropriate use of means, standard deviations, and z-scores

### Validation Features
- **Complete metadata**: Tracks all processing parameters and statistics
- **Error checking**: Validates data integrity at multiple levels
- **Transparent reporting**: Full documentation of inclusion/exclusion decisions

### Quality Assurance
- **Built-in resilience**: Handles various failure modes gracefully
- **Memory efficiency**: Manages resources effectively for large datasets
- **Professional visualization**: Enables visual validation of results

The pipeline implements established best practices for EEG spectral analysis and should produce valid, interpretable results for theta power differences between Near-Mistake types.

---

## Conclusion

The `nm_theta_cross_rats.py` script represents a sophisticated, production-quality implementation of cross-rats EEG theta analysis. Its combination of methodological rigor, robust error handling, professional visualization, and comprehensive documentation makes it suitable for high-quality neuroscientific research.

The pre-event baseline normalization approach, combined with proper cross-event statistical analysis and population-level aggregation, should provide reliable and interpretable results for understanding theta oscillation differences between different types of near-mistake events.