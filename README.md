# EEG Near-Mistake Analysis Pipeline

A comprehensive Python pipeline for analyzing theta oscillations during near-mistake events in EEG data. This repository contains the complete analysis framework used in the research investigating frontal theta power as a neural marker of cognitive control during error detection.

## Overview

This project analyzes electroencephalographic (EEG) data from rats performing a Go-NoGo task to investigate theta oscillations (3-7 Hz) during near-mistake (NM) events. Near-mistake events occur when subjects initiate an incorrect response but successfully correct it before completion, providing a unique window into real-time cognitive control mechanisms.

## Key Features

- **Multi-subject analysis** with cross-rat aggregation and reliability weighting
- **Advanced time-frequency analysis** using Morlet wavelets with adaptive cycles
- **Multiple normalization methods** (baseline, pre-event, global)
- **Robust error handling** with session-level failure recovery
- **Publication-ready visualizations** with interactive spectrograms
- **Flexible ROI analysis** supporting multiple electrode configurations
- **Memory-efficient processing** for large-scale EEG datasets

## Scientific Context

The analysis pipeline investigates the hypothesis that frontal theta oscillations serve as neural markers of cognitive control during near-mistake events. The research examines:

1. Whether theta power increases during near-mistake events compared to baseline
2. If theta increases scale with mistake magnitude (graded cognitive control)
3. Whether effects are strongest over frontal electrodes

**For detailed scientific methodology and results, see [`scientific_report.tex`](scientific_report.tex).**

## Repository Structure

```
eeg-near_mistakes/
├── src/
│   ├── config/              # Analysis configuration
│   ├── core/                # Core analysis modules
│   │   ├── implementations/ # Spectrogram computation methods
│   │   ├── normalization/   # Data normalization approaches
│   │   ├── utils/           # Utility functions
│   │   └── legacy/          # Legacy analysis methods
│   └── eeg_analysis_package/ # Core EEG processing tools
├── scripts/                 # Analysis and utility scripts
├── tests/                   # Test and validation code
├── docs/                    # Technical documentation
├── data/
│   ├── config/              # Configuration files
│   └── processed/           # Processed data files
└── results/                 # Analysis outputs and figures
```

## Installation

### Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- MNE-Python (for time-frequency analysis)
- Pandas (for data manipulation)
- Plotly (for interactive visualizations)

### Setup

```bash
git clone https://github.com/your-username/eeg-near_mistakes.git
cd eeg-near_mistakes
pip install -r requirements.txt
```

## Quick Start

### Basic Cross-Subject Analysis

```python
from src.core.nm_theta_cross_rats import run_cross_rats_analysis

# Run analysis for frontal ROI
results = run_cross_rats_analysis(
    roi="frontal",
    pkl_path="data/processed/all_eeg_data.pkl",
    freq_min=3.0,
    freq_max=7.0,
    n_freqs=12,
    window_duration=2.0,
    save_path="results/cross_rats"
)
```

### Single Subject Analysis

```python
from src.core.implementations.nm_theta_single_basic import analyze_session_nm_theta_roi

# Analyze single rat session
results = analyze_session_nm_theta_roi(
    session_data=session_data,
    roi_or_channels="frontal",
    freq_range=(3, 7),
    n_freqs=12
)
```

## Configuration

The analysis pipeline is highly configurable through `src/config/analysis_config.py`:

### Time-Frequency Parameters

```python
# Frequency analysis settings
THETA_MIN_FREQ = 3.0
THETA_MAX_FREQ = 7.0
N_FREQS_DEFAULT = 12

# Cycles configuration
CYCLES_METHOD = 'custom'  # 3 cycles at 3-5Hz, 4 cycles at 6-7Hz
```

### Analysis Windows

```python
WINDOW_DURATION_DEFAULT = 2.0  # seconds
BASELINE_TIME_WINDOW = (-1.0, -0.5)  # Pre-event baseline
```

## Analysis Methods

### 1. Time-Frequency Analysis

- **Morlet wavelets** with frequency-adaptive cycles for optimal time-frequency resolution
- **Logarithmic frequency spacing** for efficient spectral coverage
- **Pre-event baseline normalization** (-1.0 to -0.5 seconds)

### 2. Cross-Subject Aggregation

- **Reliability weighting** based on data quality metrics
- **Robust error handling** with graceful failure recovery
- **Memory-efficient processing** for large datasets

### 3. Statistical Analysis

- **Z-score normalization** using pre-event baseline statistics
- **Event-wise averaging** to remove temporal artifacts
- **Cross-frequency analysis** with consistent normalization

## Key Findings

The analysis reveals:

1. **Graded theta increases**: Frontal theta power systematically increases with near-mistake magnitude
2. **Spatial specificity**: Effects are strongest over frontal electrodes
3. **Temporal precision**: Theta increases occur during the critical error detection window

**Complete results and statistical analysis are detailed in [`scientific_report.tex`](scientific_report.tex).**

## Scripts and Utilities

- **`scripts/discover_rats.py`** - Automatically discover available subjects
- **`scripts/aggregate_results.py`** - Aggregate analysis results
- **`scripts/cleanup_results.py`** - Manage storage and cleanup
- **`scripts/create_eeg_data_file.py`** - Prepare data files

## Data Format

The pipeline expects EEG data in pickle format with the following structure:

```python
{
    'rat_id': {
        'session_index': {
            'eeg_data': ndarray,  # Shape: (channels, samples)
            'nm_events': {
                'nm_size': {
                    'event_samples': list,  # Event onset samples
                    'event_metadata': dict
                }
            }
        }
    }
}
```

## Visualization

The pipeline generates:

- **Static spectrograms** (PNG) for publication
- **Interactive plots** (HTML) for data exploration
- **Statistical summaries** with confidence intervals
- **ROI comparison plots** across electrode groups

## Testing and Validation

The repository includes comprehensive tests:

- **`tests/theta_zscore_diagnostics.py`** - Normalization validation
- **`tests/spectrogram_comparison.py`** - Method comparison
- **`tests/test_roi_analysis.py`** - ROI functionality testing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this pipeline in your research, please cite:

```
[Your publication citation will go here]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about the analysis pipeline or scientific methodology, please contact:

- **Code/Pipeline**: [Your GitHub profile]
- **Scientific Methods**: See corresponding author in [`scientific_report.tex`](scientific_report.tex)

---

**Note**: This repository contains the complete computational pipeline used for EEG near-mistake analysis. For detailed scientific methodology, statistical results, and interpretation, please refer to [`scientific_report.tex`](scientific_report.tex).