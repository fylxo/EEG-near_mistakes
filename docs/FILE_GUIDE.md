# EEG Near-Mistake Analysis - File Guide

This guide explains the purpose and usage of different files in the codebase for analyzing theta oscillations around Near Mistake (NM) events.

## 📁 Directory Structure

```
eeg-near_mistakes/
├── data/                           # Data files and configurations
├── src/                           # Source code
│   ├── core/                     # Core analysis functions
│   └── scripts/                  # Standalone analysis scripts
├── results/                      # Organized analysis results
├── docs/                         # Documentation
├── tests/                        # Test files
├── scripts/                      # Utility scripts
└── interactive_theta_analysis.ipynb  # Interactive notebook
```

## 🧠 Core Analysis Files

### `src/core/nm_theta_analysis.py`
**Purpose:** Single-session theta oscillation analysis around NM events
**Use when:** Analyzing one recording session at a time
**Key function:** `analyze_session_nm_theta_roi()`
- Extracts theta power (3-8 Hz) around NM events
- Performs z-score normalization
- Creates spectrograms showing frequency vs time
- Saves results as .pkl files and plots

### `src/core/nm_theta_multi_session.py`
**Purpose:** Multi-session analysis for combining results across sessions
**Use when:** Comparing theta patterns across multiple sessions from the same rat
**Key function:** `analyze_rat_multi_session()`
- Loads and processes multiple sessions
- Averages spectrograms across sessions 
- Creates statistical comparisons
- Memory-intensive approach

### `src/scripts/nm_theta_multi_session_memory_efficient.py`
**Purpose:** Memory-efficient multi-session analysis
**Use when:** Analyzing many sessions without running out of memory
**Recommended for:** Large datasets with many sessions per rat
**Key features:**
- Processes sessions one by one
- Saves intermediate results immediately
- Two-level averaging (session → multi-session)
- Explicit memory cleanup between sessions

### `src/core/electrode_utils.py`
**Purpose:** Electrode mapping and ROI (Region of Interest) utilities
**Use when:** Converting between channel numbers and brain regions
**Key functions:**
- `get_channels()`: Get channel numbers for specific ROIs
- `load_electrode_mappings()`: Load rat-specific electrode configurations
- ROI definitions: frontal, motor, hippocampus, etc.

### `src/core/spectral_analysis_pipeline.py`
**Purpose:** General spectral analysis functions
**Use when:** Need broader frequency analysis beyond theta
**Key features:**
- Power spectral density (PSD) analysis
- Multiple frequency bands
- Batch processing capabilities

## 📊 Interactive Analysis

### `interactive_theta_analysis.ipynb`
**Purpose:** Interactive notebook for rapid analysis and parameter testing
**Use when:** 
- Exploring data interactively
- Testing different parameters
- Quick single-session analysis
- Teaching/demonstration

**Key sections:**
1. **Setup:** Load all data into memory once
2. **Single Session:** Analyze individual sessions
3. **Multi-Session:** Process multiple sessions for one rat
4. **Parameter Testing:** Try different frequencies/ROIs

## 🔧 Utility Scripts

### `scripts/cleanup_results.py`
**Purpose:** Manage disk space by cleaning up old result folders
**Usage:**
```bash
python scripts/cleanup_results.py --list                    # List all result folders
python scripts/cleanup_results.py --clean-legacy           # Clean old folders
python scripts/cleanup_results.py --move-legacy            # Move to organized structure
```

### `src/scripts/batch_spectral_analysis.py`
**Purpose:** Run spectral analysis on multiple files in batch
**Use when:** Processing many sessions with the same parameters

## 📋 Test Files

### `tests/test_spectral_pipeline.py`
**Purpose:** Test spectral analysis functions
**Use when:** Verifying analysis pipeline works correctly

### `tests/test_roi_analysis.py`
**Purpose:** Test ROI-specific analysis functions
**Use when:** Verifying electrode mapping and ROI functions

## 🔬 Development and Research Notebooks

### `src/core/data_loading_and_analysis_functions.ipynb`
**Purpose:** Development notebook for data loading and basic EEG analysis functions
**Contains:** 
- Functions for reading .mat files and converting to Python format
- Basic time-domain and frequency-domain analysis functions
- Morlet spectrogram computation
- Multi-channel analysis utilities
**Use when:** Developing new analysis functions or understanding data structure

### `src/core/electrode_mapping_analysis.ipynb`
**Purpose:** Analysis of electrode placement and anatomical mapping
**Contains:**
- Functions for understanding electrode placement files
- Extraction of anatomical labels and mapping vectors
- Electrode placement verification
**Use when:** Working with electrode configurations or anatomical mappings

## 📈 Which File Should I Use?

### For Single Session Analysis:
- **Quick exploration:** `interactive_theta_analysis.ipynb`
- **Automated processing:** `src/core/nm_theta_analysis.py`
- **Custom scripts:** Import functions from `src/core/nm_theta_analysis.py`

### For Multi-Session Analysis:
- **Small datasets (< 10 sessions):** `src/core/nm_theta_multi_session.py`
- **Large datasets (> 10 sessions):** `src/scripts/nm_theta_multi_session_memory_efficient.py`
- **Interactive exploration:** `interactive_theta_analysis.ipynb`

### For General Spectral Analysis:
- **Beyond theta bands:** `src/core/spectral_analysis_pipeline.py`
- **Batch processing:** `src/scripts/batch_spectral_analysis.py`

## 🏗️ Typical Workflows

### Workflow 1: Analyze Single Rat, Multiple Sessions
```python
# Option A: Memory-efficient (recommended for > 10 sessions)
python src/scripts/nm_theta_multi_session_memory_efficient.py --rat_id 531 --roi frontal

# Option B: Interactive notebook
# Use interactive_theta_analysis.ipynb, multi-session section
```

### Workflow 2: Quick Parameter Testing
```python
# Use interactive_theta_analysis.ipynb
# Modify parameters in notebook cells
# Run single session analysis with different ROIs/frequencies
```

### Workflow 3: Batch Processing Multiple Rats
```python
# Create custom script importing from:
from src.core.nm_theta_multi_session import analyze_rat_multi_session
# or
from nm_theta_multi_session_memory_efficient import analyze_rat_multi_session_memory_efficient
```

## 📊 Output Files

### Analysis Results:
- **`.pkl` files:** Python pickle files with full analysis results
- **`.png` files:** Spectrogram plots and frequency profiles  
- **`.txt` files:** Human-readable summaries

### Result Organization:
- **`results/single_sessions/`:** Individual session results
- **`results/multi_session/`:** Multi-session aggregated results
- **`results/roi_analysis/`:** ROI-specific analysis results
- **`results/spectral_analysis/`:** General spectral analysis results

## 🚀 Getting Started

1. **First time users:** Start with `interactive_theta_analysis.ipynb`
2. **Single session analysis:** Use notebook or `nm_theta_analysis.py`
3. **Multi-session analysis:** Use `nm_theta_multi_session_memory_efficient.py`
4. **Clean up results:** Use `scripts/cleanup_results.py` to manage disk space

## 📝 File Naming Conventions

- **Scripts:** `nm_*` prefix for Near Mistake analysis
- **Results:** `rat_{id}_{analysis_type}_{date}` format
- **Test files:** `test_*` prefix
- **Utilities:** Descriptive names in `scripts/` folder

## 🔍 Dependencies

Key Python packages used:
- `numpy`: Numerical computations
- `matplotlib`: Plotting
- `scipy`: Signal processing 
- `pandas`: Data manipulation
- `pickle`: Data serialization