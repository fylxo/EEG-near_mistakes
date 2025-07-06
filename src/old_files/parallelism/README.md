# Parallelism Testing Framework

This directory contains parallel implementations and benchmarking tools for EEG analysis functions.

## 🎯 Purpose

Test and validate parallel versions of computationally intensive EEG analysis functions to improve performance while ensuring identical results to sequential implementations.

## 📁 Files

### `parallel_roi_analysis.py`
**Parallel implementations of core functions:**
- `compute_roi_theta_spectrogram_parallel_channels()` - Parallelize across channels
- `compute_roi_theta_spectrogram_parallel_frequencies()` - Parallelize across frequencies  
- `analyze_rat_multi_session_parallel()` - Parallelize across sessions

**Parallelization strategies:**
- **Channel-level**: Process multiple channels simultaneously
- **Frequency-level**: Process frequency batches in parallel within each channel
- **Session-level**: Process multiple sessions simultaneously

### `performance_benchmark.py`
**Comprehensive benchmarking framework:**
- Performance timing comparisons
- Result validation (ensures parallel = sequential)
- Memory usage monitoring
- Scalability analysis with different problem sizes
- Detailed reporting and recommendations

### `test_parallelism.py`
**Quick testing script:**
- Validation tests with synthetic data
- Real data testing
- Scalability analysis
- Integration with benchmark framework

## 🚀 Quick Start

### 1. Basic Validation Test
```bash
cd /path/to/eeg-near_mistakes
python src/parallelism/test_parallelism.py --quick
```

### 2. Test with Real Data
```bash
python src/parallelism/test_parallelism.py --real data/processed/all_eeg_data.pkl
```

### 3. Comprehensive Benchmark
```bash
python src/parallelism/test_parallelism.py --full --output_dir results/parallelism
```

### 4. Scalability Analysis
```bash
python src/parallelism/test_parallelism.py --scalability
```

## 🔬 Parallelization Targets

### High-Impact Opportunities

**1. Channel Processing in ROI Analysis**
- **Location**: `compute_roi_theta_spectrogram()` in `nm_theta_analysis.py`
- **Current**: `for i, ch_idx in enumerate(roi_channels)` (sequential)
- **Parallel**: Process multiple channels with `multiprocessing.Pool`
- **Expected speedup**: Linear with number of channels (up to CPU cores)

**2. Multi-Session Processing**  
- **Location**: `analyze_rat_multi_session()` in `nm_theta_multi_session.py`
- **Current**: `for session_idx, (orig_session_idx, session_data) in enumerate(rat_sessions)`
- **Parallel**: Process N sessions simultaneously
- **Expected speedup**: Significant for rats with many sessions

**3. Frequency Processing**
- **Location**: Inner frequency loop in `compute_roi_theta_spectrogram()`
- **Current**: `for j, (freq, cycles) in enumerate(zip(freqs, n_cycles))`
- **Parallel**: Process frequency batches in parallel
- **Expected speedup**: Moderate, limited by memory bandwidth

## 📊 Benchmark Results

Results are saved to `results/parallelism/` and include:

- **Performance reports**: Execution times, speedups, memory usage
- **Validation results**: Ensuring parallel results match sequential
- **Recommendations**: Which parallelization strategies work best
- **Scalability analysis**: Performance vs problem size

## ⚙️ Parallelization Methods

### 1. Multiprocessing (`multiprocessing.Pool`)
- **Best for**: CPU-intensive tasks
- **Pros**: True parallelism, good for compute-bound operations
- **Cons**: Process overhead, memory duplication

### 2. Threading (`threading.Thread`)
- **Best for**: I/O-bound tasks or when memory sharing is important
- **Pros**: Lower overhead, shared memory
- **Cons**: Limited by Python GIL for CPU-bound tasks

### 3. Joblib (`joblib.Parallel`)
- **Alternative**: Could be added for scikit-learn integration
- **Pros**: Efficient shared memory, good for NumPy arrays

## 🔧 Memory Management

**Batch Processing:**
- Process data in smaller batches to control memory usage
- Configurable batch sizes for different problem scales
- Memory monitoring during execution

**Strategies:**
- Process N channels at a time rather than all channels
- Process sessions in groups rather than all at once
- Use shared memory where possible

## 📈 Expected Performance Gains

**Channel Parallelization:**
- 4 channels: ~2-3x speedup
- 8 channels: ~3-4x speedup  
- 16+ channels: ~4-6x speedup (limited by CPU cores)

**Session Parallelization:**
- Highly dependent on I/O and memory
- Best gains with many sessions (10+)
- May be limited by memory bandwidth

**Frequency Parallelization:**
- Moderate gains (1.5-2x)
- Limited by memory access patterns
- Most effective for high frequency resolution

## 🧪 Validation Strategy

**1. Numerical Validation:**
- Compare results with tolerance `1e-10`
- Verify shapes and statistical properties
- Check edge cases and boundary conditions

**2. Performance Validation:**
- Measure actual speedups
- Monitor memory usage
- Check CPU utilization

**3. Stress Testing:**
- Different problem sizes
- Various parameter combinations
- Error handling and edge cases

## 🚨 Known Limitations

**1. GIL Impact:**
- Python's GIL limits threading for CPU-bound tasks
- Multiprocessing preferred for computation

**2. Memory Overhead:**
- Parallel processes duplicate memory
- Need to balance parallelism vs memory usage

**3. I/O Bottlenecks:**
- Disk I/O can become bottleneck
- Data loading may limit gains

**4. Setup Overhead:**
- Process creation has overhead
- May not benefit small problems

## 💡 Usage Recommendations

**When to Use Parallel Versions:**

✅ **Use when:**
- Processing multiple channels (≥4)
- Multiple sessions (≥6) 
- Large datasets (>50k samples)
- High frequency resolution (≥20 frequencies)

❌ **Don't use when:**
- Single channel analysis
- Very small datasets (<10k samples)
- Quick prototyping/testing
- Memory-constrained environments

**Best Practices:**
1. Start with channel parallelization
2. Test with your actual data sizes
3. Monitor memory usage
4. Use the benchmark framework to validate
5. Consider batch processing for very large datasets

## 🔍 Troubleshooting

**Import Errors:**
- Run from project root directory
- Check that core modules are importable

**Performance Issues:**
- Monitor CPU and memory usage
- Try different `n_jobs` values
- Check for I/O bottlenecks

**Memory Errors:**
- Reduce batch sizes
- Use fewer parallel jobs
- Consider frequency-level parallelization instead

**Validation Failures:**
- Check numerical tolerance
- Verify input data consistency
- Check for race conditions