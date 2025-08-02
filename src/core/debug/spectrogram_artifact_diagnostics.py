#!/usr/bin/env python3
"""
Spectrogram Artifact Diagnostics

This script investigates horizontal line artifacts in spectrograms by checking:
1. Time-frequency data resolution and binning
2. Plotting interpolation settings  
3. Z-score normalization discontinuities
4. Session averaging and variance consistency
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from typing import Dict, List, Tuple

def load_cross_rats_results(results_path: str) -> Dict:
    """Load cross-rats aggregated results."""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results

def check_data_resolution(results: Dict):
    """Check 1: Time-frequency data resolution and binning."""
    print("🔍 DIAGNOSTIC 1: Time-Frequency Data Resolution")
    print("=" * 60)
    
    frequencies = results['frequencies']
    
    for nm_size, window_data in results['averaged_windows'].items():
        avg_spectrogram = window_data['avg_spectrogram']
        window_times = window_data['window_times']
        
        print(f"\nNM Size {nm_size}:")
        print(f"  Spectrogram shape: {avg_spectrogram.shape}")
        print(f"  Time points: {len(window_times)}")
        print(f"  Frequency points: {len(frequencies)}")
        print(f"  Time resolution: {np.diff(window_times).mean():.4f} s")
        print(f"  Frequency spacing (linear): {np.diff(frequencies).mean():.4f} Hz")
        print(f"  Frequency spacing (log): {np.diff(np.log10(frequencies)).mean():.4f} log10(Hz)")
        
        # Check for irregular frequency spacing
        log_freq_diffs = np.diff(np.log10(frequencies))
        log_spacing_std = np.std(log_freq_diffs)
        print(f"  Log frequency spacing std: {log_spacing_std:.6f} (should be ~0 for regular log spacing)")
        
        if log_spacing_std > 0.01:
            print(f"  ⚠️  WARNING: Irregular frequency spacing detected!")
            irregular_indices = np.where(np.abs(log_freq_diffs - np.mean(log_freq_diffs)) > 2*log_spacing_std)[0]
            print(f"     Irregular gaps at frequency indices: {irregular_indices}")
            for idx in irregular_indices[:5]:  # Show first 5
                print(f"     Gap between {frequencies[idx]:.3f} and {frequencies[idx+1]:.3f} Hz")

def check_normalization_artifacts(results: Dict):
    """Check 3: Z-score normalization discontinuities."""
    print("\n\n🔍 DIAGNOSTIC 3: Z-Score Normalization Issues")
    print("=" * 60)
    
    for nm_size, window_data in results['averaged_windows'].items():
        avg_spectrogram = window_data['avg_spectrogram']
        window_times = window_data['window_times']
        
        print(f"\nNM Size {nm_size}:")
        print(f"  Z-score range: {avg_spectrogram.min():.3f} to {avg_spectrogram.max():.3f}")
        print(f"  Z-score mean: {avg_spectrogram.mean():.3f}")
        print(f"  Z-score std: {avg_spectrogram.std():.3f}")
        
        # Check for zero-variance frequencies
        freq_variances = np.var(avg_spectrogram, axis=1)  # Variance across time for each frequency
        zero_var_freqs = np.where(freq_variances < 1e-10)[0]
        
        if len(zero_var_freqs) > 0:
            print(f"  ⚠️  WARNING: {len(zero_var_freqs)} frequencies with near-zero variance detected!")
            print(f"     Zero-variance frequency indices: {zero_var_freqs[:10]}...")  # Show first 10
        
        # Check for extreme outliers that could cause artifacts
        freq_means = np.mean(avg_spectrogram, axis=1)  # Mean across time for each frequency
        outlier_threshold = 3 * np.std(freq_means)
        outlier_freqs = np.where(np.abs(freq_means - np.mean(freq_means)) > outlier_threshold)[0]
        
        if len(outlier_freqs) > 0:
            print(f"  ⚠️  WARNING: {len(outlier_freqs)} frequency outliers detected!")
            for idx in outlier_freqs[:5]:  # Show first 5
                freq_val = results['frequencies'][idx] if idx < len(results['frequencies']) else 'N/A'
                print(f"     Frequency {idx} ({freq_val:.3f} Hz): mean z-score = {freq_means[idx]:.3f}")
        
        # Check for sudden jumps between adjacent frequencies
        freq_mean_diffs = np.abs(np.diff(freq_means))
        jump_threshold = np.percentile(freq_mean_diffs, 95)  # Top 5% of differences
        large_jumps = np.where(freq_mean_diffs > jump_threshold)[0]
        
        if len(large_jumps) > 0:
            print(f"  ⚠️  INFO: {len(large_jumps)} large jumps between adjacent frequencies")
            print(f"     Jump threshold (95th percentile): {jump_threshold:.3f}")
            for idx in large_jumps[:5]:  # Show first 5
                freq1 = results['frequencies'][idx] if idx < len(results['frequencies']) else 'N/A'
                freq2 = results['frequencies'][idx+1] if idx+1 < len(results['frequencies']) else 'N/A'
                print(f"     Jump between freq {idx} ({freq1:.3f} Hz) and {idx+1} ({freq2:.3f} Hz): {freq_mean_diffs[idx]:.3f}")

def check_session_averaging_artifacts(results: Dict):
    """Check 4: Session averaging and variance consistency."""
    print("\n\n🔍 DIAGNOSTIC 4: Session Averaging Artifacts")
    print("=" * 60)
    
    for nm_size, window_data in results['averaged_windows'].items():
        print(f"\nNM Size {nm_size}:")
        print(f"  Number of rats: {window_data['n_rats']}")
        print(f"  Total events across all rats: {window_data['total_events_all_rats']}")
        print(f"  Total sessions across all rats: {window_data['total_sessions_all_rats']}")
        
        # Check for uneven contributions across rats
        events_per_rat = window_data['total_events_per_rat']
        sessions_per_rat = window_data['n_sessions_per_rat']
        
        print(f"  Events per rat - mean: {np.mean(events_per_rat):.1f}, std: {np.std(events_per_rat):.1f}")
        print(f"  Sessions per rat - mean: {np.mean(sessions_per_rat):.1f}, std: {np.std(sessions_per_rat):.1f}")
        
        # Check for highly uneven contributions
        events_cv = np.std(events_per_rat) / np.mean(events_per_rat)  # Coefficient of variation
        if events_cv > 0.5:
            print(f"  ⚠️  WARNING: High variability in events per rat (CV = {events_cv:.3f})")
            print(f"     Range: {min(events_per_rat)} - {max(events_per_rat)} events per rat")

def create_interpolation_comparison(results: Dict, save_path: str = "results/cross_rats"):
    """Check 2: Test different plotting interpolation settings."""
    print("\n\n🔍 DIAGNOSTIC 2: Plotting Interpolation Comparison")
    print("=" * 60)
    
    # Get data for first NM size
    nm_size = list(results['averaged_windows'].keys())[0]
    window_data = results['averaged_windows'][nm_size]
    avg_spectrogram = window_data['avg_spectrogram']
    window_times = window_data['window_times']
    frequencies = results['frequencies']
    
    # Create comparison plot with different interpolation methods
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Different plotting approaches to test
    plot_configs = [
        {'title': 'pcolormesh + shading=auto (current)', 'method': 'pcolormesh', 'shading': 'auto'},
        {'title': 'pcolormesh + shading=nearest', 'method': 'pcolormesh', 'shading': 'nearest'},
        {'title': 'imshow + interpolation=bilinear', 'method': 'imshow', 'interpolation': 'bilinear'},
        {'title': 'imshow + interpolation=none', 'method': 'imshow', 'interpolation': 'none'}
    ]
    
    for i, config in enumerate(plot_configs):
        ax = axes[i]
        
        if config['method'] == 'pcolormesh':
            log_frequencies = np.log10(frequencies)
            im = ax.pcolormesh(window_times, log_frequencies, avg_spectrogram,
                              shading=config['shading'], cmap='RdBu_r')
            ax.set_yticks(log_frequencies[::len(log_frequencies)//5])
            ax.set_yticklabels([f'{f:.1f}' for f in frequencies[::len(frequencies)//5]])
        else:  # imshow
            im = ax.imshow(avg_spectrogram, aspect='auto', origin='lower',
                          extent=[window_times[0], window_times[-1], 0, len(frequencies)-1],
                          cmap='RdBu_r', interpolation=config['interpolation'])
            # Set y-ticks for imshow
            freq_ticks = np.linspace(0, len(frequencies)-1, 6)
            ax.set_yticks(freq_ticks)
            freq_labels = [f'{frequencies[int(idx)]:.1f}' for idx in freq_ticks]
            ax.set_yticklabels(freq_labels)
        
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax.set_title(config['title'])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        plt.colorbar(im, ax=ax, label='Z-score')
    
    plt.suptitle(f'Interpolation Method Comparison - NM Size {nm_size}', fontsize=14)
    plt.tight_layout()
    
    plot_file = os.path.join(save_path, 'interpolation_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Interpolation comparison plot saved: {plot_file}")
    plt.show()

def main():
    """Run all diagnostics."""
    results_path = "results/cross_rats/cross_rats_aggregated_results.pkl"
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return
    
    print("🔬 SPECTROGRAM ARTIFACT DIAGNOSTICS")
    print("=" * 80)
    print(f"Analyzing: {results_path}")
    
    # Load results
    results = load_cross_rats_results(results_path)
    
    # Run diagnostics
    check_data_resolution(results)
    check_normalization_artifacts(results) 
    check_session_averaging_artifacts(results)
    create_interpolation_comparison(results)
    
    print("\n" + "=" * 80)
    print("🔬 DIAGNOSTICS COMPLETE")
    print("=" * 80)
    print("\nLook for:")
    print("• Irregular frequency spacing (can cause horizontal lines)")
    print("• Zero-variance frequencies (appear as flat lines)")
    print("• Outlier frequencies (appear as horizontal streaks)")
    print("• Large jumps between adjacent frequencies (visible as bands)")
    print("• Uneven rat contributions (can cause averaging artifacts)")
    print("• Compare interpolation methods to identify best rendering")

if __name__ == "__main__":
    main()