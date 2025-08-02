#!/usr/bin/env python3
"""
Debug Pipeline Validation

This script investigates why the synthetic validation has 31.6% error
by examining each step of the pipeline in detail.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from typing import Dict, Tuple

def debug_synthetic_data_generation():
    """Debug the synthetic data generation to verify ground truth."""
    print("🔍 DEBUGGING SYNTHETIC DATA GENERATION")
    print("=" * 60)
    
    # Load synthetic data
    synthetic_path = "data/synthetic/synthetic_eeg_validation.pkl"
    if not os.path.exists(synthetic_path):
        print(f"❌ Synthetic data not found: {synthetic_path}")
        return
    
    with open(synthetic_path, 'rb') as f:
        all_sessions = pickle.load(f)
    
    print(f"Loaded {len(all_sessions)} synthetic sessions")
    
    # Check ground truth ratios
    print("\n📊 GROUND TRUTH VERIFICATION:")
    for session_idx, session in enumerate(all_sessions[:3]):  # First 3 sessions
        print(f"\nSession {session_idx} (Rat {session['rat_id']}):")
        ground_truth = session['ground_truth_theta_ratios']
        print(f"  Ground truth ratios: {ground_truth}")
        
        # Count events per NM type
        nm_sizes = session['nm_sizes']
        for nm_type in [1.0, 2.0, 3.0]:
            count = np.sum(nm_sizes == nm_type)
            print(f"  NM Type {nm_type}: {count} events")
    
    # Analyze the raw EEG data around events
    print("\n🔬 RAW EEG ANALYSIS AROUND EVENTS:")
    session = all_sessions[0]  # First session
    eeg_data = session['eeg']
    times = session['eeg_time']
    nm_peak_times = session['nm_peak_times']
    nm_sizes = session['nm_sizes']
    
    # Focus on ROI channels (7, 15, 23 in 0-based indexing)
    roi_channels = [7, 15, 23]
    sfreq = 200.0
    
    # Analyze power around events for each NM type
    nm_types = [1.0, 2.0, 3.0]
    measured_powers = {}
    
    for nm_type in nm_types:
        event_indices = np.where(nm_sizes == nm_type)[0]
        if len(event_indices) == 0:
            continue
            
        print(f"\nNM Type {nm_type}: {len(event_indices)} events")
        
        # Extract windows around events
        window_duration = 2.0  # ±1 second
        window_samples = int(window_duration * sfreq)
        
        event_powers = []
        
        for event_idx in event_indices[:5]:  # First 5 events
            event_time = nm_peak_times[event_idx]
            
            # Find time indices
            center_idx = np.argmin(np.abs(times - event_time))
            start_idx = center_idx - window_samples // 2
            end_idx = start_idx + window_samples
            
            if start_idx >= 0 and end_idx < len(times):
                # Extract window for ROI channels
                window_data = eeg_data[roi_channels, start_idx:end_idx]
                
                # Simple power estimation: RMS in theta band (rough approximation)
                # Focus on 5Hz by looking at amplitude variations
                mean_power = np.mean(np.var(window_data, axis=1))  # Variance across time per channel, then mean
                event_powers.append(mean_power)
                
                if event_idx == event_indices[0]:  # Detailed analysis for first event
                    print(f"    Event at {event_time:.2f}s:")
                    print(f"      Time window: {times[start_idx]:.2f} to {times[end_idx-1]:.2f}s")
                    print(f"      Data shape: {window_data.shape}")
                    print(f"      Power estimate: {mean_power:.6f}")
        
        if event_powers:
            measured_powers[nm_type] = np.mean(event_powers)
            print(f"    Average raw power: {measured_powers[nm_type]:.6f}")
    
    # Check if raw power ratios match expectations
    if 1.0 in measured_powers:
        print(f"\n🎯 RAW POWER RATIOS (vs NM Type 1.0):")
        baseline = measured_powers[1.0]
        for nm_type in [1.0, 2.0, 3.0]:
            if nm_type in measured_powers:
                ratio = measured_powers[nm_type] / baseline
                expected = session['ground_truth_theta_ratios'][nm_type]
                print(f"  NM {nm_type}: Measured ratio = {ratio:.3f}, Expected = {expected:.3f}")


def debug_pipeline_processing():
    """Debug the pipeline processing steps."""
    print("\n\n🔍 DEBUGGING PIPELINE PROCESSING")
    print("=" * 60)
    
    # Load pipeline results
    results_path = "results/synthetic_validation/cross_rats_aggregated_results.pkl"
    if not os.path.exists(results_path):
        print(f"❌ Pipeline results not found: {results_path}")
        return
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    print("✅ Loaded pipeline results")
    print(f"Frequencies: {len(results['frequencies'])} from {results['frequencies'][0]:.2f} to {results['frequencies'][-1]:.2f} Hz")
    print(f"NM sizes: {list(results['averaged_windows'].keys())}")
    
    # Analyze each NM size processing
    print(f"\n📊 PIPELINE SPECTROGRAMS ANALYSIS:")
    
    for nm_size, window_data in results['averaged_windows'].items():
        print(f"\nNM Size {nm_size}:")
        avg_spectrogram = window_data['avg_spectrogram']
        window_times = window_data['window_times']
        n_rats = window_data['n_rats']
        total_events = window_data['total_events_all_rats']
        
        print(f"  Spectrogram shape: {avg_spectrogram.shape}")
        print(f"  Time window: {window_times[0]:.3f} to {window_times[-1]:.3f}s")
        print(f"  Total events: {total_events} from {n_rats} rats")
        print(f"  Z-score range: {avg_spectrogram.min():.3f} to {avg_spectrogram.max():.3f}")
        
        # Focus on 5Hz region where we added synthetic theta
        frequencies = results['frequencies']
        freq_5hz_idx = np.argmin(np.abs(frequencies - 5.0))
        freq_around_5hz = slice(max(0, freq_5hz_idx-2), min(len(frequencies), freq_5hz_idx+3))
        
        print(f"  5Hz region ({frequencies[freq_around_5hz][0]:.2f}-{frequencies[freq_around_5hz][-1]:.2f} Hz):")
        
        # Analyze power at event time (t=0)
        event_time_idx = np.argmin(np.abs(window_times))
        power_at_event = avg_spectrogram[freq_around_5hz, event_time_idx]
        mean_power_at_event = np.mean(power_at_event)
        
        print(f"    Power at t=0: {mean_power_at_event:.4f}")
        
        # Analyze power in pre-event baseline (-1.0 to -0.5s)
        baseline_mask = (window_times >= -1.0) & (window_times <= -0.5)
        if np.any(baseline_mask):
            baseline_power = np.mean(avg_spectrogram[freq_around_5hz, :][:, baseline_mask])
            print(f"    Baseline power (-1.0 to -0.5s): {baseline_power:.4f}")
            print(f"    Event vs baseline ratio: {mean_power_at_event/baseline_power:.3f}")
        
        # Analyze overall window power (what our validation uses)
        overall_power = np.mean(avg_spectrogram[freq_around_5hz, :])
        print(f"    Overall window power: {overall_power:.4f}")


def debug_normalization_effects():
    """Debug how pre-event baseline normalization affects synthetic data."""
    print("\n\n🔍 DEBUGGING NORMALIZATION EFFECTS")  
    print("=" * 60)
    
    # This would require looking at the normalization step in detail
    # Let's check if the pre-event baseline normalization is interfering with our synthetic bursts
    
    print("Pre-event baseline normalization uses -1.0 to -0.5s as baseline")
    print("Our synthetic theta bursts are centered at t=0 with ±1s duration")
    print("So the normalization baseline period OVERLAPS with our synthetic theta bursts!")
    print("")
    print("This could explain the errors:")
    print("1. Baseline period (-1.0 to -0.5s) includes synthetic theta → biased baseline")  
    print("2. Different NM types have different theta strengths in baseline → different normalization")
    print("3. This creates non-linear effects in the final z-scores")
    
    return True


def create_diagnostic_plots():
    """Create diagnostic plots to visualize the issues."""
    print("\n\n📊 CREATING DIAGNOSTIC PLOTS")
    print("=" * 40)
    
    # Load results for plotting
    results_path = "results/synthetic_validation/cross_rats_aggregated_results.pkl"
    if not os.path.exists(results_path):
        print("❌ No results to plot")
        return
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    frequencies = results['frequencies']
    averaged_windows = results['averaged_windows']
    
    # Create figure comparing spectrograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (nm_size, window_data) in enumerate(averaged_windows.items()):
        ax = axes[i]
        avg_spectrogram = window_data['avg_spectrogram']
        window_times = window_data['window_times']
        
        # Plot spectrogram
        im = ax.pcolormesh(window_times, frequencies, avg_spectrogram, 
                          shading='auto', cmap='RdBu_r')
        ax.axvline(0, color='black', linestyle='--', alpha=0.7)
        ax.axvline(-1.0, color='red', linestyle=':', alpha=0.7, label='Baseline start')
        ax.axvline(-0.5, color='red', linestyle=':', alpha=0.7, label='Baseline end')
        
        ax.set_title(f'NM Size {nm_size}\nMax z-score: {avg_spectrogram.max():.3f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(3, 7)  # Focus on theta range
        
        plt.colorbar(im, ax=ax, label='Z-score')
        
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/synthetic_validation/debug_spectrograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Diagnostic plots saved to results/synthetic_validation/debug_spectrograms.png")


def main():
    """Run complete debugging analysis."""
    print("🔬 SYNTHETIC VALIDATION DEBUGGING")
    print("=" * 80)
    print("Investigating why validation has 31.6% error instead of ~0%")
    print("=" * 80)
    
    # Step 1: Check synthetic data generation
    debug_synthetic_data_generation()
    
    # Step 2: Check pipeline processing  
    debug_pipeline_processing()
    
    # Step 3: Check normalization effects
    debug_normalization_effects()
    
    # Step 4: Create diagnostic plots
    create_diagnostic_plots()
    
    print("\n" + "=" * 80)
    print("🔬 DEBUGGING COMPLETE")
    print("=" * 80)
    print("\nKey insights:")
    print("1. Check if synthetic theta bursts are generated correctly")
    print("2. Verify pre-event baseline normalization doesn't interfere")
    print("3. Ensure frequency analysis targets the right spectral content")
    print("4. Look for non-linear interactions in the processing pipeline")


if __name__ == "__main__":
    main()