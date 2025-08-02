#!/usr/bin/env python3
"""
Test Synthetic Theta Bursts

This script tests ONLY the synthetic theta burst generation to verify
it's creating the correct power ratios before running the full pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.validate_pipeline_synthetic import generate_synthetic_eeg_session

def test_synthetic_bursts():
    """Test if synthetic theta bursts are generated correctly."""
    print("🧪 TESTING SYNTHETIC THETA BURST GENERATION")
    print("=" * 60)
    
    # Generate one test session
    session_data = generate_synthetic_eeg_session(
        rat_id="9151",
        session_idx=0,
        session_duration=60.0,  # Short session for testing
        nm_events_per_type=5    # Few events for focused testing
    )
    
    print("\n🔬 ANALYZING GENERATED BURSTS:")
    
    eeg_data = session_data['eeg']
    times = session_data['eeg_time']
    nm_peak_times = session_data['nm_peak_times']
    nm_sizes = session_data['nm_sizes']
    ground_truth = session_data['ground_truth_theta_ratios']
    
    print(f"Ground truth ratios: {ground_truth}")
    
    # Focus on ROI channels where we added theta bursts
    roi_channels = [7, 15, 23]  # 0-based indexing
    sfreq = 200.0
    
    # Analyze power around each event
    nm_types = [1.0, 2.0, 3.0]
    event_powers = {nm_type: [] for nm_type in nm_types}
    
    for i, (event_time, nm_size) in enumerate(zip(nm_peak_times, nm_sizes)):
        print(f"\nEvent {i+1}: NM size {nm_size} at {event_time:.2f}s")
        
        # Extract tight window around event (±0.5s to capture our ±0.4s burst)
        window_duration = 1.0  
        center_idx = np.argmin(np.abs(times - event_time))
        half_window = int(window_duration * sfreq / 2)
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window
        
        if start_idx >= 0 and end_idx < len(times):
            # Extract window for ROI channels
            window_data = eeg_data[roi_channels, start_idx:end_idx]
            window_times = times[start_idx:end_idx]
            
            # Calculate RMS power in the window (proxy for theta power)
            rms_power = np.sqrt(np.mean(window_data ** 2, axis=1))  # Per channel
            mean_rms = np.mean(rms_power)  # Across ROI channels
            
            event_powers[nm_size].append(mean_rms)
            
            print(f"  Window: {window_times[0]:.2f} to {window_times[-1]:.2f}s")
            print(f"  ROI RMS power: {mean_rms:.6f}")
            
            # Detailed analysis of the burst region (±0.4s around event)
            burst_mask = np.abs(window_times - event_time) <= 0.4
            if np.any(burst_mask):
                burst_data = window_data[:, burst_mask]
                burst_rms = np.sqrt(np.mean(burst_data ** 2))
                print(f"  Burst region (±0.4s) RMS: {burst_rms:.6f}")
                
                # Check for theta-like oscillations (4-6 Hz)
                # Simple frequency analysis: count zero crossings
                for ch_idx, ch in enumerate(roi_channels):
                    signal = burst_data[ch_idx, :]
                    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
                    estimated_freq = zero_crossings / (2 * 0.8)  # 0.8s burst duration
                    print(f"    Channel {ch}: {zero_crossings} zero crossings → ~{estimated_freq:.1f} Hz")
    
    # Calculate average power per NM type
    print(f"\n🎯 POWER ANALYSIS BY NM TYPE:")
    measured_powers = {}
    
    for nm_type in nm_types:
        if event_powers[nm_type]:
            avg_power = np.mean(event_powers[nm_type])
            measured_powers[nm_type] = avg_power
            expected = ground_truth[nm_type]
            print(f"NM Type {nm_type}: Average power = {avg_power:.6f} (expected ratio: {expected:.1f}x)")
    
    # Calculate measured ratios vs NM Type 1
    if 1.0 in measured_powers:
        baseline_power = measured_powers[1.0]
        print(f"\n📊 MEASURED RATIOS (vs NM Type 1.0):")
        
        for nm_type in nm_types:
            if nm_type in measured_powers:
                measured_ratio = measured_powers[nm_type] / baseline_power
                expected_ratio = ground_truth[nm_type]
                error = abs(measured_ratio - expected_ratio) / expected_ratio * 100
                
                print(f"  NM {nm_type}: Measured = {measured_ratio:.3f}, Expected = {expected_ratio:.3f}, Error = {error:.1f}%")
    
    # Create diagnostic plot
    create_burst_diagnostic_plot(session_data)

def create_burst_diagnostic_plot(session_data):
    """Create diagnostic plot showing synthetic bursts."""
    print(f"\n📊 Creating diagnostic plot...")
    
    eeg_data = session_data['eeg']
    times = session_data['eeg_time']
    nm_peak_times = session_data['nm_peak_times']
    nm_sizes = session_data['nm_sizes']
    
    # Focus on first ROI channel and first few events
    roi_channel = 7  # 0-based indexing
    n_events_to_plot = min(6, len(nm_peak_times))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(n_events_to_plot):
        ax = axes[i]
        event_time = nm_peak_times[i]
        nm_size = nm_sizes[i]
        
        # Extract window around event
        window_duration = 2.0  # ±1s for visualization
        center_idx = np.argmin(np.abs(times - event_time))
        half_window = int(window_duration * 200 / 2)  # 200 Hz sampling
        start_idx = max(0, center_idx - half_window)
        end_idx = min(len(times), center_idx + half_window)
        
        window_times = times[start_idx:end_idx]
        window_signal = eeg_data[roi_channel, start_idx:end_idx]
        
        # Plot signal
        ax.plot(window_times - event_time, window_signal, 'b-', linewidth=1)
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Event')
        ax.axvspan(-0.4, 0.4, alpha=0.2, color='green', label='Burst region')
        ax.axvspan(-1.0, -0.5, alpha=0.2, color='orange', label='Baseline')
        
        ax.set_title(f'Event {i+1}: NM Size {nm_size}\nTime: {event_time:.1f}s')
        ax.set_xlabel('Time relative to event (s)')
        ax.set_ylabel('EEG amplitude')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    # Hide unused subplots
    for i in range(n_events_to_plot, 6):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('synthetic_burst_diagnostic.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Diagnostic plot saved as synthetic_burst_diagnostic.png")

if __name__ == "__main__":
    test_synthetic_bursts()