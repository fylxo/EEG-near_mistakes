#!/usr/bin/env python3
"""
Synthetic EEG Data Pipeline Validation

This script generates synthetic EEG data with known theta oscillation properties
to validate the nm_theta_cross_rats.py and nm_theta_power_plots.py pipeline.

Test Design:
- 3 synthetic rats, each with 3 sessions
- Known theta power differences between NM types
- Controlled frequency content and timing
- Validates entire pipeline from raw data to final statistics

Author: Generated for EEG analysis pipeline validation
"""

import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def generate_synthetic_eeg_session(rat_id: str, 
                                 session_idx: int,
                                 session_duration: float = 600.0,  # 10 minutes
                                 sfreq: float = 200.0,
                                 n_channels: int = 32,
                                 nm_events_per_type: int = 50) -> Dict:
    """
    Generate one synthetic EEG session with known theta properties.
    
    Parameters:
    -----------
    rat_id : str
        Rat identifier
    session_idx : int
        Session number (0-based)
    session_duration : float
        Session length in seconds
    sfreq : float
        Sampling frequency
    n_channels : int
        Number of EEG channels
    nm_events_per_type : int
        Number of NM events per type to generate
    
    Returns:
    --------
    session_data : Dict
        Complete session data structure matching real data format
    """
    
    print(f"Generating synthetic session for rat {rat_id}, session {session_idx}")
    
    # Time vector
    n_samples = int(session_duration * sfreq)
    times = np.arange(n_samples) / sfreq
    
    # Initialize EEG data (channels x time)
    eeg_data = np.zeros((n_channels, n_samples))
    
    # Add realistic background activity
    # Use rat_id directly as integer (e.g., "9151" -> 9151)
    rat_numeric = int(rat_id)  
    np.random.seed(42 + rat_numeric + session_idx * 1000)  # Reproducible but different per rat/session
    
    # Background: 1/f noise + some rhythmic activity
    for ch in range(n_channels):
        # 1/f background noise
        background = generate_pink_noise(n_samples, sfreq)
        
        # Add minimal ongoing theta to provide non-zero baseline for normalization
        # This ensures baseline period (-1.0 to -0.5s) has some activity for proper normalization
        if ch in [7, 15, 23]:  # ROI channels (0-based) = electrodes 8, 16, 24 (1-based)
            theta_strength = 0.1  # Minimal ongoing theta in ROI channels
        else:
            theta_strength = 0.02  # Very weak theta in other channels
            
        ongoing_theta = theta_strength * np.sin(2 * np.pi * 5.0 * times + np.random.uniform(0, 2*np.pi))
        
        eeg_data[ch, :] = background + ongoing_theta
    
    # Generate NM events with KNOWN theta power differences
    nm_peak_times = []
    nm_sizes = []
    
    # Known ground truth: NM Type 1 < NM Type 2 < NM Type 3 (theta power increases)
    nm_types = [1.0, 2.0, 3.0]
    theta_power_multipliers = {
        1.0: 1.0,  # Baseline theta power
        2.0: 1.5,  # 50% more theta power  
        3.0: 2.0   # 100% more theta power (double)
    }
    
    print(f"  Ground truth theta power ratios: {theta_power_multipliers}")
    
    for nm_type in nm_types:
        power_multiplier = theta_power_multipliers[nm_type]
        
        # Generate event times (avoiding edges and overlaps)
        event_times = np.random.uniform(5.0, session_duration - 5.0, nm_events_per_type)
        event_times = np.sort(event_times)
        
        # Ensure minimum 2s separation between events
        filtered_times = [event_times[0]]
        for t in event_times[1:]:
            if t - filtered_times[-1] >= 2.0:
                filtered_times.append(t)
        
        event_times = np.array(filtered_times[:nm_events_per_type])  # Keep only requested number
        
        for event_time in event_times:
            # Add theta burst around this event
            add_synthetic_theta_burst(
                eeg_data, times, event_time, 
                power_multiplier=power_multiplier,
                roi_channels=[7, 15, 23],  # Same channels with stronger ongoing theta
                sfreq=sfreq
            )
            
            nm_peak_times.append(event_time)
            nm_sizes.append(nm_type)
    
    nm_peak_times = np.array(nm_peak_times)
    nm_sizes = np.array(nm_sizes)
    
    # Sort by time
    sort_idx = np.argsort(nm_peak_times)
    nm_peak_times = nm_peak_times[sort_idx]
    nm_sizes = nm_sizes[sort_idx]
    
    print(f"  Generated {len(nm_peak_times)} total NM events")
    for nm_type in nm_types:
        count = np.sum(nm_sizes == nm_type)
        print(f"    NM Type {nm_type}: {count} events")
    
    # Create session data structure matching real format
    session_data = {
        'rat_id': rat_id,
        'session_date': (datetime(2024, 1, 1) + timedelta(days=session_idx)).strftime('%Y-%m-%d'),
        'session_index': session_idx,
        'eeg': eeg_data,  # (n_channels, n_samples) 
        'eeg_time': times,  # Pipeline expects 'eeg_time' not 'times'
        'nm_peak_times': nm_peak_times,
        'nm_sizes': nm_sizes,
        'sfreq': sfreq,
        'n_channels': n_channels,
        'duration': session_duration,
        'synthetic': True,  # Flag to identify synthetic data
        'ground_truth_theta_ratios': theta_power_multipliers
    }
    
    return session_data


def generate_pink_noise(n_samples: int, sfreq: float, alpha: float = 1.0) -> np.ndarray:
    """Generate 1/f^alpha noise (pink noise when alpha=1)."""
    # Generate white noise in frequency domain
    freqs = np.fft.fftfreq(n_samples, 1/sfreq)
    freqs[0] = 1  # Avoid division by zero
    
    # Create 1/f spectrum
    spectrum = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    spectrum *= (np.abs(freqs) ** (-alpha/2))
    spectrum[0] = 0  # Remove DC component
    
    # Convert back to time domain
    signal = np.fft.ifft(spectrum).real
    
    # Normalize to reasonable amplitude
    signal = signal / np.std(signal) * 0.5
    
    return signal


def add_synthetic_theta_burst(eeg_data: np.ndarray, 
                            times: np.ndarray,
                            event_time: float,
                            power_multiplier: float = 1.0,
                            roi_channels: List[int] = [7, 15, 23],
                            sfreq: float = 200.0):
    """
    Add a theta burst around an event time with specified power.
    
    This creates a realistic theta oscillation (4-6 Hz) that is:
    - Stronger in ROI channels
    - Time-locked to the NM event
    - Scaled by power_multiplier to create known differences between NM types
    """
    
    # Create theta burst that AVOIDS the baseline period (-1.0 to -0.5s)
    # Focus the burst from -0.4s to +0.4s to avoid interfering with baseline normalization
    burst_duration = 0.8  # Total duration (±0.4s around event)
    start_time = event_time - burst_duration/2
    end_time = event_time + burst_duration/2
    
    start_idx = int(start_time * sfreq)
    end_idx = int(end_time * sfreq)
    
    if start_idx < 0 or end_idx >= len(times):
        return  # Skip if burst would be outside recording
    
    burst_times = times[start_idx:end_idx]
    
    # Create strong, focused theta burst envelope centered exactly at the event
    # Single Gaussian centered at event time for maximum power and minimal baseline interference
    envelope = np.exp(-((burst_times - event_time) ** 2) / (2 * (0.2 ** 2)))  # 200ms std, tightly focused
    
    # Generate theta oscillation with realistic properties
    theta_freq = np.random.uniform(4.5, 5.5)  # Theta range frequency
    theta_phase = np.random.uniform(0, 2*np.pi)  # Random starting phase
    
    # Base theta oscillation
    theta_oscillation = np.sin(2 * np.pi * theta_freq * burst_times + theta_phase)
    
    # Apply envelope and scale by power multiplier
    # Use sqrt because power scales as amplitude^2, so amplitude scales as sqrt(power)
    base_amplitude = 2.0  # Much stronger amplitude for clear detection
    theta_burst = envelope * theta_oscillation * base_amplitude * np.sqrt(power_multiplier)
    
    # Debug: Print burst strength for verification
    max_burst_amplitude = np.max(np.abs(theta_burst))
    print(f"    Generated theta burst: power_multiplier={power_multiplier:.1f}, max_amplitude={max_burst_amplitude:.3f}")
    
    # Add to ROI channels with strongest effect
    for ch_idx in roi_channels:
        if ch_idx < eeg_data.shape[0]:
            # Add realistic channel-specific variations
            channel_gain = np.random.uniform(0.9, 1.1)  # ±10% gain variation
            phase_shift = np.random.uniform(-np.pi/8, np.pi/8)  # ±22.5° phase shift
            
            # Apply phase shift
            channel_theta = theta_burst * channel_gain * np.cos(phase_shift)
            eeg_data[ch_idx, start_idx:end_idx] += channel_theta
    
    # Add much weaker theta to other channels (volume conduction effect)
    other_channels = [ch for ch in range(eeg_data.shape[0]) if ch not in roi_channels]
    for ch_idx in other_channels:
        # Very weak volume conduction effect
        volume_conduction_strength = np.random.uniform(0.1, 0.2)  # 10-20% of ROI strength
        weak_theta = theta_burst * volume_conduction_strength
        eeg_data[ch_idx, start_idx:end_idx] += weak_theta


def create_synthetic_dataset(n_rats: int = 3, 
                           n_sessions_per_rat: int = 3,
                           save_path: str = "data/synthetic/synthetic_eeg_data.pkl") -> str:
    """
    Create complete synthetic dataset with multiple rats and sessions.
    
    Parameters:
    -----------
    n_rats : int
        Number of synthetic rats to generate
    n_sessions_per_rat : int  
        Number of sessions per rat
    save_path : str
        Path to save the synthetic dataset
        
    Returns:
    --------
    save_path : str
        Path where data was saved
    """
    
    print("🧪 CREATING SYNTHETIC EEG DATASET")
    print("=" * 60) 
    print(f"Rats: {n_rats}")
    print(f"Sessions per rat: {n_sessions_per_rat}")
    print(f"Total sessions: {n_rats * n_sessions_per_rat}")
    print()
    
    # Use existing real rat IDs from the electrode mapping file to avoid mapping issues
    # These are rats that exist in data/config/consistent_electrode_mappings.csv
    available_real_rat_ids = ["9151", "9441", "9591"]  # Use first 3 real rat IDs
    rat_ids = available_real_rat_ids[:n_rats]
    print(f"Using real rat IDs for synthetic data: {rat_ids}")
    print()
    
    # Generate all sessions
    all_sessions = []
    
    for rat_idx, rat_id in enumerate(rat_ids):
        print(f"Generating rat {rat_id} ({rat_idx+1}/{n_rats}):")
        
        for session_idx in range(n_sessions_per_rat):
            session_data = generate_synthetic_eeg_session(
                rat_id=rat_id,
                session_idx=session_idx,
                session_duration=300.0,  # 5 minutes per session (shorter for testing)
                nm_events_per_type=20    # 20 events per NM type per session
            )
            all_sessions.append(session_data)
        print()
    
    # Save dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(all_sessions, f)
    
    print(f"✅ Synthetic dataset saved to: {save_path}")
    print(f"Total sessions: {len(all_sessions)}")
    
    # Print summary statistics
    print("\n📊 DATASET SUMMARY:")
    print("=" * 40)
    
    total_events_by_type = {1.0: 0, 2.0: 0, 3.0: 0}
    
    for session in all_sessions:
        for nm_type in [1.0, 2.0, 3.0]:
            count = np.sum(session['nm_sizes'] == nm_type)
            total_events_by_type[nm_type] += count
    
    print(f"Total NM events by type:")
    for nm_type, count in total_events_by_type.items():
        print(f"  NM Type {nm_type}: {count} events across all sessions")
    
    print(f"\nGround truth theta power ratios:")
    print(f"  NM Type 1.0: 1.0x (baseline)")
    print(f"  NM Type 2.0: 1.5x (50% increase)")  
    print(f"  NM Type 3.0: 2.0x (100% increase)")
    
    return save_path


def extract_theta_power_from_aggregate_results(results_path: str, theta_range: Tuple[float, float] = (4.5, 5.5)) -> Dict:
    """
    Extract theta power directly from cross-rats aggregate results for validation.
    
    Parameters:
    -----------
    results_path : str
        Path to cross_rats_aggregated_results.pkl
    theta_range : Tuple[float, float]
        Frequency range to extract power from
        
    Returns:
    --------
    power_results : Dict
        Dictionary with nm_types, means, and standard_errors
    """
    
    import pickle
    
    # Load aggregate results
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    frequencies = results['frequencies']
    averaged_windows = results['averaged_windows']
    
    # Find frequency indices for theta range
    theta_mask = (frequencies >= theta_range[0]) & (frequencies <= theta_range[1])
    
    if not np.any(theta_mask):
        raise ValueError(f"No frequencies found in theta range {theta_range}")
    
    print(f"Using {np.sum(theta_mask)} frequencies in {theta_range[0]}-{theta_range[1]} Hz range")
    
    nm_types = []
    nm_means = []
    nm_ses = []
    
    for nm_size, window_data in averaged_windows.items():
        avg_spectrogram = window_data['avg_spectrogram']
        
        # Extract theta frequencies
        theta_spectrogram = avg_spectrogram[theta_mask, :]
        
        # Average across frequencies and time to get single power value
        theta_power = np.mean(theta_spectrogram)
        
        nm_types.append(float(nm_size))
        nm_means.append(theta_power)
        nm_ses.append(0.0)  # No SE for aggregate data
        
        print(f"NM {nm_size}: Theta power = {theta_power:.4f}")
    
    return {
        'nm_types': nm_types,
        'means': nm_means,
        'standard_errors': nm_ses,
        'theta_range': theta_range
    }


def validate_pipeline_with_synthetic_data(synthetic_data_path: str):
    """
    Run the complete pipeline on synthetic data and validate results.
    
    Parameters:
    -----------
    synthetic_data_path : str
        Path to synthetic EEG dataset
    """
    
    print("\n🔬 VALIDATING PIPELINE WITH SYNTHETIC DATA")
    print("=" * 60)
    
    # Import the pipeline functions
    try:
        from nm_theta_cross_rats import run_cross_rats_analysis
        from nm_theta_power_plots import create_theta_power_analysis
        print("✅ Successfully imported pipeline functions")
    except ImportError as e:
        print(f"❌ Error importing pipeline functions: {e}")
        return
    
    # Step 1: Run cross-rats analysis
    print("\n🔄 Step 1: Running cross-rats analysis...")
    
    try:
        cross_rats_results = run_cross_rats_analysis(
            roi="8,16,24",  # ROI channels where we added synthetic theta bursts
            pkl_path=synthetic_data_path,
            freq_min=3.0,
            freq_max=7.0,
            window_duration=2.0,
            save_path="results/synthetic_validation",
            cleanup_intermediate_files=False  # Keep for inspection
        )
        print("✅ Cross-rats analysis completed")
        
    except Exception as e:
        print(f"❌ Error in cross-rats analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Extract theta power directly from aggregate results for validation
    print("\n🔄 Step 2: Extracting theta power from synthetic results...")
    
    try:
        power_results = extract_theta_power_from_aggregate_results(
            results_path="results/synthetic_validation/cross_rats_aggregated_results.pkl",
            theta_range=(4.5, 5.5)  # Focus on 5Hz where we added synthetic theta bursts
        )
        print("✅ Theta power extraction completed")
        
    except Exception as e:
        print(f"❌ Error in theta power extraction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Validate results against ground truth
    print("\n🎯 Step 3: Validating results against ground truth...")
    validate_results_against_ground_truth(power_results)


def validate_results_against_ground_truth(power_results: Dict):
    """
    Compare pipeline results with known ground truth.
    
    Parameters:
    -----------
    power_results : Dict
        Results from theta power analysis
    """
    
    print("\n📈 VALIDATION RESULTS:")
    print("=" * 40)
    
    # Ground truth ratios
    ground_truth = {1.0: 1.0, 2.0: 1.5, 3.0: 2.0}
    
    # Check if we have valid results
    if not power_results or not power_results.get('nm_types'):
        print("❌ No valid power results to validate!")
        print("This suggests the theta power extraction failed.")
        return
    
    # Extract measured values
    nm_types = power_results['nm_types']
    measured_means = power_results['means']
    measured_ses = power_results['standard_errors']
    
    print("Expected vs Measured theta power:")
    print("NM Type | Expected Ratio | Measured Mean ± SE | Ratio vs NM1")
    print("-" * 65)
    
    # Calculate ratios relative to NM Type 1
    nm1_mean = None
    measured_ratios = {}
    
    for i, nm_type in enumerate(nm_types):
        mean_val = measured_means[i]
        se_val = measured_ses[i]
        
        if nm_type == 1.0:
            nm1_mean = mean_val
            measured_ratios[nm_type] = 1.0
        else:
            measured_ratios[nm_type] = mean_val / nm1_mean if nm1_mean != 0 else 0
        
        expected_ratio = ground_truth.get(nm_type, 0)
        measured_ratio = measured_ratios[nm_type]
        
        print(f"NM {nm_type:<4} | {expected_ratio:<13.1f} | {mean_val:<6.3f} ± {se_val:<6.3f} | {measured_ratio:<12.2f}")
    
    # Calculate validation metrics
    print(f"\n🎯 VALIDATION METRICS:")
    print("-" * 30)
    
    ratio_errors = []
    for nm_type in [2.0, 3.0]:  # Compare ratios for NM2 and NM3 vs NM1
        if nm_type in ground_truth and nm_type in measured_ratios:
            expected = ground_truth[nm_type]
            measured = measured_ratios[nm_type]
            error = abs(measured - expected) / expected * 100  # Percent error
            ratio_errors.append(error)
            
            print(f"NM {nm_type} ratio error: {error:.1f}%")
    
    mean_error = np.mean(ratio_errors)
    print(f"Mean ratio error: {mean_error:.1f}%")
    
    # Validation status
    if mean_error < 20:  # Allow 20% tolerance
        print("✅ VALIDATION PASSED: Pipeline correctly detects synthetic theta differences")
    else: 
        print("❌ VALIDATION FAILED: Pipeline does not accurately measure synthetic theta differences")
        print("   This suggests issues in the analysis methodology")
    
    # Statistical significance check
    print(f"\n📊 STATISTICAL ASSESSMENT:")
    if len(nm_types) >= 3:
        nm1_mean, nm1_se = measured_means[0], measured_ses[0]
        nm3_mean, nm3_se = measured_means[2], measured_ses[2]
        
        # Rough significance test (difference / pooled SE)
        pooled_se = np.sqrt(nm1_se**2 + nm3_se**2)
        t_stat = abs(nm3_mean - nm1_mean) / pooled_se
        
        print(f"NM1 vs NM3 effect size: {t_stat:.2f} (t-statistic approximation)")
        if t_stat > 2:  # Rough significance threshold
            print("✅ Strong evidence for theta power differences between NM types")
        else:
            print("⚠️  Weak evidence for theta power differences - check methodology")


def main():
    """Main validation workflow."""
    
    print("🧪 SYNTHETIC EEG PIPELINE VALIDATION")
    print("=" * 80)
    print("This script validates nm_theta_cross_rats.py and nm_theta_power_plots.py")
    print("using synthetic EEG data with known theta power properties.")
    print("=" * 80)
    
    # Step 1: Create synthetic dataset
    synthetic_path = create_synthetic_dataset(
        n_rats=3,
        n_sessions_per_rat=3,
        save_path="data/synthetic/synthetic_eeg_validation.pkl"
    )
    
    # Step 2: Run pipeline validation
    validate_pipeline_with_synthetic_data(synthetic_path)
    
    print("\n🎉 VALIDATION COMPLETE!")
    print("Check the results above to verify pipeline accuracy.")


if __name__ == "__main__":
    main()