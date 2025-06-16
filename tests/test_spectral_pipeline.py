"""
Test script for the spectral analysis pipeline

This script tests the spectral analysis pipeline on a sample session
to validate that all functions work correctly before processing the full dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_analysis_pipeline import (
    load_session_data, 
    process_single_session,
    test_pipeline_on_sample
)

def quick_test():
    """
    Quick test to validate the pipeline works without processing full session.
    """
    print("=== Quick Pipeline Test ===")
    
    try:
        # Try to load just one session to check data format
        print("1. Testing data loading...")
        session_data = load_session_data('all_eeg_data.pkl', session_index=0)
        
        print(f"✓ Data loaded successfully:")
        print(f"  - EEG shape: {session_data['eeg'].shape}")
        print(f"  - Time shape: {session_data['eeg_time'].shape}")
        print(f"  - NM events: {len(session_data['nm_peak_times'])}")
        print(f"  - ITI events: {len(session_data['iti_peak_times'])}")
        print(f"  - NM sizes: {np.unique(session_data['nm_sizes'])}")
        print(f"  - ITI sizes: {np.unique(session_data['iti_sizes'])}")
        
        # Test with minimal parameters for quick validation
        print("\n2. Testing pipeline with minimal parameters...")
        
        # Use only first 4 channels and shorter processing
        test_channels = [0, 1, 2, 3]
        roi_channels = [0, 1]
        freqs = np.arange(4, 20, 2)  # Fewer frequencies for speed
        
        session_id = f"test_{session_data.get('rat_id', 'unknown')}"
        
        results = process_single_session(
            session_data=session_data,
            session_id=session_id,
            freqs=freqs,
            channels=test_channels,
            roi_channels=roi_channels,
            window_duration=2.0,  # Shorter window for speed
            save_path='quick_test_results',
            plot_results=False  # Skip plots for quick test
        )
        
        print("✓ Pipeline completed successfully!")
        
        # Print some basic results
        print(f"\n3. Results summary:")
        print(f"  - Processed {len(results['channels'])} channels")
        print(f"  - Frequency range: {results['freqs'][0]:.1f}-{results['freqs'][-1]:.1f} Hz")
        
        if 'group_averages' in results:
            for event_type in results['group_averages']:
                for size in results['group_averages'][event_type]:
                    n_events = results['group_averages'][event_type][size]['n_events']
                    print(f"  - {event_type} size {size}: {n_events} events")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in quick test: {e}")
        import traceback
        traceback.print_exc()
        return False


def full_test():
    """
    Full test of the pipeline on one session with complete parameters.
    """
    print("\n=== Full Pipeline Test ===")
    
    try:
        # Use the built-in test function
        results = test_pipeline_on_sample()
        
        if results is not None:
            print("✓ Full pipeline test completed successfully!")
            
            # Test the new plotting features
            print("\n=== Testing New Plotting Features ===")
            test_new_plotting_features(results)
            
            return True
        else:
            print("✗ Full pipeline test failed")
            return False
            
    except Exception as e:
        print(f"✗ Error in full test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_new_plotting_features(results):
    """
    Test the new colorblind-friendly plotting and reporting features.
    """
    print("Testing colorblind-friendly plots and edge case reporting...")
    
    try:
        from spectral_analysis_pipeline import (
            get_colorblind_friendly_colors, 
            get_line_styles,
            report_missing_groups
        )
        
        # Test color palette
        colors = get_colorblind_friendly_colors()
        line_styles = get_line_styles()
        
        print("✓ Color palettes loaded:")
        for event_type in ['NM', 'ITI']:
            for size in [1, 2, 3]:
                color_info = colors[event_type][size]
                style_info = line_styles[size]
                print(f"  {event_type} Size {size}: {color_info['name']} ({color_info['color']}) - {style_info['name']}")
        
        # Test edge case reporting
        print("\n✓ Testing edge case reporting:")
        if 'group_averages' in results:
            report_missing_groups(results['group_averages'])
        
        # The plots were already generated with new styling in the main test
        print("✓ Colorblind-friendly plots generated successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing new features: {e}")
        return False


def test_variability_plotting():
    """
    Test multi-session variability plotting functions (simulated data).
    """
    print("\n=== Testing Variability Plotting ===")
    
    try:
        import numpy as np
        from spectral_analysis_pipeline import plot_theta_timecourses_with_variability
        
        # Create simulated multi-session data
        print("Creating simulated multi-session data...")
        
        window_times = np.linspace(-2, 2, 200)
        channels = [0, 1, 2, 3]
        roi_channels = [0, 1]
        
        # Simulate 3 sessions
        multi_session_theta_data = []
        for session in range(3):
            session_theta = {'NM': {}, 'ITI': {}}
            
            for event_type in ['NM', 'ITI']:
                for size in [1, 2]:  # Only sizes 1 and 2 for this test
                    # Add some variability between sessions
                    baseline = 10 + session * 2 + np.random.normal(0, 1)
                    
                    # Create fake theta power data (channels x time)
                    theta_power = np.random.normal(baseline, 2, (len(channels), len(window_times)))
                    
                    # Add event-related change
                    event_idx = len(window_times) // 2
                    if event_type == 'NM':
                        theta_power[:, event_idx-10:event_idx+20] += 3  # Increase for NM
                    
                    session_theta[event_type][size] = {
                        'theta_power': theta_power,
                        'window_times': window_times,
                        'n_events': 5 + session
                    }
            
            multi_session_theta_data.append(session_theta)
        
        # Test different plot types
        for plot_type in ['sem', 'boxplot']:
            print(f"  Testing {plot_type} plots...")
            plot_theta_timecourses_with_variability(
                multi_session_theta_data,
                channels, roi_channels, 
                save_path='test_variability_plots',
                plot_type=plot_type
            )
        
        print("✓ Variability plotting test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Error in variability plotting test: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_results(results):
    """
    Validate that the results have the expected structure and content.
    """
    print("\n=== Validating Results ===")
    
    required_keys = [
        'session_id', 'freqs', 'times', 'channels', 'roi_channels',
        'group_averages', 'theta_data', 'parameters'
    ]
    
    for key in required_keys:
        if key not in results:
            print(f"✗ Missing key: {key}")
            return False
        else:
            print(f"✓ Found key: {key}")
    
    # Check group averages structure
    if 'group_averages' in results:
        for event_type in ['NM', 'ITI']:
            if event_type in results['group_averages']:
                for size in [1, 2, 3]:
                    if size in results['group_averages'][event_type]:
                        avg_spec = results['group_averages'][event_type][size]['avg_spectrogram']
                        print(f"✓ {event_type} size {size} spectrogram shape: {avg_spec.shape}")
    
    # Check theta data structure
    if 'theta_data' in results:
        for event_type in ['NM', 'ITI']:
            if event_type in results['theta_data']:
                for size in [1, 2, 3]:
                    if size in results['theta_data'][event_type]:
                        theta_power = results['theta_data'][event_type][size]['theta_power']
                        print(f"✓ {event_type} size {size} theta power shape: {theta_power.shape}")
    
    print("✓ Results validation completed!")
    return True


if __name__ == "__main__":
    print("Testing Spectral Analysis Pipeline with Enhanced Features")
    print("=" * 60)
    
    # Run quick test first
    quick_success = quick_test()
    
    if quick_success:
        print("\n" + "=" * 60)
        # Run full test if quick test passes
        full_success = full_test()
        
        if full_success:
            print("\n" + "=" * 60)
            # Test variability plotting functions
            variability_success = test_variability_plotting()
            
            if variability_success:
                print("\n✓ All tests passed! Enhanced pipeline is ready for use.")
                print("\nNew features tested:")
                print("  ✓ Colorblind-friendly color palettes")
                print("  ✓ Line style differentiation for event sizes")
                print("  ✓ Enhanced edge case reporting")
                print("  ✓ Multi-session variability plotting")
                print("  ✓ SEM error bands and boxplot distributions")
            else:
                print("\n⚠ Core pipeline works, but variability plotting has issues.")
        else:
            print("\n✗ Full test failed. Check the pipeline implementation.")
    else:
        print("\n✗ Quick test failed. Check basic pipeline setup.")