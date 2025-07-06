#!/usr/bin/env python3
"""
Parameter Management Examples

This shows different ways to manage parameters with the new configuration system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import configuration
from config import AnalysisConfig, DataConfig, PlottingConfig

def example_1_quick_ide_testing():
    """
    Example 1: Quick IDE testing - Override specific parameters
    Edit the bottom of nm_theta_cross_rats.py like this:
    """
    print("🎯 Example 1: Quick IDE Testing")
    print("=" * 50)
    
    # This is what you'd put at the bottom of nm_theta_cross_rats.py:
    example_code = '''
    # For testing different frequency ranges:
    results = run_cross_rats_analysis(
        roi="8,9,6,11",                   # Frontal channels
        freq_min=6.0,                     # Narrow theta (override config)
        freq_max=8.0,                     # Narrow theta (override config)
        n_freqs=20,                       # Lower resolution (override config)
        verbose=True                      # See progress (override config)
        # method, pkl_path, save_path, etc. will use config defaults
    )
    
    # For testing different regions:
    results = run_cross_rats_analysis(
        roi="1,2,3",                      # Hippocampus channels (override)
        freq_min=3.0,                     # Broader range (override config)
        freq_max=12.0,                    # Broader range (override config)
        window_duration=2.0               # Longer window (override config)
        # Everything else uses config defaults
    )
    '''
    print("📝 Code to add:")
    print(example_code)

def example_2_change_config_defaults():
    """
    Example 2: Change configuration defaults
    Edit src/config/analysis_config.py to change defaults for everything:
    """
    print("\n🔧 Example 2: Change Configuration Defaults")
    print("=" * 50)
    
    example_config = '''
    # Edit src/config/analysis_config.py:
    class AnalysisConfig:
        # Change these values to affect ALL analyses:
        THETA_MIN_FREQ = 6.0              # Changed from 4.0
        THETA_MAX_FREQ = 10.0             # Changed from 8.0  
        N_FREQS_DEFAULT = 25              # Changed from 30
        WINDOW_DURATION_DEFAULT = 1.5     # Changed from 1.0
        
        # ROI shortcuts for convenience:
        ROI_FRONTAL_CHANNELS = "8,9,6,11"
        ROI_HIPPOCAMPUS_CHANNELS = "1,2,3"
    '''
    print("📝 Config changes:")
    print(example_config)
    
    print("\n✨ Result: Now ALL functions use these new defaults!")
    print("   - run_cross_rats_analysis() with roi only → uses new defaults")
    print("   - nm_theta_power_plots.py → uses new theta range") 
    print("   - Command line usage → shows new defaults in --help")

def example_3_use_all_defaults():
    """
    Example 3: Use all defaults - minimal code
    """
    print("\n🎨 Example 3: Use All Defaults (Recommended for Standard Analysis)")
    print("=" * 50)
    
    minimal_code = '''
    # Minimal code - uses smart defaults for everything:
    results = run_cross_rats_analysis(
        roi=AnalysisConfig.ROI_FRONTAL_CHANNELS
        # That's it! Everything else is automatic
    )
    
    # Or even simpler with custom ROI:
    results = run_cross_rats_analysis(roi="8,9,6,11")
    '''
    print("📝 Minimal code:")
    print(minimal_code)
    
    print(f"\n📊 Current default values:")
    print(f"   - Theta range: {AnalysisConfig.get_theta_range()} Hz")
    print(f"   - N frequencies: {AnalysisConfig.N_FREQS_DEFAULT}")
    print(f"   - Window duration: {AnalysisConfig.WINDOW_DURATION_DEFAULT} s")
    print(f"   - Method: {AnalysisConfig.SPECTROGRAM_METHOD_DEFAULT}")
    print(f"   - Data file: {DataConfig.MAIN_EEG_DATA_FILE}")

def show_current_parameters():
    """Show what parameters are currently active"""
    print("\n📋 Current Active Parameters")
    print("=" * 50)
    
    print("🔬 Analysis Settings:")
    summary = AnalysisConfig.get_analysis_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print(f"\n📁 Data Settings:")
    print(f"   Main data file: {DataConfig.MAIN_EEG_DATA_FILE}")
    print(f"   Results directory: {DataConfig.CROSS_RATS_RESULTS_DIR}")
    
    print(f"\n🎨 Plotting Settings:")
    print(f"   Theta power Y-limits: {PlottingConfig.get_theta_power_ylim()}")
    print(f"   Figure size: {PlottingConfig.get_figure_size('default')}")

def show_parameter_priority():
    """Show how parameter priority works"""
    print("\n📊 Parameter Priority (Highest to Lowest)")
    print("=" * 50)
    
    priority_explanation = '''
    1. 🥇 EXPLICIT PARAMETERS (Highest Priority)
       - Parameters you pass directly to the function
       - Example: run_cross_rats_analysis(roi="1,2,3", freq_min=6.0)
       - These ALWAYS override everything else
    
    2. 🥈 CONFIGURATION DEFAULTS (Medium Priority) 
       - Values from AnalysisConfig, DataConfig, PlottingConfig
       - Used when function parameter = None
       - Easy to change in one place
    
    3. 🥉 FALLBACK DEFAULTS (Lowest Priority)
       - Built-in defaults in case config is missing
       - Should rarely be used
    
    📝 Example:
       run_cross_rats_analysis(
           roi="8,9,6,11",        # 🥇 Explicit (always used)
           freq_min=6.0,          # 🥇 Explicit (overrides config)
           freq_max=None,         # 🥈 Will use AnalysisConfig.THETA_MAX_FREQ
           n_freqs=None,          # 🥈 Will use AnalysisConfig.N_FREQS_DEFAULT
           verbose=None           # 🥈 Will use AnalysisConfig.CROSS_RATS_VERBOSE
       )
    '''
    print(priority_explanation)

if __name__ == "__main__":
    print("🧠 EEG Analysis - Parameter Management Guide")
    print("=" * 60)
    
    example_1_quick_ide_testing()
    example_2_change_config_defaults()
    example_3_use_all_defaults()
    show_current_parameters()
    show_parameter_priority()
    
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATION:")
    print("   - For daily use: Edit bottom of nm_theta_cross_rats.py (Method 1)")
    print("   - For systematic changes: Edit config files (Method 2)")
    print("   - For standard analysis: Use minimal code (Method 3)")
    print("=" * 60)