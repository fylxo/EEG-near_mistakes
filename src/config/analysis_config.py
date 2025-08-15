"""
Analysis Configuration Settings

This module contains all analysis-related parameters used throughout the EEG analysis pipeline.
Centralizing these settings makes it easy to experiment with different parameters and ensures
consistency across all analysis functions.
"""


class AnalysisConfig:
    """
    Core analysis parameters for EEG near-mistake theta analysis.
    
    This class contains all the key parameters used in the analysis pipeline.
    Modify these values to experiment with different analysis settings.
    """
    
    # =============================================================================
    # FREQUENCY ANALYSIS SETTINGS
    # =============================================================================
    
    # Theta band frequency range (Hz)
    THETA_MIN_FREQ = 4.0
    THETA_MAX_FREQ = 8.0
    
    # Full frequency analysis range (Hz)
    FREQ_MIN_DEFAULT = 1.0
    FREQ_MAX_DEFAULT = 45.0
    
    # Number of frequencies for analysis
    N_FREQS_DEFAULT = 30
    N_FREQS_HIGH_RES = 40
    
    # Spectral analysis method
    SPECTROGRAM_METHOD_DEFAULT = 'mne'  # 'mne' or 'cwt'
    
    # =============================================================================
    # TIME WINDOW SETTINGS
    # =============================================================================
    
    # Event window duration (seconds)
    WINDOW_DURATION_DEFAULT = 1.0
    WINDOW_DURATION_EXTENDED = 2.0
    
    # =============================================================================
    # SPECTRAL RESOLUTION SETTINGS (CYCLES)
    # =============================================================================
    
    # Cycles method: 'fixed', 'adaptive', or 'hybrid'
    CYCLES_METHOD = 'fixed'  # TESTING: Use fixed 5 cycles to check artifacts
    
    # Fixed cycles (used when method='fixed')
    CYCLES_FIXED = 5
    
    # Adaptive cycles settings (used when method='adaptive' or 'hybrid')
    N_CYCLES_FACTOR_DEFAULT = 1.0  # Multiplier: n_cycles = freq * factor
    CYCLES_MIN = 6                 # Minimum cycles (for hybrid mode)
    
    # Legacy parameter (kept for backward compatibility)
    # Will be removed in future versions - use CYCLES_* settings instead
    @classmethod
    def get_n_cycles_factor(cls):
        """Get n_cycles_factor based on current cycles method."""
        if cls.CYCLES_METHOD == 'fixed':
            return 0.1  # Force max(CYCLES_MIN, freq*0.1) = CYCLES_MIN
        else:
            return cls.N_CYCLES_FACTOR_DEFAULT
    
    # Time windows for specific analyses (seconds relative to event)
    THETA_POWER_TIME_WINDOW = (-0.2, 0.5)  # Pre to post event
    BASELINE_TIME_WINDOW = (-0.5, -0.2)    # Pre-event baseline
    
    # =============================================================================
    # DATA PROCESSING SETTINGS  
    # =============================================================================
    
    # Sampling rate (Hz)
    SAMPLING_RATE = 200.0
    
    # Normalization method
    NORMALIZATION_METHOD = 'zscore'  # 'zscore', 'baseline', 'none'
    
    # Batch processing settings
    BATCH_SIZE_DEFAULT = 8
    CHANNEL_BATCH_SIZE = 8
    
    # Memory management
    ENABLE_GARBAGE_COLLECTION = True
    
    # =============================================================================
    # PARALLEL PROCESSING SETTINGS
    # =============================================================================
    
    # Number of parallel jobs (None = auto-detect)
    N_JOBS_DEFAULT = None
    
    # Parallel processing method
    PARALLEL_TYPE_DEFAULT = None  # None, 'threading', 'multiprocessing'
    
    # =============================================================================
    # CROSS-RATS ANALYSIS SETTINGS
    # =============================================================================
    
    # Default ROI specifications
    ROI_FRONTAL_CHANNELS = "8,9,6,11"
    ROI_HIPPOCAMPUS_CHANNELS = "1,2,3"
    
    # Cross-rats analysis defaults
    CROSS_RATS_SHOW_PLOTS = False
    CROSS_RATS_VERBOSE = True
    
    # =============================================================================
    # STATISTICAL SETTINGS
    # =============================================================================
    
    # Confidence levels
    CONFIDENCE_LEVEL = 0.95
    ALPHA_LEVEL = 0.05
    
    # Error bar type for plots
    ERROR_BAR_TYPE = 'se'  # 'se' (standard error), 'std' (standard deviation), 'ci' (confidence interval)
    
    # =============================================================================
    # QUALITY CONTROL SETTINGS
    # =============================================================================
    
    # Minimum number of events required per NM type
    MIN_EVENTS_PER_NM_TYPE = 10
    
    # Minimum number of sessions required per rat
    MIN_SESSIONS_PER_RAT = 3
    
    # Data validation settings
    VALIDATE_DATA_INTEGRITY = True
    CHECK_ELECTRODE_CONSISTENCY = True
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    @classmethod
    def get_theta_range(cls):
        """Get theta frequency range as tuple."""
        return (cls.THETA_MIN_FREQ, cls.THETA_MAX_FREQ)
    
    @classmethod
    def get_default_freq_range(cls):
        """Get default frequency range as tuple."""
        return (cls.FREQ_MIN_DEFAULT, cls.FREQ_MAX_DEFAULT)
    
    @classmethod
    def get_theta_power_time_window(cls):
        """Get theta power analysis time window as tuple."""
        return cls.THETA_POWER_TIME_WINDOW
    
    @classmethod
    def compute_n_cycles(cls, frequencies):
        """
        Compute n_cycles array based on current cycles method.
        
        Parameters:
        -----------
        frequencies : array-like
            Frequencies for which to compute cycles
            
        Returns:
        --------
        n_cycles : np.ndarray
            Array of cycles for each frequency
        """
        import numpy as np
        frequencies = np.asarray(frequencies)
        
        if cls.CYCLES_METHOD == 'fixed':
            return np.full_like(frequencies, cls.CYCLES_FIXED, dtype=float)
        elif cls.CYCLES_METHOD == 'adaptive':
            return frequencies * cls.N_CYCLES_FACTOR_DEFAULT
        elif cls.CYCLES_METHOD == 'hybrid':
            return np.maximum(cls.CYCLES_MIN, frequencies * cls.N_CYCLES_FACTOR_DEFAULT)
        else:
            raise ValueError(f"Unknown cycles method: {cls.CYCLES_METHOD}")

    @classmethod
    def get_analysis_summary(cls):
        """Get a summary of current analysis settings."""
        return {
            'theta_range_hz': cls.get_theta_range(),
            'default_freq_range_hz': cls.get_default_freq_range(),
            'window_duration_s': cls.WINDOW_DURATION_DEFAULT,
            'sampling_rate_hz': cls.SAMPLING_RATE,
            'n_freqs': cls.N_FREQS_DEFAULT,
            'normalization': cls.NORMALIZATION_METHOD,
            'spectrogram_method': cls.SPECTROGRAM_METHOD_DEFAULT,
            'cycles_method': cls.CYCLES_METHOD,
            'cycles_fixed': cls.CYCLES_FIXED if cls.CYCLES_METHOD == 'fixed' else None,
            'cycles_min': cls.CYCLES_MIN if cls.CYCLES_METHOD == 'hybrid' else None,
            'n_cycles_factor': cls.N_CYCLES_FACTOR_DEFAULT if cls.CYCLES_METHOD != 'fixed' else None
        }
    
    @classmethod
    def validate_settings(cls):
        """
        Validate configuration settings for common issues.
        
        Returns:
        --------
        bool
            True if all settings are valid
            
        Raises:
        -------
        ValueError
            If any settings are invalid
        """
        # Check frequency ranges
        if cls.THETA_MIN_FREQ >= cls.THETA_MAX_FREQ:
            raise ValueError("THETA_MIN_FREQ must be less than THETA_MAX_FREQ")
        
        if cls.FREQ_MIN_DEFAULT >= cls.FREQ_MAX_DEFAULT:
            raise ValueError("FREQ_MIN_DEFAULT must be less than FREQ_MAX_DEFAULT")
        
        # Check theta range is within default range
        if cls.THETA_MIN_FREQ < cls.FREQ_MIN_DEFAULT or cls.THETA_MAX_FREQ > cls.FREQ_MAX_DEFAULT:
            raise ValueError("Theta range must be within default frequency range")
        
        # Check positive values
        if cls.SAMPLING_RATE <= 0:
            raise ValueError("SAMPLING_RATE must be positive")
        
        if cls.WINDOW_DURATION_DEFAULT <= 0:
            raise ValueError("WINDOW_DURATION_DEFAULT must be positive")
        
        if cls.N_FREQS_DEFAULT <= 0:
            raise ValueError("N_FREQS_DEFAULT must be positive")
        
        # Check valid methods
        valid_methods = ['mne', 'cwt']
        if cls.SPECTROGRAM_METHOD_DEFAULT not in valid_methods:
            raise ValueError(f"SPECTROGRAM_METHOD_DEFAULT must be one of {valid_methods}")
        
        valid_normalizations = ['zscore', 'baseline', 'none']
        if cls.NORMALIZATION_METHOD not in valid_normalizations:
            raise ValueError(f"NORMALIZATION_METHOD must be one of {valid_normalizations}")
        
        return True


# Validate settings on import
AnalysisConfig.validate_settings()