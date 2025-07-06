"""
Data Configuration Settings

This module contains all data-related parameters including file paths,
data formats, and data processing settings.
"""

import os
from pathlib import Path


class DataConfig:
    """
    Data file paths and data processing parameters.
    
    This class contains all settings related to data files, paths, and data handling.
    """
    
    # =============================================================================
    # FILE PATHS
    # =============================================================================
    
    # Default data file paths (relative to project root)
    DATA_ROOT = "data"
    PROCESSED_DATA_ROOT = "data/processed"
    CONFIG_DATA_ROOT = "data/config"
    RESULTS_ROOT = "results"
    
    # Main data files
    MAIN_EEG_DATA_FILE = "data/processed/all_eeg_data.pkl"
    ELECTRODE_MAPPINGS_FILE = "data/config/consistent_electrode_mappings.csv"
    ELECTRODE_PLACEMENT_FILE = "data/config/electrodes_placement.mat"
    
    # Configuration files
    ROI_MAPPING_FILE = "data/config/ROI_mapping.txt"
    RAT_IDS_FILE = "data/config/rat_IDs.txt"
    DATASET_INFO_FILE = "data/config/dataset.txt"
    
    # Results directories
    CROSS_RATS_RESULTS_DIR = "results/cross_rats"
    SINGLE_SESSION_RESULTS_DIR = "results/single_session"
    THETA_POWER_RESULTS_DIR = "results/theta_power"
    
    # =============================================================================
    # DATA VALIDATION SETTINGS
    # =============================================================================
    
    # Expected data structure validation
    EXPECTED_SESSION_KEYS = [
        'rat_id', 'session_date', 'eeg', 'eeg_time', 
        'nm_peak_times', 'nm_sizes', 'iti_peak_times', 'iti_sizes'
    ]
    
    # Data quality checks
    MIN_SESSION_DURATION_SECONDS = 60  # Minimum session length
    MAX_SESSION_DURATION_SECONDS = 3600  # Maximum session length (1 hour)
    
    # Channel validation
    EXPECTED_CHANNEL_COUNTS = [20, 32]  # Valid channel counts
    MIN_CHANNELS = 8  # Minimum channels for analysis
    
    # Event validation
    MIN_EVENTS_PER_SESSION = 5  # Minimum events per session
    MAX_EVENTS_PER_SESSION = 1000  # Maximum events per session
    
    # =============================================================================
    # RAT-SPECIFIC SETTINGS
    # =============================================================================
    
    # Rat 9442 special handling
    RAT_9442_32_CHANNEL_SESSIONS = ['070419', '080419', '090419', '190419']
    RAT_9442_20_CHANNEL_ELECTRODES = [10, 11, 12, 13, 14, 15, 16, 19, 1, 24, 25, 29, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # Known problematic rats or sessions (if any)
    EXCLUDED_RATS = []  # Add rat IDs to exclude
    EXCLUDED_SESSIONS = []  # Add specific sessions to exclude
    
    # =============================================================================
    # FILE FORMAT SETTINGS
    # =============================================================================
    
    # Output file formats
    RESULTS_FORMAT_PKL = True  # Save pickle files
    RESULTS_FORMAT_JSON = True  # Save JSON summaries
    RESULTS_FORMAT_CSV = False  # Save CSV exports (optional)
    
    # Plot output formats
    PLOT_FORMATS = ['png']  # Can add 'pdf', 'svg', etc.
    PLOT_DPI = 300
    
    # Compression settings
    PICKLE_PROTOCOL = 4  # Pickle protocol version
    ENABLE_COMPRESSION = False  # Enable gzip compression for large files
    
    # =============================================================================
    # MEMORY MANAGEMENT
    # =============================================================================
    
    # Memory limits for large datasets
    MAX_MEMORY_GB = 8  # Maximum memory usage in GB
    ENABLE_MEMORY_MONITORING = True
    
    # Chunking settings for large files
    CHUNK_SIZE_SESSIONS = 10  # Process N sessions at a time
    CHUNK_SIZE_CHANNELS = 8   # Process N channels at a time
    
    # Garbage collection settings
    GC_FREQUENCY = 'after_rat'  # 'never', 'after_session', 'after_rat', 'frequent'
    
    # =============================================================================
    # BACKUP AND VERSIONING
    # =============================================================================
    
    # Automatic backup settings
    CREATE_RESULT_BACKUPS = False  # Create backup of existing results
    BACKUP_SUFFIX = "_backup"
    
    # Versioning
    ADD_TIMESTAMP_TO_RESULTS = True  # Add timestamp to result filenames
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    @classmethod
    def get_data_file_path(cls, relative_path):
        """
        Get absolute path for a data file.
        
        Parameters:
        -----------
        relative_path : str
            Relative path from project root
            
        Returns:
        --------
        str
            Absolute path to file
        """
        # Assume project root is parent of src directory
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / relative_path)
    
    @classmethod
    def ensure_results_dir(cls, results_dir=None):
        """
        Ensure results directory exists.
        
        Parameters:
        -----------
        results_dir : str, optional
            Results directory path (default: RESULTS_ROOT)
            
        Returns:
        --------
        str
            Absolute path to results directory
        """
        if results_dir is None:
            results_dir = cls.RESULTS_ROOT
        
        abs_path = cls.get_data_file_path(results_dir)
        os.makedirs(abs_path, exist_ok=True)
        return abs_path
    
    @classmethod
    def get_default_save_path(cls, analysis_type):
        """
        Get default save path for different analysis types.
        
        Parameters:
        -----------
        analysis_type : str
            Type of analysis ('cross_rats', 'single_session', 'theta_power')
            
        Returns:
        --------
        str
            Default save path for analysis type
        """
        path_map = {
            'cross_rats': cls.CROSS_RATS_RESULTS_DIR,
            'single_session': cls.SINGLE_SESSION_RESULTS_DIR,
            'theta_power': cls.THETA_POWER_RESULTS_DIR
        }
        
        relative_path = path_map.get(analysis_type, cls.RESULTS_ROOT)
        return cls.get_data_file_path(relative_path)
    
    @classmethod
    def validate_data_file_exists(cls, file_type):
        """
        Validate that required data files exist.
        
        Parameters:
        -----------
        file_type : str
            Type of file to check ('main_data', 'electrode_mappings', etc.)
            
        Returns:
        --------
        bool
            True if file exists
            
        Raises:
        -------
        FileNotFoundError
            If required file is missing
        """
        file_map = {
            'main_data': cls.MAIN_EEG_DATA_FILE,
            'electrode_mappings': cls.ELECTRODE_MAPPINGS_FILE,
            'electrode_placement': cls.ELECTRODE_PLACEMENT_FILE
        }
        
        if file_type not in file_map:
            raise ValueError(f"Unknown file type: {file_type}")
        
        file_path = cls.get_data_file_path(file_map[file_type])
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required {file_type} file not found: {file_path}\n"
                f"Please ensure the file exists before running analysis."
            )
        
        return True
    
    @classmethod
    def get_rat_9442_electrode_info(cls, session_date):
        """
        Get electrode information for rat 9442 based on session date.
        
        Parameters:
        -----------
        session_date : str
            Session date string
            
        Returns:
        --------
        dict
            Dictionary with electrode information
        """
        if session_date in cls.RAT_9442_32_CHANNEL_SESSIONS:
            return {
                'n_channels': 32,
                'session_type': '32_channel',
                'mapping_rat': '9151'  # Use rat 9151's mapping
            }
        else:
            return {
                'n_channels': 20,
                'session_type': '20_channel',
                'mapping_rat': '9442',
                'available_electrodes': cls.RAT_9442_20_CHANNEL_ELECTRODES
            }
    
    @classmethod
    def get_file_summary(cls):
        """
        Get summary of all configured file paths.
        
        Returns:
        --------
        dict
            Summary of file paths and their existence status
        """
        files_to_check = {
            'Main EEG Data': cls.MAIN_EEG_DATA_FILE,
            'Electrode Mappings': cls.ELECTRODE_MAPPINGS_FILE,
            'Electrode Placement': cls.ELECTRODE_PLACEMENT_FILE,
            'ROI Mapping': cls.ROI_MAPPING_FILE,
            'Rat IDs': cls.RAT_IDS_FILE,
            'Dataset Info': cls.DATASET_INFO_FILE
        }
        
        summary = {}
        for name, path in files_to_check.items():
            abs_path = cls.get_data_file_path(path)
            summary[name] = {
                'path': path,
                'absolute_path': abs_path,
                'exists': os.path.exists(abs_path)
            }
        
        return summary