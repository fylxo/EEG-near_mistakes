"""
Plotting Configuration Settings

This module contains all visualization and plotting parameters used throughout
the EEG analysis pipeline.
"""


class PlottingConfig:
    """
    Plotting and visualization parameters for EEG analysis.
    
    This class contains all settings related to plots, figures, colors, and visualization.
    """
    
    # =============================================================================
    # FIGURE SETTINGS
    # =============================================================================
    
    # Default figure sizes (width, height in inches)
    FIGURE_SIZE_DEFAULT = (10, 6)
    FIGURE_SIZE_LARGE = (12, 8)
    FIGURE_SIZE_SPECTROGRAM = (10, 5)
    FIGURE_SIZE_COMPARISON = (18, 12)
    
    # DPI settings
    DPI_DEFAULT = 300
    DPI_HIGH_QUALITY = 600
    
    # Figure format for saving
    FIGURE_FORMAT = 'png'
    
    # =============================================================================
    # THETA POWER PLOT SETTINGS
    # =============================================================================
    
    # Y-axis limits for theta power plots
    THETA_POWER_YLIM_MIN = -0.3
    THETA_POWER_YLIM_MAX = 0.3
    
    # Bar plot settings
    BAR_ALPHA = 0.7
    BAR_EDGE_COLOR = 'black'
    BAR_COLOR_DEFAULT = 'skyblue'
    BAR_CAPSIZE = 5  # Error bar cap size
    
    # Value label settings
    VALUE_LABEL_FONTSIZE = 10
    VALUE_LABEL_OFFSET = 0.01
    
    # =============================================================================
    # SPECTROGRAM SETTINGS
    # =============================================================================
    
    # Colormap settings
    COLORMAP_DEFAULT = 'viridis'  # Will be overridden by Parula
    COLORMAP_USE_PARULA = True
    
    # Color limits calculation
    COLOR_LIMIT_PERCENTILE = 95.0
    COLOR_LIMIT_SYMMETRIC = True
    
    # Frequency axis settings
    FREQ_AXIS_LOG_SCALE = True
    FREQ_AXIS_N_TICKS = 7  # Total number of frequency ticks to show
    
    # Time axis settings
    TIME_AXIS_SHOW_ZERO_LINE = True
    TIME_AXIS_ZERO_LINE_COLOR = 'black'
    TIME_AXIS_ZERO_LINE_STYLE = '--'
    TIME_AXIS_ZERO_LINE_ALPHA = 0.7
    
    # Colorbar settings
    COLORBAR_LABEL = 'Z-score'
    COLORBAR_SHRINK = 1.0
    
    # =============================================================================
    # LAYOUT AND SPACING
    # =============================================================================
    
    # Subplot spacing parameters
    SUBPLOT_LEFT = 0.052
    SUBPLOT_BOTTOM = 0.07
    SUBPLOT_RIGHT = 0.55
    SUBPLOT_TOP = 0.924
    SUBPLOT_WSPACE = 0.206
    SUBPLOT_HSPACE = 0.656
    
    # Title and label settings
    TITLE_FONTSIZE = 14
    AXIS_LABEL_FONTSIZE = 12
    TICK_LABEL_FONTSIZE = 10
    LEGEND_FONTSIZE = 10
    
    # Grid settings
    GRID_ENABLE = True
    GRID_ALPHA = 0.3
    
    # =============================================================================
    # COLOR SCHEMES
    # =============================================================================
    
    # Colors for different NM types
    NM_COLORS = {
        1: '#1f77b4',  # Blue
        2: '#ff7f0e',  # Orange  
        3: '#2ca02c',  # Green
        4: '#d62728',  # Red
        5: '#9467bd'   # Purple
    }
    
    # Colors for different analysis conditions
    CONDITION_COLORS = {
        'baseline': '#cccccc',
        'pre_event': '#ff9999',
        'post_event': '#99ff99',
        'theta_peak': '#ffff99'
    }
    
    # =============================================================================
    # TEXT AND ANNOTATION SETTINGS
    # =============================================================================
    
    # Info box settings for plots
    INFO_BOX_FONTSIZE = 10
    INFO_BOX_POSITION = (0.02, 0.98)  # (x, y) in axes coordinates
    INFO_BOX_VERTICAL_ALIGNMENT = 'top'
    INFO_BOX_STYLE = {
        'boxstyle': 'round',
        'facecolor': 'white',
        'alpha': 0.8
    }
    
    # Statistical annotation settings
    STATS_ANNOTATION_FONTSIZE = 9
    SIGNIFICANCE_MARKERS = {
        'p < 0.001': '***',
        'p < 0.01': '**', 
        'p < 0.05': '*',
        'p >= 0.05': 'ns'
    }
    
    # =============================================================================
    # INDIVIDUAL RAT PLOT SETTINGS
    # =============================================================================
    
    # Number of individual rats to show in comparison plots
    MAX_INDIVIDUAL_RATS_DISPLAY = 6
    
    # Individual rat plot layout
    INDIVIDUAL_RATS_SUBPLOT_ROWS = 2
    INDIVIDUAL_RATS_SUBPLOT_COLS = 3
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    @classmethod
    def get_theta_power_ylim(cls):
        """Get theta power plot y-axis limits as tuple."""
        return (cls.THETA_POWER_YLIM_MIN, cls.THETA_POWER_YLIM_MAX)
    
    @classmethod
    def get_figure_size(cls, plot_type='default'):
        """
        Get figure size for different plot types.
        
        Parameters:
        -----------
        plot_type : str
            Type of plot ('default', 'large', 'spectrogram', 'comparison')
            
        Returns:
        --------
        tuple
            Figure size (width, height) in inches
        """
        size_map = {
            'default': cls.FIGURE_SIZE_DEFAULT,
            'large': cls.FIGURE_SIZE_LARGE,
            'spectrogram': cls.FIGURE_SIZE_SPECTROGRAM,
            'comparison': cls.FIGURE_SIZE_COMPARISON
        }
        return size_map.get(plot_type, cls.FIGURE_SIZE_DEFAULT)
    
    @classmethod
    def get_nm_color(cls, nm_type):
        """
        Get color for specific NM type.
        
        Parameters:
        -----------
        nm_type : int or float
            Near-mistake type
            
        Returns:
        --------
        str
            Color hex code
        """
        nm_int = int(float(nm_type))
        return cls.NM_COLORS.get(nm_int, '#000000')  # Default to black
    
    @classmethod
    def setup_matplotlib_defaults(cls):
        """
        Set up matplotlib with default parameters for consistent plotting.
        """
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            'figure.dpi': cls.DPI_DEFAULT,
            'savefig.dpi': cls.DPI_DEFAULT,
            'figure.figsize': cls.FIGURE_SIZE_DEFAULT,
            'font.size': cls.TICK_LABEL_FONTSIZE,
            'axes.titlesize': cls.TITLE_FONTSIZE,
            'axes.labelsize': cls.AXIS_LABEL_FONTSIZE,
            'xtick.labelsize': cls.TICK_LABEL_FONTSIZE,
            'ytick.labelsize': cls.TICK_LABEL_FONTSIZE,
            'legend.fontsize': cls.LEGEND_FONTSIZE,
            'grid.alpha': cls.GRID_ALPHA,
            'savefig.format': cls.FIGURE_FORMAT,
            'savefig.bbox': 'tight'
        })
    
    @classmethod
    def get_subplot_params(cls):
        """
        Get subplot adjustment parameters.
        
        Returns:
        --------
        dict
            Parameters for plt.subplots_adjust()
        """
        return {
            'left': cls.SUBPLOT_LEFT,
            'bottom': cls.SUBPLOT_BOTTOM,
            'right': cls.SUBPLOT_RIGHT,
            'top': cls.SUBPLOT_TOP,
            'wspace': cls.SUBPLOT_WSPACE,
            'hspace': cls.SUBPLOT_HSPACE
        }
    
    @classmethod
    def get_info_box_text(cls, roi, freq_range, time_window=None, n_rats=None):
        """
        Generate standardized info box text for plots.
        
        Parameters:
        -----------
        roi : str
            ROI specification
        freq_range : tuple
            Frequency range (min, max)
        time_window : tuple, optional
            Time window (start, end)
        n_rats : int, optional
            Number of rats
            
        Returns:
        --------
        str
            Formatted info box text
        """
        lines = []
        lines.append(f"ROI: {roi}")
        lines.append(f"Freq: {freq_range[0]}-{freq_range[1]} Hz")
        
        if time_window:
            lines.append(f"Time: {time_window[0]}-{time_window[1]} s")
        
        if n_rats:
            lines.append(f"Rats: {n_rats}")
        
        return '\n'.join(lines)