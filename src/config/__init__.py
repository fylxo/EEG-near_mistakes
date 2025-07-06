"""
Configuration package for EEG Near-Mistake Analysis

This package provides centralized configuration management for all analysis parameters.
"""

from .analysis_config import AnalysisConfig
from .data_config import DataConfig
from .plotting_config import PlottingConfig

__all__ = ['AnalysisConfig', 'DataConfig', 'PlottingConfig']