"""Visualization modules for RockWatch."""
from .eda_plots import (
    plot_class_distribution,
    plot_feature_distributions,
    plot_correlation_matrix,
    plot_lulc_distribution,
    plot_elevation_slope_profile,
)
from .spatial_plots import plot_spatial_splits

__all__ = [
    "plot_class_distribution",
    "plot_feature_distributions",
    "plot_correlation_matrix",
    "plot_lulc_distribution",
    "plot_elevation_slope_profile",
    "plot_spatial_splits",
]
