"""Feature engineering and selection modules for RockWatch."""
from .engineer import add_water_body_flag, compute_aspect_trig

__all__ = [
    "add_water_body_flag",
    "compute_aspect_trig",
]
