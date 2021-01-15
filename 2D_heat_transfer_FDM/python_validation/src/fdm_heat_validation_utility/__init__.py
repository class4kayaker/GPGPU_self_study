"""Utilities to validate a FDM implementation based on providing
and checking the error using HDF5 files generated"""

__version__ = "0.0.1"

from .initial_conditions import init_by_name
from .model_utils import FDM_State

__all__ = ["init_by_name", "FDM_State"]
