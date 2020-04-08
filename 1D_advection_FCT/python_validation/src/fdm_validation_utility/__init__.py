"""Utilities to validate a FDM implementation based on providing
and checking the error using HDF5 files generated"""

__version__ = "0.0.1"

from .problem_state import (
    FDM_Problem_Config,
    FDM_Advection_State,
)

from .initial_conditions import init_by_name

from .calculate_error import FDM_Error

from .convergence import convergence_test


__all__ = [
    "FDM_Problem_Config",
    "FDM_Advection_State",
    "FDM_Error",
    "init_by_name",
    "convergence_test",
]
