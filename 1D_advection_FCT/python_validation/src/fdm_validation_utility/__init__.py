"""Utilities to validate a FDM implementation based on providing
and checking the error using HDF5 files generated"""

__version__ = "0.0.1"

from fdm_validation_utility.problem_state import (
    FDM_Problem_Config,
    FDM_Advection_State,
)

from fdm_validation_utility.initial_conditions import init_by_name

from fdm_validation_utility.calculate_error import FDM_Error


__all__ = [
    "FDM_Problem_Config",
    "FDM_Advection_State",
    "FDM_Error",
    "init_by_name",
]
