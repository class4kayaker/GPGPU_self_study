import timeit
import numpy
import subprocess
import tempfile
import tomlkit
from pathlib import Path

from fdm_validation_utility import (
    FDM_Advection_State,
    FDM_Error,
    init_by_name,
)

from .toml_utils import problem_config_from_toml


def calculate_run_error(executable, initname, config_base, ndx):
    config = problem_config_from_toml(config_base)
    config_dat = config_base.copy()
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirpath = Path(tmpdirname)
        initfilepath = tmpdirpath / "init.h5"
        endfilepath = tmpdirpath / "end.h5"
        config_path = tmpdirpath / "config.toml"

        init_state = init_by_name[initname](config, ndx, time=0.0)
        init_state.to_h5(initfilepath)

        config_dat["Init file"] = str(initfilepath)
        config_dat["Output file"] = str(endfilepath)

        tic = timeit.default_timer()
        with config_path.open("w") as f:
            f.write(tomlkit.dumps(config_dat))
        subprocess.check_call([executable, str(config_path)])
        toc = timeit.default_timer()

        run_time = toc - tic
        end_true = init_by_name[initname](config, ndx, time=config.end_time)
        end_computed = FDM_Advection_State.from_h5(endfilepath)
        return FDM_Error(end_computed, end_true), run_time


convergence_data_dtype = [
    ("ndx", numpy.int64),
    ("dx", numpy.float64),
    ("L1_err", numpy.float64),
    ("L1_rate", numpy.float64),
    ("L2_err", numpy.float64),
    ("L2_rate", numpy.float64),
    ("LI_err", numpy.float64),
    ("LI_rate", numpy.float64),
    ("time", numpy.float64),
    ("t_rate", numpy.float64),
]


def convergence_runs(executable, initname, config_base, kstart, kend):
    err_data = numpy.ndarray((kend - kstart,), dtype=convergence_data_dtype)
    for norm in ["L1", "L2", "LI"]:
        err_data[norm + "_rate"] = numpy.nan
    err_data["t_rate"] = numpy.nan
    for i, k in enumerate(range(kstart, kend)):
        ndx = 2 ** k
        err_data["ndx"][i] = ndx
        error, runtime = calculate_run_error(
            executable, initname, config_base, ndx
        )
        err_data["dx"][i] = error.computed.dx
        err_data["L1_err"][i] = error.L1_err()
        err_data["L2_err"][i] = error.L2_err()
        err_data["LI_err"][i] = error.Linf_err()
        err_data["time"][i] = runtime
        del error
    for norm in ["L1", "L2", "LI"]:
        err_data[norm + "_rate"][1:] = numpy.log2(
            err_data[norm + "_err"][:-1] / err_data[norm + "_err"][1:]
        ) / numpy.log2(err_data["dx"][:-1] / err_data["dx"][1:])
    err_data["t_rate"][1:] = numpy.log2(
        err_data["time"][1:] / err_data["time"][:-1]
    ) / numpy.log2(err_data["dx"][:-1] / err_data["dx"][1:])
    return err_data


def pprint_convergence(data):
    colmap = {k: k for k, _ in convergence_data_dtype}
    fmt_head = (
        "|"
        + "|".join(
            [
                "{r[ndx]:12s}",
                "{r[dx]:12s}",
                "{r[L1_err]:12s}",
                "{r[L1_rate]:7s}",
                "{r[L2_err]:12s}",
                "{r[L2_rate]:7s}",
                "{r[LI_err]:12s}",
                "{r[LI_rate]:7s}",
                "{r[time]:12s}",
                "{r[t_rate]:7s}",
            ]
        )
        + "|"
    )
    fmt_row = (
        "|"
        + "|".join(
            [
                "{r[ndx]:12d}",
                "{r[dx]:12.6e}",
                "{r[L1_err]:>12.6e}",
                "{r[L1_rate]:7.4f}",
                "{r[L2_err]:>12.6e}",
                "{r[L2_rate]:7.4f}",
                "{r[LI_err]:>12.6e}",
                "{r[LI_rate]:7.4f}",
                "{r[time]:>12.6e}",
                "{r[t_rate]:7.4f}",
            ]
        )
        + "|"
    )

    print(fmt_head.format(r=colmap))
    for r in data:
        print(fmt_row.format(r=r))


def convergence_test(executable, initname, config_base, kstart, kend):
    data = convergence_runs(executable, initname, config_base, kstart, kend)
    pprint_convergence(data)
