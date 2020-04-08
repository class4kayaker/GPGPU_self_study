"""Utilities to validate a FDM implementation based on providing
and checking the error using HDF5 files generated"""

__version__ = "0.0.1"

import sys
import argparse
import numpy
import h5py


class FDM_Problem_Config:
    def __init__(self, ndx, a=3.0, sigma=0.9):
        self.ndx = ndx
        self.a = a
        self.sigma = sigma


def square_wave(x):
    return numpy.where((x >= 0.25) & (x <= 0.75), 1.0, 0.0)


def square_wave_exc(x):
    return numpy.where((x > 0.25) & (x < 0.75), 1.0, 0.0)


def semicircle(x):
    return numpy.sqrt(0.25 - (x - 0.5) ** 2)


def gaussian(x):
    return numpy.exp(-256.0 * (x - 0.5) ** 2)


def sinusoidal(x):
    return 0.5 * (1 - numpy.cos(2 * numpy.pi * x))


class FDM_Advection_State:
    def __init__(self, u, dx, time=0.0):
        self.ndx = u.shape[0]
        self.dx = dx
        self.time = time
        self.u = numpy.copy(u)

    @classmethod
    def from_h5(cls, h5fn):
        with h5py.File(h5fn, "r") as f:
            dx = f["dx"][()]
            time = f["time"][()]
            u = f["state"]
            return cls(u, dx, time)

    def to_h5(self, h5fn):
        with h5py.File(h5fn, "w") as f:
            f["dx"] = self.dx
            f["time"] = self.time
            f["state"] = self.u

    @classmethod
    def func_init(cls, config, func, time=0.0):
        ndx = config.ndx
        dx = 1.0 / ndx
        x = numpy.linspace(0.0, 1.0, num=ndx + 1)
        shift_x = 0.5 * (x[1:] + x[:-1]) - config.a * time
        n_shift_x = shift_x - numpy.floor(shift_x)
        u = func(n_shift_x)
        return cls(u, dx, time=time)


init_by_name = {
    "Sinusoidal": lambda config, time=0.0: FDM_Advection_State.func_init(
        config, sinusoidal, time
    ),
    "Semicircle": lambda config, time=0.0: FDM_Advection_State.func_init(
        config, semicircle, time
    ),
    "Gaussian": lambda config, time=0.0: FDM_Advection_State.func_init(
        config, gaussian, time
    ),
    "Square_Wave": lambda config, time=0.0: FDM_Advection_State.func_init(
        config, square_wave, time
    ),
}


class FDM_Error:
    def __init__(self, computed, true, eps=1e-10):
        self.computed = computed
        self.true = true
        assert abs(self.computed.dx - self.true.dx) < eps * abs(
            self.computed.dx
        )
        if abs(self.computed.time - self.true.time) >= eps:
            raise Exception(
                f"Time mismatch {self.computed.time:g}!={self.true.time:g}"
            )

    def L1_err(self):
        diff = self.computed.u - self.true.u
        return self.computed.dx * numpy.sum(numpy.abs(diff))

    def L2_err(self):
        diff = self.computed.u - self.true.u
        return numpy.sqrt(self.computed.dx * numpy.sum(diff * diff))

    def Linf_err(self):
        diff = self.computed.u - self.true.u
        return numpy.amax(numpy.abs(diff))

    def pprint_string(self):
        return (
            f"Errors: "
            f"L1 {self.L1_err():.6e} "
            f"L2 {self.L2_err():.6e} "
            f"Linf {self.Linf_err():.6e}"
        )


def generate_init(args):
    config = FDM_Problem_Config(args.ndx, args.vel, args.sigma)
    init_cond = init_by_name[args.init](config)
    init_cond.to_h5(args.output)


def validate(args):
    computed = FDM_Advection_State.from_h5(args.state)
    config = FDM_Problem_Config(computed.ndx, args.vel, args.sigma)
    true = init_by_name[args.init](config, time=computed.time)
    err = FDM_Error(computed, true)
    print(err.pprint_string())


def parse_args(args):
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--vel", type=float, default=3.0, help="Advection velocity"
    )

    parser.add_argument("--sigma", type=float, default=0.9, help="CFL number")

    subparsers = parser.add_subparsers()

    # Init parser

    init_create_parser = subparsers.add_parser(
        "create_init", description="Create test problem initial state"
    )

    init_create_parser.add_argument(
        "--init",
        choices=init_by_name.keys(),
        required=True,
        help="Name of initial state to generate",
    )

    init_create_parser.add_argument(
        "--ndx", type=int, required=True, help="Mesh size",
    )

    init_create_parser.add_argument(
        "--output", "-o", default="initial_condition.h5", help="Output File",
    )

    init_create_parser.set_defaults(utilname="INIT")

    validate_parser = subparsers.add_parser(
        "validate", description="Calculate norm error for solution"
    )

    validate_parser.add_argument(
        "--init",
        choices=init_by_name.keys(),
        required=True,
        help="Name of initial state to generate",
    )

    validate_parser.add_argument(
        "--state",
        required=True,
        help="Filename of HDF5 end state for validation",
    )

    validate_parser.set_defaults(utilname="VALIDATE")

    return parser.parse_args(args)


def main():
    parsed_args = parse_args(sys.argv[1:])
    if parsed_args.utilname == "INIT":
        generate_init(parsed_args)
    elif parsed_args.utilname == "VALIDATE":
        validate(parsed_args)
