import sys
import argparse
import tomlkit

from fdm_validation_utility import (
    FDM_Problem_Config,
    FDM_Advection_State,
    FDM_Error,
    init_by_name,
    convergence_test,
)


def generate_init(args):
    config = FDM_Problem_Config(args.vel, args.sigma)
    init_cond = init_by_name[args.init](config, args.ndx, args.time)
    init_cond.to_h5(args.output)


def calculate_error(args):
    computed = FDM_Advection_State.from_h5(args.state)
    config = FDM_Problem_Config(args.vel, args.sigma)
    true = init_by_name[args.init](config, computed.ndx, time=computed.time)
    err = FDM_Error(computed, true)
    print(err.pprint_string())


def convergence(args):
    with open(args.base_toml) as f:
        base_toml = tomlkit.parse(f.read())
    convergence_test(
        args.executable, args.init, base_toml, args.kstart, args.kend,
    )


def diff_output(args):
    state1 = FDM_Advection_State.from_h5(args.file1)
    state2 = FDM_Advection_State.from_h5(args.file2)
    diff = FDM_Error(state1, state2)
    print(diff.pprint_string())


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Utility to aid in validating FCT implementations"
    )

    subparsers = parser.add_subparsers()

    # Init parser

    init_create_parser = subparsers.add_parser(
        "create_init", description="Create test problem initial state"
    )

    init_create_parser.add_argument(
        "--vel", type=float, default=3.0, help="Advection velocity"
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
        "--time", type=float, default=0.0, help="Initialization time",
    )

    init_create_parser.add_argument(
        "--output", "-o", default="initial_condition.h5", help="Output File",
    )

    init_create_parser.set_defaults(utilname="INIT")

    error_parser = subparsers.add_parser(
        "error", description="Calculate norm error for solution"
    )

    error_parser.add_argument(
        "--vel", type=float, default=3.0, help="Advection velocity"
    )

    error_parser.add_argument(
        "--sigma", type=float, default=0.9, help="CFL number"
    )

    error_parser.add_argument(
        "--init",
        choices=init_by_name.keys(),
        required=True,
        help="Name of initial state to generate",
    )

    error_parser.add_argument(
        "--state",
        required=True,
        help="Filename of HDF5 end state for validation",
    )

    error_parser.set_defaults(utilname="ERROR")

    diff_parser = subparsers.add_parser(
        "diff", description="Compare two output files"
    )

    diff_parser.add_argument("file1", help="File 1")

    diff_parser.add_argument("file2", help="File 2")

    diff_parser.set_defaults(utilname="DIFF")

    convergence_parser = subparsers.add_parser(
        "convergence", description="Utility to run a standard convergence test"
    )

    convergence_parser.add_argument("executable", help="Executable to test")

    convergence_parser.add_argument(
        "base_toml", help="Base configuration file"
    )

    convergence_parser.add_argument(
        "--init",
        choices=init_by_name.keys(),
        required=True,
        help="Name of initial state to generate",
    )

    convergence_parser.add_argument("--kstart", type=int, default=3)

    convergence_parser.add_argument("--kend", type=int, default=7)

    convergence_parser.set_defaults(utilname="CONVERGENCE")

    return parser.parse_args(args)


def main():
    parsed_args = parse_args(sys.argv[1:])
    if parsed_args.utilname == "INIT":
        generate_init(parsed_args)
    elif parsed_args.utilname == "ERROR":
        calculate_error(parsed_args)
    elif parsed_args.utilname == "CONVERGENCE":
        convergence(parsed_args)
    elif parsed_args.utilname == "DIFF":
        diff_output(parsed_args)
