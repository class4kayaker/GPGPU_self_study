import sys
import argparse

from fdm_heat_validation_utility import init_by_name, FDM_State, FDM_Diff


def generate_init(args):
    init_cond = init_by_name[args.init](args.k)
    init_cond.to_h5(args.output)


def diff_output(args):
    state1 = FDM_State.from_h5(args.file1)
    state2 = FDM_State.from_h5(args.file2)
    diff = FDM_Diff(state1, state2)
    print(diff.pprint_string())
    if(args.diff_state):
        diff.diff_state().to_h5(args.diff_state)


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Utility to aid in validating FCT implementations"
    )

    parser.set_defaults(utilcall=None)

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
        "--k",
        type=int,
        required=True,
        help="Mesh size",
    )

    init_create_parser.add_argument(
        "--output",
        "-o",
        default="initial_condition.h5",
        help="Output File",
    )

    init_create_parser.set_defaults(utilcall=generate_init)

    diff_parser = subparsers.add_parser(
        "diff", description="Compare two output files"
    )

    diff_parser.add_argument("file1", help="File 1")

    diff_parser.add_argument("file2", help="File 2")

    diff_parser.add_argument(
        "--diff_state",
        "-o",
        default="",
        help="Output File",
    )

    diff_parser.set_defaults(utilcall=diff_output)

    return parser.parse_args(args)


def main():
    parsed_args = parse_args(sys.argv[1:])
    if parsed_args.utilcall is not None:
        parsed_args.utilcall(parsed_args)
