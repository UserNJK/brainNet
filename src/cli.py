"""Console entry point for the bioneuron library (`bioneuron ...`)."""

from __future__ import annotations
import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="bioneuron",
        description="Biologically accurate spiking neural network — CLI.")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("version", help="print the installed version")

    d = sub.add_parser("demo", help="run a closed-loop learning demo")
    d.add_argument("which", nargs="?", default="conditioning",
                   choices=["conditioning", "discrimination"])
    d.add_argument("--trials", type=int, default=150)
    d.add_argument("--neurons", type=int, default=None, help="network size N")

    args = parser.parse_args(argv)

    try:
        from . import __version__
    except ImportError:                       # running flat
        __version__ = "dev"

    if args.cmd in (None, "version"):
        print(f"bioneuron {__version__}")
        return 0

    if args.cmd == "demo":
        try:
            from .closed_loop import demo_conditioning, demo_discrimination
        except ImportError:
            from closed_loop import demo_conditioning, demo_discrimination
        if args.which == "discrimination":
            kw = {"n_trials": args.trials}
            if args.neurons:
                kw["N"] = args.neurons
            early, late, _ = demo_discrimination(**kw)
            print(f"\n2-way discrimination accuracy: {early:.2f} -> {late:.2f} (chance 0.50)")
        else:
            kw = {"n_trials": args.trials}
            if args.neurons:
                kw["N"] = args.neurons
            early, late, _ = demo_conditioning(**kw)
            print(f"\noperant conditioning response rate: {early:.2f} -> {late:.2f}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
