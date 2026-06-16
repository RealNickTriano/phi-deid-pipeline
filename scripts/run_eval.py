#!/usr/bin/env python
"""Generate synthetic notes, run the pipeline, print the benchmark report.

    python scripts/run_eval.py --n 100 --show
"""

import argparse

from phi_deid.cli import _cmd_eval, _configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable DEBUG pipeline logging"
    )
    args = parser.parse_args()
    _configure_logging(args.debug)
    return _cmd_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
