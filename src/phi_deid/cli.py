"""Command-line interface.

    phi-deid deidentify path/to/note.txt      # redact a file (or stdin)
    phi-deid eval --n 100                      # benchmark on synthetic notes
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

from .evaluate import evaluate, format_report
from .pseudonymize import pseudonymize, reidentify
from .pipeline import deidentify, detect_entities
from .synthetic import generate_notes

# Salt for the hashed-key output. Override per deployment; secret in production.
_DEFAULT_SALT = "phi-deid-default-salt-change-me"


def _configure_logging(debug: bool) -> None:
    """Wire up logging. ``--debug`` raises only our package to DEBUG, leaving
    Presidio's very chatty internal loggers at WARNING so the step logs stay
    readable."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if debug:
        logging.getLogger("phi_deid").setLevel(logging.DEBUG)


def _cmd_deidentify(args: argparse.Namespace) -> int:
    if args.path:
        with open(args.path, "r", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()
    redacted, entities = deidentify(text)
    print(f"============ REDACTED OUTPUT =============\n")
    sys.stdout.write(redacted)
    if not redacted.endswith("\n"):
        sys.stdout.write("\n")
    print(f"\n[{len(entities)} PHI spans redacted]", file=sys.stderr)

    print("\n============ DETECTED ENTITIES ===========\n")
    for e in sorted(entities, key=lambda e: e.start):
        print(
            f"  {e.entity_type:<24} [{e.start}:{e.end}]  score={e.score:.4f}  "
            f"{text[e.start:e.end]!r}"
        )

    salt = os.environ.get("PHI_DEID_SALT", _DEFAULT_SALT)
    pseudonymized_text, vault = pseudonymize(text, entities, salt)
    print("\n========== PSEUDONYMIZED OUTPUT ==========\n")
    sys.stdout.write(pseudonymized_text)
    if not pseudonymized_text.endswith("\n"):
        sys.stdout.write("\n")

    reidentified = reidentify(pseudonymized_text, vault)
    print("\n========== RE-IDENTIFIED OUTPUT ==========\n")
    sys.stdout.write(reidentified)
    if not reidentified.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    notes = generate_notes(n=args.n, seed=args.seed)
    triples = []
    for note in notes:
        gold = [(s.start, s.end, s.entity_type) for s in note.spans]
        pred = [(e.start, e.end, e.entity_type) for e in detect_entities(note.text)]
        triples.append((note.text, gold, pred))
    report = evaluate(triples)
    print(format_report(report))

    if args.show:
        sample = notes[0]
        redacted, _ = deidentify(sample.text)
        print("\n--- sample note (original) ---\n" + sample.text)
        print("\n--- sample note (redacted) ---\n" + redacted)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phi-deid", description=__doc__)

    # Shared options available on every subcommand (e.g. `phi-deid eval --debug`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-d", "--debug", action="store_true", help="enable DEBUG pipeline logging"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_de = sub.add_parser(
        "deidentify", parents=[common], help="redact PHI from a file or stdin"
    )
    p_de.add_argument("path", nargs="?", help="input text file (default: stdin)")
    p_de.set_defaults(func=_cmd_deidentify)

    p_ev = sub.add_parser(
        "eval", parents=[common], help="benchmark on synthetic labeled notes"
    )
    p_ev.add_argument("--n", type=int, default=50, help="number of notes")
    p_ev.add_argument("--seed", type=int, default=7, help="random seed")
    p_ev.add_argument("--show", action="store_true", help="print a redacted sample")
    p_ev.set_defaults(func=_cmd_eval)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.debug)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
