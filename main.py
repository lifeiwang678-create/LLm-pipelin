from __future__ import annotations

from core.cli import parse_args
from core.runner import run_from_args


def main() -> None:
    args = parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()

