from __future__ import annotations

from core.cli import parse_args
from core.runner import run_from_args


def main() -> None:
    args = parse_args()
    try:
        run_from_args(args)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == "__main__":
    main()
