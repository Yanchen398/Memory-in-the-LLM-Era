"""Allow ``python -m Method_memoryarena`` to use the unified CLI."""

from .run import main


if __name__ == "__main__":
    raise SystemExit(main())
