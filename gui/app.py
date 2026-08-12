from __future__ import annotations

try:
    from .server import main
except ImportError:
    from server import main


if __name__ == "__main__":
    main()
