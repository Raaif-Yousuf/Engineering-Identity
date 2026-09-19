"""Entry point for `python -m ei_model.experiments <command> ...`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
