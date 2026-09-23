"""``python -m dolores.unified run --config <yaml>`` — the unified run entry point.

Importing ``cli`` (rather than running it as ``__main__``) keeps its ``if test():``
blocks from executing in every parent and child process.
"""
from dolores.unified.cli import main

if __name__ == "__main__":
    main()
