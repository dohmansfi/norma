"""``python -m norma``: delegate to the CLI; no parser lives here.

``norma.cli`` arrives at stage S1. Until then this entry point fails with a
clear ImportError rather than pretending to run.
"""

if __name__ == "__main__":
    import sys

    from norma.cli import main

    sys.exit(main())
