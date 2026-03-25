# Kindred

Kindred is a desktop GUI for chemical kinetics simulation and fitting.

## Status

Kindred supports Python 3.10, 3.11, and 3.12. The repository CI currently validates on Python 3.12.

## Installation

Create a Python 3.10-3.12 environment, then install the project:

    python -m pip install --upgrade pip
    python -m pip install .

The pinned GUI runtime dependency set used by this repository is listed in `requirements.txt`:

    python -m pip install -r requirements.txt

For development and tests:

    python -m pip install -e ".[test]"

## Launch

    python -m kindred

or, if installed as a script:

    kindred

## Repository

GitHub: https://github.com/kindredsim/kindred

## License

Kindred source code is released under the MIT License. Third-party dependencies are distributed under their own licenses. In particular, PySide6 / Qt is licensed separately and may impose additional obligations for binary redistribution.
