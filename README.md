# Kindred

Kindred is a desktop application for building chemical kinetic mechanisms,
simulating their behavior, comparing results with experimental data, and
fitting kinetic parameters.

## Capabilities

- Build and inspect multi-step reaction mechanisms in a line-oriented DSL.
- Run kinetic simulations with configurable time ranges, solvers, tolerances,
  temperatures, initial conditions, temperature schedules, and interventions.
- Explore simulation results in interactive plots and analysis views.
- Import experimental datasets from CSV and XLSX files, map observations to
  model species or observables, and fit one or more datasets.
- Copy fit results, export current result data to XLSX, and save complete
  projects as `.kin` files.
- Start from bundled mechanism presets and intervention examples.

Mechanism steps may use bare reaction lines or the `reaction:` and
`equilibrium:` prefixes. Irreversible arrows may be written as `->` or `=>`,
and reversible arrows as `<->` or `<=>`. The bundled presets use the same
accepted DSL. Open the Reactions Help tab in Kindred for the syntax reference
and examples.

## Installation

Kindred requires Python 3.10, 3.11, or 3.12. From the repository root, create
and activate a virtual environment, then install the application:

    python -m venv .venv
    python -m pip install --upgrade pip
    python -m pip install .

The standard installation includes the required desktop and plotting
dependencies, including PySide6 and PyQtGraph. The same pinned runtime set is
listed in `requirements.txt` for environments that install dependencies from a
requirements file.

## Launch

Launch the installed application with either command:

    kindred

    python -m kindred

Kindred launches only the desktop interface and does not accept command-line
arguments.

## Basic workflow

1. Choose a bundled example or enter a mechanism in the Reactions editor.
2. Set initial concentrations and simulation settings, then select **Run**.
3. Inspect and configure the plot, parameters, statistics, and analysis views.
4. To compare with experiments, load CSV or XLSX data and map its time and
   observation columns to the mechanism.
5. Open **Fitting → Global Fit**, choose datasets and adjustable values, and run
   the fit.
6. Export current results to XLSX or save the project as a `.kin` file.

## Links

- [Source repository](https://github.com/kindredsim/kindred)
- [Issue tracker](https://github.com/kindredsim/kindred/issues)
- [Kindred license](LICENSE)
- [Third-party notices](NOTICE.txt)
- [Third-party licenses](THIRD_PARTY_LICENSES.txt)
