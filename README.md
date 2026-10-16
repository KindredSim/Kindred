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
- Copy fit results, export current result data to XLSX, and save projects as
  `.kin` files. A project keeps its mechanisms, settings, initial conditions,
  datasets, plotted sets, and plot appearance. Run and fit results, display
  units, and zoom are not saved, so Run again after opening.
- Start from bundled mechanism presets and intervention examples.

Mechanism steps use unprefixed reaction lines. Irreversible arrows may be
written as `->` or `=>`,
and reversible arrows as `<->` or `<=>`. The bundled presets use the same
accepted DSL. Open the Reactions Help tab in Kindred for the syntax reference
and examples.

## Installation

Kindred is a Windows desktop application. Its GUI requires Python 3.12.
From the repository root, use native Windows Python 3.12 to create an
environment and install the application:

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -e .

Install in editable mode: the fitting runtime imports this same checkout, and a
regular install would give the GUI a separate copy of the code.

The standard installation includes the required desktop and plotting
dependencies, including PySide6 and PyQtGraph. To work on Kindred itself, also
install the development tools:

    .venv\Scripts\python.exe -m pip install -e . --group dev

### Fitting runtime

Global Fit runs in a separate native Windows Python 3.15t (free-threaded)
installation inside the GUI environment,
`.venv\kindred-fitting-runtime\python\python.exe`. Its scientific dependencies
are separate from the GUI dependencies, and the fitting entry point loads only
Qt-free Kindred modules. Kindred never downloads or installs this runtime.
Provision it once, and again whenever `[tool.kindred.fitting-runtime]` or the
`fitting-runtime` dependency group in `pyproject.toml` changes; the GUI's
Python constraint and dependencies remain under `[project]`.

1. Obtain the Windows 64-bit free-threaded Python archive for
   `python-version` (currently 3.15.0rc2t).
2. Unpack the complete installation so that `python.exe` sits directly in
   `.venv\kindred-fitting-runtime\python\`. Keep this depth; one level higher,
   CPython would read the GUI environment's `pyvenv.cfg`.
3. Install the pinned dependencies into the runtime itself:

       .venv\kindred-fitting-runtime\python\python.exe -E -s -m pip install --group fitting-runtime

   If pip is missing, run `-m ensurepip` with the same interpreter first.
4. Create `.venv\kindred-fitting-runtime\python\Lib\site-packages\kindred-source.pth`
   containing one line: the absolute path of the repository root (the folder
   containing `kindred\`). Do not add the GUI's site-packages.
5. Verify from the runtime folder, as Kindred itself starts the runtime:

       Push-Location .venv\kindred-fitting-runtime\python
       .\python.exe -E -s -m kindred.gui.fitting.process --verify-only; "exit code: $LASTEXITCODE"
       Pop-Location

   Exit code 0 means free-threaded 3.15 with the GIL still disabled after the
   scientific imports, no Qt modules loaded, and Kindred importable from this
   checkout. Failures print a traceback.

Kindred starts the runtime with `-E -s` (ignoring PYTHONPATH, PYTHONHOME,
PYTHON_GIL and the user site-packages) and limits BLAS/OpenMP to one thread.

## Launch

Launch the installed application with either command:

    .venv\Scripts\kindred.exe

    .venv\Scripts\python.exe -m kindred

Kindred launches only the desktop interface and does not accept command-line
arguments.

## Diagnostics

Kindred keeps detailed rotating logs in the platform log directory:

- Linux: `~/.cache/kindred/logs/kindred.log` (or
  `$XDG_CACHE_HOME/kindred/logs/kindred.log`)
- macOS: `~/Library/Logs/kindred/kindred.log`
- Windows: `%LOCALAPPDATA%\kindred\logs\kindred.log`

Set `KINDRED_LOG_DIR` to choose another normal log directory. If that location
cannot be opened, Kindred writes the log to the system temporary directory at
`kindred/logs/kindred.log`. `LOG_LEVEL` or `KINDRED_LOG_LEVEL` may be set to
`DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`.

Fitting checks for the required Python build and scientific dependencies.
Launch errors report missing or incompatible runtime
components and include available diagnostics from the fitting process.

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
