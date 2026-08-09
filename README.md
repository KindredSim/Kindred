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

## Current architecture map

Kindred keeps semantic state in a small set of accepted owners. `ProjectSession`
admits complete `ProjectDocument` state, `WorkspaceTabRegistry` owns tab
identity/selection, and `ProjectController` owns Save/disposition and the
application Close lifecycle. `SimulationJobOwner` owns direct Run/Preview
generations, workers, cancellation, and publication; `ResultRepository` owns
exact result identity and compatible numerical cache entries. `FitSessionOwner`
owns modal Fit setup, Local initial-condition state plus an explicitly active
Global configuration, and scalar-scope state, Apply/Reset, and terminal result
admission, while `GlobalFitWorker` retains QThread custody. `DatasetImportStaging`
owns a raw import batch until `ProjectSession` admits it; `ProjectDocument`
retains each imported source once and datasets carry source references. The
strict archive stores one source member per source, rejects unknown members
structurally, and has no byte, member-count, batch, or other defensive limit as
product policy. Import publication has one terminal success/failure outcome
before staging is ready for the next import. `kindred.core.datasets` owns
immutable dataset values and shared source rules. `MainWindow` composes these
owners and wires presentation without duplicating their semantic state.

Run and Preview use the typed `SimulationExecutionRequest` boundary and the
canonical `prepare_simulation_worker_run` preparation path. Run publishes
canonical results through `SimulationJobOwner` and `ResultRepository`; Preview
uses the same execution contract but remains transient and latest-only. Fit
starts from an accepted `FitSessionOwner` setup, gives each candidate a fresh
canonical preparation through the same boundary, and applies terminal results
only through the owner’s current-result checks.

Scientific policy stays below the GUI: `dsl_parse` owns the line parser and
the sole syntax-admission boundary, including `prepare_dsl_request_set`.
`dsl_derivation` owns only the narrow source-coordinate rewrite helpers used
when an admitted typed request is derived for Fit or another explicitly
admitted consumer. The preparation boundary owns validated solver/ODE
lowering. `param` is static solver-parameter algebra while `let` is dynamic
row-local algebra; chemistry rate/equilibrium and detailed-balance policy
remain canonical core authorities. Typing and ordinary presentation do not
perform scientific preparation.

`ProjectDocument` owns Fit method, solver, evaluation, and seed rosters and
ranges. `ftol` and `xtol` are submitted as exact positive finite values, with
no product-level upper bound of 1; core `FitProgress` owns improvement and the
best-simulation payload. Statistics and plot identity remain current-result,
projected-roster, and display-label aware.

Developer validation is:

    bash tools/audit/run_ci.sh

This runs the strict audit and then `pytest -q`. At present the repository has
no test modules matching pytest’s collection patterns, so `pytest -q` exits 5
with no tests collected; that status is reported explicitly rather than as a
passing test suite.

## Repository

GitHub: https://github.com/kindredsim/kindred

## License

Kindred source code is released under the MIT License. Third-party dependencies are distributed under their own licenses. In particular, PySide6 / Qt is licensed separately and may impose additional obligations for binary redistribution.
