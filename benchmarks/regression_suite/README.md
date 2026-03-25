# Kindred Regression Benchmark Suite

Compact set of mechanisms and paired CSV datasets used for regression/performance smoke checks. All CSVs use a `time` column (seconds) followed by species columns in the order produced during simulation. Concentrations are in molar units and are strictly non-negative.

- `nonstiff_first_order.dsl` + `nonstiff_first_order.csv` — Non-stiff A → B decay with a single mild exponential timescale; 17 points over 0–8 s.
- `moderately_stiff_pre_equil.dsl` + `moderately_stiff_pre_equil.csv` — Fast A ⇌ B pre-equilibrium feeding a slower B → C product build; 51 points over 0–25 s showing rapid equilibration then a slower tail.
- `very_stiff_rober.dsl` + `very_stiff_rober.csv` — ROBER-like triad (A → B, B + C → A, 2B → C) capturing the sharp initial transient and slow depletion; 16 log-spaced points from 10⁻⁶ to 10³ s.
- `global_consecutive.dsl` + `global_consecutive_dataset_01..03.csv` — Two-step irreversible chain A → B → C with shared rates and differing A₀ values (0.4, 0.8, 1.2). Common 61-point grid over 0–30 s for global-fit workflows.

Datasets were generated via `parse_dsl_to_mechanism` + `solve_ode` (LSODA) with small Gaussian noise clipped at zero for physical consistency (seeds: 7, 11, none for ROBER, 1/2/3 for the global trio).
