# Synthetic kinetics datasets

This directory hosts curated CSV files that exercise various kinetic motifs for fitting demos and regression tests. All files use comma-separated values, seconds for time (`time`), and molar concentration units for species columns named directly after each species (e.g., `A`, `PBMP`). Gaussian noise (sigma noted below) keeps the data realistic but still well behaved.

## Files

### `first_order_decay_single.csv`
- Scenario: irreversible A → B decay sampled over 60 s.
- Columns: `time`, `A`, `B`.
- Parameters: k = 0.08 s⁻¹, A₀ = 1.0; noise σ = 0.01 concentration units.
- Use case: introductory single-dataset fitting or simulator sanity checks.

### `first_order_decay_global/`
Contains three CSVs (`dataset_01`–`dataset_03`) for global fitting of the same first-order mechanism with different initial concentrations (0.5, 1.0, 1.5). Shared parameters and layout:
- Columns: `time`, `A`, `B`.
- Parameters: k = 0.08 s⁻¹, identical time grid (0–40 s, 61 points).
- Noise: σ = 1% of the initial concentration for each species.
- Use case: validating multi-dataset/global workflows that share rate constants while varying only initials.

### `consecutive_A_B_C.csv`
- Scenario: A → B → C with k₁ = 0.12 s⁻¹, k₂ = 0.03 s⁻¹, sampled for 120 s.
- Columns: `time`, `A`, `B`, `C`.
- Noise: σ = 0.01 for A and 0.008 for B/C.
- Use case: testing fitting routines that must capture transient intermediates.

### `parallel_A_to_B_C.csv`
- Scenario: competitive A → B (k₁ = 0.05 s⁻¹) and A → C (k₂ = 0.08 s⁻¹) from A₀ = 1.2.
- Columns: `time`, `A`, `B`, `C`.
- Noise: σ = 0.01 for each species.
- Use case: exploring branching-ratio effects and datasets where multiple products grow simultaneously.

### `complex_mechanism_global/`
Global-fitting bundle (datasets `01`–`06`) that exercises a multi-step reversible/irreversible mechanism:

1. A + B ⇌ Int + Water (k_f = 1, k_r = 0.01)
2. Int + C → PBMPBPIN (k = 0.01)
3. PBMPBPIN + Water → PBMP + pinBOH (k = 100)

Each CSV shares the same time grid (0–120 s, 0.5 s spacing) and columns:
`time`, `A`, `B`, `C`, `Int`, `Water`, `PBMPBPIN`, `PBMP`, `pinBOH`.

- Dataset 01: A₀ = B₀ = C₀ = 0.2 M (benchmark condition requested by the user).
- Datasets 02–06: Additional A/B/C combinations ranging from 0.05–0.35 M to stress-test shared-parameter fits.
- Initial Water is fixed at 0.05 M for all files; other species start at 0.
- Noise: σ = 0.002 for reactants (A/B/C) and σ = 0.0015 for intermediates/products.
- Use case: demanding regression/global-fitting workflows that require simultaneous handling of reversible steps, intermediates, and fast consumption reactions.

All datasets maintain strictly increasing time axes, non-negative concentrations, and are ready to load via `kindred.gui.widgets.data_manager.load_csv_dataset()`.
