"""Small Fit-local UI constants; simulation science comes from each mapped set."""

INITIAL_PREFIX = "init:"
FITTING_METHODS = ("trf", "dogbox", "differential_evolution")
DIFFERENTIAL_EVOLUTION_METHOD = "differential_evolution"
LEAST_SQUARES_METHODS = ("trf", "dogbox")
FUNCTION_TOLERANCE_LABEL = "Function tolerance"
DE_POPULATION_TOLERANCE_LABEL = "Population-convergence tolerance"
FITTING_MAX_NFEV_RANGE = (10, 10_000)
FITTING_SEED_RANGE = (0, 999_999)
SIMULATION_SOLVERS = ("Radau", "BDF")
