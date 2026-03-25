# Benchmark: Global consecutive decay (A -> B -> C)
# Two-step irreversible chain used for shared-parameter global fits.
reaction: A -> B; k=0.14
reaction: B -> C; k=0.05

# Baseline initial concentrations (datasets vary A0)
[A] = 1.0
[B] = 0.0
[C] = 0.0
