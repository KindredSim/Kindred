# Benchmark: Very stiff ROBER-like triad
# Adapted classic stiff system with two fast reactions and one slow decay.
reaction: A -> B; k=0.04
reaction: B + C -> A; k=10000.0
reaction: 2*B -> C; k=30000000.0

# Initial concentrations
[A] = 1.0
[B] = 0.0
[C] = 0.0
