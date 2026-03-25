# Benchmark: Moderately stiff pre-equilibrium with product formation
# Fast reversible pre-equilibrium followed by slower irreversible product.
A <-> B ; kf = 30.0 ; kr = 6.0
reaction: B -> C; k=0.12

# Initial concentrations
[A] = 1.0
[B] = 0.0
[C] = 0.0
