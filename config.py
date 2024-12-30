import numpy as np

# Genetic Algorithm hyperparameters
pop_size = 1000
parents_ratio = 0.01
generations = 100
seq_length = 100
mutation_rate = 0.2

# Logging and file-saving settings
save_log_path = "logs"

# Crossover & Mutation types
# e.g., "single_point", "two_point", "uniform", "block_based"
crossover_type = "block_based"
# e.g., "point", "swap", "inversion", "shuffle", "block", "segment_rotation", "segment_conjugation"
mutation_type = "segment_rotation"

# Additional parameters (e.g., for block-based methods)
block_size = 5

# Target unitary (e.g. single-qubit Hadamard gate)
target_unitary = (1 / np.sqrt(2)) * np.array([
    [1, 1],
    [1, -1]
])
