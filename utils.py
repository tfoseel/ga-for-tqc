import random
import numpy as np

# -----------------------------------------------------------
# (A) Fitness function
# -----------------------------------------------------------


def fitness(seq, unitary, generators):
    """
    This function calculates the Frobenius distance between the matrix obtained from
    the given sequence and the target 'unitary' matrix.

    Parameters:
    - seq: A list of indices representing the sequence of anyon operations.
    - unitary: The target unitary matrix to approximate.
    - generators: A list of matrices (the anyon generators).

    Returns:
    - distance: The Frobenius norm between the resulting matrix (after sequence application)
                and the phase-adjusted identity matrix.
    """
    M = unitary
    for idx in seq:
        M = np.dot(M, generators[idx])
    trace_M = np.trace(M)
    if abs(trace_M) > 1e-12:  # to avoid division by zero
        phase = trace_M / abs(trace_M)
    else:
        phase = 1.0
    optimal_matrix = phase * np.eye(2)
    distance = np.linalg.norm(M - optimal_matrix, 'fro')
    return distance


# -----------------------------------------------------------
# (B) Crossover Methods
# -----------------------------------------------------------

def single_point_crossover(seq1, seq2):
    """
    Single-point crossover:
    Randomly choose a crossover point, then swap the genes after that point.

    Example (short sequence):
      Parent1: [0, 1, 2, 3]
      Parent2: [3, 2, 1, 0]
      Suppose the crossover point is 2
      Offspring1: [0, 1] + [1, 0] = [0, 1, 1, 0]
      Offspring2: [3, 2] + [2, 3] = [3, 2, 2, 3]
    """
    if len(seq1) < 2:
        return seq1[:], seq2[:]
    crossover_point = random.randint(1, len(seq1) - 1)
    new_seq1 = seq1[:crossover_point] + seq2[crossover_point:]
    new_seq2 = seq2[:crossover_point] + seq1[crossover_point:]
    return new_seq1, new_seq2


def two_point_crossover(seq1, seq2):
    """
    Two-point crossover:
    Choose two crossover points, then swap the segment between these two points.

    Example (short sequence):
      Parent1: [0, 1, 2, 3, 4]
      Parent2: [4, 3, 2, 1, 0]
      Suppose the points are (1, 3)
      Offspring1: [0] + [3, 2] + [3, 4] = [0, 3, 2, 3, 4]
      Offspring2: [4] + [1, 2] + [1, 0] = [4, 1, 2, 1, 0]
    """
    if len(seq1) < 4:
        return single_point_crossover(seq1, seq2)

    pt1 = random.randint(1, len(seq1) // 2)
    pt2 = random.randint(pt1 + 1, len(seq1) - 1)
    new_seq1 = seq1[:pt1] + seq2[pt1:pt2] + seq1[pt2:]
    new_seq2 = seq2[:pt1] + seq1[pt1:pt2] + seq2[pt2:]
    return new_seq1, new_seq2


def uniform_crossover(seq1, seq2):
    """
    Uniform crossover:
    For each position, randomly pick the gene from one of the two parents.

    Example (short sequence):
      Parent1: [0, 0, 0, 0]
      Parent2: [1, 1, 1, 1]
      For each index, flip a coin:
        e.g., result might be [0, 1, 1, 0]
        or [1, 0, 1, 1], etc.
    """
    new_seq1 = []
    new_seq2 = []
    for s1, s2 in zip(seq1, seq2):
        if random.random() < 0.5:
            new_seq1.append(s1)
            new_seq2.append(s2)
        else:
            new_seq1.append(s2)
            new_seq2.append(s1)
    return new_seq1, new_seq2


def block_based_crossover(seq1, seq2, block_size=5):
    """
    Block-based crossover:
    Divide each parent sequence into blocks of size block_size, then choose one block to swap.

    Example (short sequence):
      Parent1: [0, 0, 0, 1, 1, 1]
      Parent2: [2, 2, 2, 3, 3, 3]
      block_size = 3
      blocks for Parent1: [0, 0, 0], [1, 1, 1]
      blocks for Parent2: [2, 2, 2], [3, 3, 3]
      Suppose we choose the first block to swap:
      Offspring1: [2, 2, 2, 1, 1, 1]
      Offspring2: [0, 0, 0, 3, 3, 3]
    """
    length = len(seq1)
    new_seq1 = seq1[:]
    new_seq2 = seq2[:]

    block_starts = list(range(0, length, block_size))
    if len(block_starts) <= 1:
        return single_point_crossover(seq1, seq2)

    chosen_block_start = random.choice(block_starts)
    chosen_block_end = min(chosen_block_start + block_size, length)

    new_seq1[chosen_block_start:chosen_block_end], new_seq2[chosen_block_start:chosen_block_end] = \
        seq2[chosen_block_start:chosen_block_end], seq1[chosen_block_start:chosen_block_end]

    return new_seq1, new_seq2


def crossover(seq1, seq2, config):
    """
    This function performs the crossover operation depending on the 'crossover_type' 
    specified in the config.

    Valid 'crossover_type' values:
      - single_point
      - two_point
      - uniform
      - block_based
    """
    crossover_type = config.get("crossover_type", "single_point")

    if crossover_type == "single_point":
        return single_point_crossover(seq1, seq2)
    elif crossover_type == "two_point":
        return two_point_crossover(seq1, seq2)
    elif crossover_type == "uniform":
        return uniform_crossover(seq1, seq2)
    elif crossover_type == "block_based":
        block_size = config.get("block_size", 5)
        return block_based_crossover(seq1, seq2, block_size)
    else:
        return single_point_crossover(seq1, seq2)


# -----------------------------------------------------------
# (C) Mutation Methods
# -----------------------------------------------------------

def point_mutation(seq, mutation_rate, num_generators):
    """
    Point Mutation:
    Randomly replace genes with new random genes according to the 'mutation_rate'.

    Example:
      Original: [0, 1, 2, 3], num_generators=4
      mutation_rate=0.5
      Possibly mutated: [0, 1, 2, 3] (no change)
                       [1, 1, 0, 3] (some changes)
                       etc.
    """
    new_seq = seq[:]
    for i in range(len(new_seq)):
        if random.random() < mutation_rate:
            new_seq[i] = random.randint(0, num_generators - 1)
    return new_seq


def swap_mutation(seq, mutation_rate):
    """
    Swap Mutation:
    Randomly choose two positions and swap them if random.random() < mutation_rate.

    Example:
      Original: [0, 1, 2, 3]
      Suppose random.random() < mutation_rate, choose positions (1, 3)
      Result:   [0, 3, 2, 1]
    """
    new_seq = seq[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(new_seq)), 2)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    return new_seq


def inversion_mutation(seq, mutation_rate):
    """
    Inversion Mutation:
    Randomly choose a sub-range (i, j) and reverse it.

    Example:
      Original: [0, 1, 2, 3, 4]
      Suppose (i, j) = (1, 4) -> subrange is [1, 2, 3]
      Reversed subrange = [3, 2, 1]
      Result: [0, 3, 2, 1, 4]
    """
    new_seq = seq[:]
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(new_seq)), 2))
        new_seq[i:j] = reversed(new_seq[i:j])
    return new_seq


def shuffle_mutation(seq, mutation_rate):
    """
    Shuffle Mutation:
    Randomly choose a sub-range (i, j) and shuffle its elements.

    Example:
      Original: [0, 1, 2, 3, 4]
      Suppose (i, j) = (1, 4)
      Subrange: [1, 2, 3]
      Shuffled subrange could be [3, 1, 2], etc.
      Possible result: [0, 3, 1, 2, 4]
    """
    new_seq = seq[:]
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(new_seq)), 2))
        sub = new_seq[i:j]
        random.shuffle(sub)
        new_seq[i:j] = sub
    return new_seq


def block_mutation(seq, mutation_rate, num_generators, block_size=5):
    """
    Block Mutation:
    Randomly choose a sub-range of length block_size and replace it with newly random elements.

    Example:
      Original: [0, 1, 2, 3, 4, 5], block_size=3
      Suppose random.random() < mutation_rate, and we choose start=2
      The sub-range is [2, 3, 4]
      It could be replaced by [0, 3, 3] if num_generators=4
      Result: [0, 1, 0, 3, 3, 5]
    """
    new_seq = seq[:]
    length = len(new_seq)
    if random.random() < mutation_rate:
        start = random.randint(0, length - 1)
        end = min(start + block_size, length)
        for i in range(start, end):
            new_seq[i] = random.randint(0, num_generators - 1)
    return new_seq


def guided_mutation(seq, mutation_rate):
    """
    Guided Mutation:
    This requires domain-specific logic, e.g., substituting certain segments with 
    known beneficial patterns.

    Example: Not implemented
    """
    raise NotImplementedError


def mutate(seq, mutation_rate, num_generators, config):
    """
    This function performs the mutation operation depending on the 'mutation_type' 
    specified in the config.

    Valid 'mutation_type' values:
      - point
      - swap
      - inversion
      - shuffle
      - block
      - guided (not implemented)
    """
    mutation_type = config.get("mutation_type", "point")

    if mutation_type == "point":
        return point_mutation(seq, mutation_rate, num_generators)
    elif mutation_type == "swap":
        return swap_mutation(seq, mutation_rate)
    elif mutation_type == "inversion":
        return inversion_mutation(seq, mutation_rate)
    elif mutation_type == "shuffle":
        return shuffle_mutation(seq, mutation_rate)
    elif mutation_type == "block":
        block_size = config.get("block_size", 5)
        return block_mutation(seq, mutation_rate, num_generators, block_size)
    elif mutation_type == "guided":
        return guided_mutation(seq, mutation_rate)
    else:
        return point_mutation(seq, mutation_rate, num_generators)
