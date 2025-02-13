import random
import numpy as np

# -----------------------------------------------------------
# (A) Fitness function
# -----------------------------------------------------------


def fitness(seq, unitary, generators_dict):
    """
    Calculate the Frobenius distance between the final matrix (obtained by
    applying the generator sequence to 'unitary') and the phase-adjusted identity.

    Parameters
    ----------
    seq : list of int
        A list of indices (possibly including -1 for identity).
    unitary : np.array
        The target unitary (2x2) we multiply from the left or measure from.
    generators_dict : dict
        A dictionary mapping indices to 2x2 matrices, e.g. {-1: I, 0: sigma_1, 1: sigma_2, ...}.

    Returns
    -------
    distance : float
        The Frobenius norm distance to the phase-adjusted identity.

    Example
    -------
    >>> generators_dict = {
    ...     -1: np.eye(2),
    ...      0: sigma_1,
    ...      1: sigma_2,
    ...      2: sigma_1_inv,
    ...      3: sigma_2_inv
    ... }
    >>> seq = [0, 1, -1, 2]
    >>> dist = fitness(seq, target_unitary, generators_dict)
    >>> print(dist)
    0.314159...  # some float
    """
    M = unitary
    for idx in seq:
        M = np.dot(M, generators_dict[idx])
    trace_M = np.trace(M)
    if abs(trace_M) > 1e-12:
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
    Single-point crossover.

    Example
    -------
    >>> seq1 = [0, 0, 1, 1]
    >>> seq2 = [1, 1, 2, 2]
    >>> child1, child2 = single_point_crossover(seq1, seq2)
    >>> # Suppose the point is 2
    >>> # child1 might be [0, 0] + [2, 2] = [0, 0, 2, 2]
    >>> # child2 might be [1, 1] + [1, 1] = [1, 1, 1, 1]
    """
    if len(seq1) < 2:
        return seq1[:], seq2[:]
    point = random.randint(1, len(seq1) - 1)
    child1 = seq1[:point] + seq2[point:]
    child2 = seq2[:point] + seq1[point:]
    return child1, child2


def two_point_crossover(seq1, seq2):
    """
    Two-point crossover.

    Example
    -------
    >>> seq1 = [0, 0, 0, 1, 1]
    >>> seq2 = [1, 1, 2, 2, 2]
    >>> child1, child2 = two_point_crossover(seq1, seq2)
    >>> # Suppose points are (1, 3)
    >>> # child1 = [0] + [1, 2] + [1] = [0, 1, 2, 1]
    >>> # child2 = [1] + [0, 0] + [2] = [1, 0, 0, 2]
    """
    length = len(seq1)
    if length < 4:
        return single_point_crossover(seq1, seq2)

    pt1 = random.randint(1, length // 2)
    pt2 = random.randint(pt1 + 1, length - 1)
    child1 = seq1[:pt1] + seq2[pt1:pt2] + seq1[pt2:]
    child2 = seq2[:pt1] + seq1[pt1:pt2] + seq2[pt2:]
    return child1, child2


def uniform_crossover(seq1, seq2):
    """
    Uniform crossover: For each position, randomly pick from one parent or the other.

    Example
    -------
    >>> seq1 = [0, 0, 1, 1]
    >>> seq2 = [1, -1, 2, 2]
    >>> child1, child2 = uniform_crossover(seq1, seq2)
    >>> # For each index i, pick seq1[i] or seq2[i] with ~50% chance
    >>> # child1 = [0, -1, 2, 1], child2 = [1, 0, 1, 2] etc.
    """
    length = len(seq1)
    child1 = []
    child2 = []
    for i in range(length):
        if random.random() < 0.5:
            child1.append(seq1[i])
            child2.append(seq2[i])
        else:
            child1.append(seq2[i])
            child2.append(seq1[i])
    return child1, child2


def block_based_crossover(seq1, seq2, block_size=5):
    """
    Block-based crossover:
    Pick a block of length block_size and swap it.
    If the sequence is too short or block coverage is minimal,
    fallback to single-point.

    Example
    -------
    >>> seq1 = [0, 0, 0, 1, 1, 1]
    >>> seq2 = [1, 1, 2, 2, 2, 2]
    >>> child1, child2 = block_based_crossover(seq1, seq2, block_size=3)
    >>> # Suppose the block chosen is indices [1..4)
    >>> # child1 = [0, 1, 2, 2] + [1, 1] = ...
    >>> # child2 = [1, 0, 0, 1] + [2, 2] = ...
    """
    length = len(seq1)
    if length <= block_size:
        return single_point_crossover(seq1, seq2)

    child1 = seq1[:]
    child2 = seq2[:]
    start = random.randint(0, length - block_size)
    end = start + block_size
    child1[start:end], child2[start:end] = seq2[start:end], seq1[start:end]
    return child1, child2


def crossover(seq1, seq2, config):
    """
    Dispatch function for crossover, based on 'crossover_type' in config.
    """
    crossover_type = config.get("crossover_type", "single_point")
    block_size = config.get("block_size", 5)

    if crossover_type == "single_point":
        return single_point_crossover(seq1, seq2)
    elif crossover_type == "two_point":
        return two_point_crossover(seq1, seq2)
    elif crossover_type == "uniform":
        return uniform_crossover(seq1, seq2)
    elif crossover_type == "block_based":
        return block_based_crossover(seq1, seq2, block_size)
    else:
        # default
        return single_point_crossover(seq1, seq2)


# -----------------------------------------------------------
# (C) Mutation Methods
# -----------------------------------------------------------

def point_mutation(seq, mutation_rate, generators_dict, config):
    """
    Point Mutation:
    Replace random positions with random valid indices (including -1 for identity).

    Example
    -------
    >>> seq = [0, 0, 1, 1, -1]
    >>> new_seq = point_mutation(seq, 0.2, generators_dict, {"possible_indices": [-1, 0, 1, 2, 3]})
    >>> # With 20% chance per position, pick a new index from [-1, 0, 1, 2, 3]
    """
    possible_indices = config.get(
        "possible_indices", list(generators_dict.keys()))
    new_seq = seq[:]
    for i in range(len(new_seq)):
        if random.random() < mutation_rate:
            new_seq[i] = random.choice(possible_indices)
    return new_seq


def swap_mutation(seq, mutation_rate, **kwargs):
    """
    Swap Mutation:
    Randomly swap two positions in the sequence if random.random() < mutation_rate.

    Example
    -------
    >>> seq = [0, 1, 1, 2]
    >>> new_seq = swap_mutation(seq, 0.5)
    >>> # With 50% chance, we might swap e.g. positions (1, 3) => [0, 2, 1, 1]
    """
    new_seq = seq[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(new_seq)), 2)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    return new_seq


def inversion_mutation(seq, mutation_rate, **kwargs):
    """
    Inversion Mutation:
    Pick a subrange and reverse it.

    Example
    -------
    >>> seq = [0, 1, 2, 3, -1]
    >>> new_seq = inversion_mutation(seq, 0.8)
    >>> # With 80% chance, choose e.g. i=1, j=4 => subrange [1,2,3], reversed to [3,2,1]
    >>> # new_seq => [0, 3, 2, 1, -1]
    """
    new_seq = seq[:]
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(new_seq)), 2))
        new_seq[i:j] = reversed(new_seq[i:j])
    return new_seq


def shuffle_mutation(seq, mutation_rate, **kwargs):
    """
    Shuffle Mutation:
    Pick a subrange and shuffle it randomly.

    Example
    -------
    >>> seq = [0, 1, 2, 3, -1]
    >>> new_seq = shuffle_mutation(seq, 0.8)
    >>> # With 80% chance, choose e.g. subrange i=1, j=4 => [1,2,3], shuffle => [2,1,3]
    >>> # new_seq => [0, 2, 1, 3, -1]
    """
    new_seq = seq[:]
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(new_seq)), 2))
        sub = new_seq[i:j]
        random.shuffle(sub)
        new_seq[i:j] = sub
    return new_seq


def block_mutation(seq, mutation_rate, generators_dict, config):
    """
    Block Mutation:
    Replace a subrange of length block_size with new random elements.

    Example
    -------
    >>> seq = [0, 1, 2, 3, -1, 0]
    >>> config = {"block_size": 3, "possible_indices": [-1, 0, 1, 2, 3]}
    >>> new_seq = block_mutation(seq, 0.5, generators_dict, config)
    >>> # With 50% chance, pick e.g. start=2 => subrange [2,3,-1], replace each with random
    """
    block_size = config.get("block_size", 5)
    possible_indices = config.get(
        "possible_indices", list(generators_dict.keys()))

    new_seq = seq[:]
    length = len(new_seq)
    if random.random() < mutation_rate:
        start = random.randint(0, max(0, length - block_size))
        end = start + block_size
        for i in range(start, min(end, length)):
            new_seq[i] = random.choice(possible_indices)
    return new_seq


def guided_mutation(seq, mutation_rate, **kwargs):
    """
    Guided Mutation:
    Placeholder - domain-specific logic to replace certain segments with
    known beneficial patterns.

    Example
    -------
    This function is not implemented. If you need domain knowledge,
    implement your own logic for known better subsequences.
    """
    raise NotImplementedError("Guided mutation is not implemented.")


# -----------------------------------------------------------
# (D) New Mutation Methods - Segment Rotation & Segment Conjugation
# -----------------------------------------------------------

def segment_rotation_mutation(seq, mutation_rate, **kwargs):
    """
    Segment Rotation Mutation:
    - Randomly choose a subrange [i, j).
    - Rotate the elements in that subrange by some random offset k (1 <= k < length_of_subrange).

    Example
    -------
    >>> seq = [0, 1, 2, 3, 3, 2]
    >>> new_seq = segment_rotation_mutation(seq, 0.5)
    >>> # With 50% chance, pick e.g. subrange i=1, j=5 => [1,2,3,3]
    >>> # Suppose k=2 => rotate => [3,3,1,2]
    >>> # new_seq => [0, 3, 3, 1, 2, 2]
    """
    new_seq = seq[:]
    if random.random() < mutation_rate and len(new_seq) > 1:
        i, j = sorted(random.sample(range(len(new_seq)), 2))
        subrange = new_seq[i:j]  # subrange to rotate
        if len(subrange) > 1:
            k = random.randint(1, len(subrange) - 1)
            rotated = subrange[k:] + subrange[:k]
            new_seq[i:j] = rotated
    return new_seq


def segment_conjugation_mutation(seq, mutation_rate, generators_dict, config):
    """
    Segment Conjugation Mutation:
    - Look for positions in the sequence that are '-1' (identity).
    - Randomly pick two distinct identity positions => define subrange S between them.
    - Then replace those two '-1' with a random pair of (g, g_inv) from:
        (sigma_1, sigma_1_inv), (sigma_1_inv, sigma_1),
        (sigma_2, sigma_2_inv), (sigma_2_inv, sigma_2).
    - This simulates a conjugation of the subrange S by that generator.

    Example
    -------
    >>> seq = [-1, 0, 0, -1, 3, 1]
    >>> # Suppose identity_positions = [0, 3].
    >>> # We pick (sigma_1, sigma_1_inv) => (0, 2)
    >>> # Then seq => [0, 0, 0, 2, 3, 1]
    >>> # The subrange between 0..3 is effectively "conjugated" by sigma_1.
    >>> new_seq = segment_conjugation_mutation(seq, 0.8, generators_dict, config)
    """
    new_seq = seq[:]
    if random.random() < mutation_rate:
        identity_positions = [idx for idx,
                              val in enumerate(new_seq) if val == -1]
        if len(identity_positions) < 2:
            return new_seq

        pos1, pos2 = sorted(random.sample(identity_positions, 2))
        conjugation_candidates = [
            (0, 2),  # (sigma_1, sigma_1_inv)
            (2, 0),  # (sigma_1_inv, sigma_1)
            (1, 3),  # (sigma_2, sigma_2_inv)
            (3, 1)   # (sigma_2_inv, sigma_2)
        ]
        pair = random.choice(conjugation_candidates)
        new_seq[pos1] = pair[0]
        new_seq[pos2] = pair[1]

    return new_seq


# -----------------------------------------------------------
# (E) Final Mutate Dispatcher
# -----------------------------------------------------------
def mutate(seq, mutation_rate, generators_dict, config):
    """
    Dispatch function for mutation, based on 'mutation_type' in config.

    Additional possible values for 'mutation_type' now include:
    - 'segment_rotation'
    - 'segment_conjugation'
    """
    mutation_type = config.get("mutation_type", "point")

    if mutation_type == "point":
        return point_mutation(seq, mutation_rate, generators_dict, config)
    elif mutation_type == "swap":
        return swap_mutation(seq, mutation_rate)
    elif mutation_type == "inversion":
        return inversion_mutation(seq, mutation_rate)
    elif mutation_type == "shuffle":
        return shuffle_mutation(seq, mutation_rate)
    elif mutation_type == "block":
        return block_mutation(seq, mutation_rate, generators_dict, config)
    elif mutation_type == "guided":
        return guided_mutation(seq, mutation_rate)
    elif mutation_type == "segment_rotation":
        return segment_rotation_mutation(seq, mutation_rate)
    elif mutation_type == "segment_conjugation":
        return segment_conjugation_mutation(seq, mutation_rate, generators_dict, config)
    else:
        raise NameError("Not found: Run with point mutation (default)")
