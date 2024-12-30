import os
import csv
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import GA utility functions from utils.py
from utils import fitness, crossover, mutate

# Import all settings from config.py
import config

phi = (1 + np.sqrt(5)) / 2
sigma_1 = np.array([
    [np.exp(-4j * np.pi / 5), 0],
    [0, np.exp(3j * np.pi / 5)]
])
sigma_2 = np.array([
    [np.exp(4j * np.pi / 5) / phi,
     np.exp(-3j * np.pi / 5) / np.sqrt(phi)],
    [np.exp(-3j * np.pi / 5) / np.sqrt(phi),               -1 / phi]
])
sigma_1_inv = np.linalg.inv(sigma_1)
sigma_2_inv = np.linalg.inv(sigma_2)

generators = [sigma_1, sigma_2, sigma_1_inv, sigma_2_inv]
NUM_GENERATORS = len(generators)


def copy_config_file(src, dst):
    """
    Copies the entire config.py to the logs folder as config_used.py
    to keep track of the exact config used in each run.
    """
    with open(src, 'r', encoding='utf-8') as f_in:
        with open(dst, 'w', encoding='utf-8') as f_out:
            f_out.write(f_in.read())


def genetic_algorithm(unitary):
    """
    Run the genetic algorithm to approximate the given 'unitary' matrix.
    All hyperparameters and GA settings are defined in config.py.
    """
    # Load hyperparameters from config.py
    pop_size = config.pop_size
    parents_ratio = config.parents_ratio
    generations = config.generations
    seq_length = config.seq_length
    mutation_rate = config.mutation_rate

    # Prepare log directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config.save_log_path, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Copy the original config.py to config_used.py in the logs folder
    copy_config_file("config.py", os.path.join(log_dir, "config_used.py"))

    # CSV log initialization
    csv_file_path = os.path.join(log_dir, 'log.csv')
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Generation", "Best Fitness", "Best Sequence"])

    # Create initial population
    population = [
        [random.randint(0, NUM_GENERATORS - 1) for _ in range(seq_length)]
        for _ in range(pop_size)
    ]

    best_fitness_history = []
    best_seq_global = None
    best_fitness_global = float('inf')

    print("=== Genetic Algorithm Start ===")
    print(f"Population Size: {pop_size}")
    print(f"Parents Ratio: {parents_ratio}")
    print(f"Generations: {generations}")
    print(f"Sequence Length: {seq_length}")
    print(f"Mutation Rate: {mutation_rate}")
    print(f"Crossover Type: {config.crossover_type}")
    print(f"Mutation Type: {config.mutation_type}\n")

    # Main GA loop
    with tqdm(range(generations)) as pbar:
        for generation in pbar:
            # Evaluate fitness
            fitness_scores = [
                (seq, fitness(seq, unitary, generators))
                for seq in population
            ]
            fitness_scores.sort(key=lambda x: x[1])  # ascending order

            # Best in this generation
            best_seq, best_fit = fitness_scores[0]

            # Update global best
            if best_fit < best_fitness_global:
                best_fitness_global = best_fit
                best_seq_global = best_seq

            best_fitness_history.append(best_fit)

            # Elitism
            num_elites = max(1, int(pop_size * parents_ratio))
            elites = [seq for seq, _ in fitness_scores[:num_elites]]

            new_population = elites[:]

            # Create new individuals
            while len(new_population) < pop_size:
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)

                # We pass some config values as a dict to crossover/mutate
                child1, child2 = crossover(parent1, parent2, {
                    "crossover_type": config.crossover_type,
                    "block_size": config.block_size
                })
                child1 = mutate(child1, mutation_rate, NUM_GENERATORS, {
                    "mutation_type": config.mutation_type,
                    "block_size": config.block_size
                })
                child2 = mutate(child2, mutation_rate, NUM_GENERATORS, {
                    "mutation_type": config.mutation_type,
                    "block_size": config.block_size
                })
                new_population.extend([child1, child2])

            population = new_population[:pop_size]

            # Logging
            pbar.set_description(f'Generation {generation + 1}')
            pbar.set_postfix(best_fitness=best_fit)

            with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([generation + 1, best_fit, best_seq])

            # Early stop
            if best_fit < 0.001:
                print("Optimal solution found!")
                break

    # Final results
    print("\n=== Genetic Algorithm Finished ===")
    print(f"Best Fitness: {best_fitness_global}")
    print(f"Best Sequence: {best_seq_global}\n")

    # Plot best fitness curve
    plt.figure(figsize=(8, 5))
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Genetic Algorithm - Fitness over Generations')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(log_dir, 'fitness_plot.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return best_seq_global, best_fitness_global, log_dir


# Main entry
if __name__ == "__main__":
    # (1) load unitary from config
    unitary = config.target_unitary

    # (2) Optional: check fitness of an example sequence
    seq_example = [1, 1, 2, 2, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1,
                   0, 0, 3, 3, 2, 2, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1]

    fitness_example = fitness(seq_example, unitary, generators)
    print(f"Fitness for the example sequence: {fitness_example}\n")

    # (3) run GA
    best_sequence, best_fitness_val, log_dir = genetic_algorithm(unitary)

    print(f"Results are saved in: {log_dir}")
