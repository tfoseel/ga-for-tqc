import os
import json
import csv
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import necessary functions from utils.py
from utils import fitness, crossover, mutate

# -----------------------------------------------------------
# 1. Load config.json
# -----------------------------------------------------------
CONFIG_FILE = "config.json"


def load_config(config_path=CONFIG_FILE):
    """
    Load settings from the config.json file.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


# -----------------------------------------------------------
# 2. Anyon Generators (Fibonacci anyons)
# -----------------------------------------------------------
phi = (1 + np.sqrt(5)) / 2
sigma_1 = np.array([
    [np.exp(-4j * np.pi / 5), 0],
    [0, np.exp(3j * np.pi / 5)]
])
sigma_2 = np.array([
    [np.exp(4j * np.pi / 5) / phi,
     np.exp(-3j * np.pi / 5) / np.sqrt(phi)],
    [np.exp(-3j * np.pi / 5) / np.sqrt(phi), -1 / phi]
])
sigma_1_inv = np.linalg.inv(sigma_1)
sigma_2_inv = np.linalg.inv(sigma_2)

generators = [sigma_1, sigma_2, sigma_1_inv, sigma_2_inv]
NUM_GENERATORS = len(generators)

# -----------------------------------------------------------
# 3. Genetic Algorithm
# -----------------------------------------------------------


def genetic_algorithm(unitary, config):
    """
    Run the genetic algorithm to approximate the given 'unitary'.
    It uses hyperparameters and crossover/mutation types specified in config.json.
    """
    pop_size = config.get("pop_size", 2000)
    parents_ratio = config.get("parents_ratio", 0.01)
    generations = config.get("generations", 100)
    seq_length = config.get("seq_length", 50)
    mutation_rate = config.get("mutation_rate", 0.2)

    # Set up the log folder
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root_dir = config.get("save_log_path", "logs")
    log_dir = os.path.join(log_root_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize CSV log file
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
    print(f"Crossover Type: {config.get('crossover_type', 'single_point')}")
    print(f"Mutation Type: {config.get('mutation_type', 'point')}\n")

    # Main GA loop
    with tqdm(range(generations)) as pbar:
        for generation in pbar:
            # Evaluate fitness of each sequence
            fitness_scores = [
                (seq, fitness(seq, unitary, generators))
                for seq in population
            ]
            # Sort by fitness (ascending order)
            fitness_scores.sort(key=lambda x: x[1])

            # Best sequence and fitness in this generation
            best_seq, best_fit = fitness_scores[0]

            # Update global best
            if best_fit < best_fitness_global:
                best_fitness_global = best_fit
                best_seq_global = best_seq

            best_fitness_history.append(best_fit)

            # Select elites
            num_elites = max(1, int(pop_size * parents_ratio))
            elites = [seq for seq, sc in fitness_scores[:num_elites]]

            new_population = elites[:]

            # Generate new population using crossover and mutation
            while len(new_population) < pop_size:
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)
                child1, child2 = crossover(parent1, parent2, config)
                child1 = mutate(child1, mutation_rate, NUM_GENERATORS, config)
                child2 = mutate(child2, mutation_rate, NUM_GENERATORS, config)
                new_population.extend([child1, child2])

            # Trim population to maintain original size
            population = new_population[:pop_size]

            # Update progress bar
            pbar.set_description(f'Generation {generation + 1}')
            pbar.set_postfix(best_fitness=best_fit)

            # Append this generation's data to CSV log
            with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([generation + 1, best_fit, best_seq])

            # Early stop if fitness is below threshold
            if best_fit < 0.001:
                print("Optimal solution found!")
                break

    # Final results
    print("\n=== Genetic Algorithm Finished ===")
    print(f"Best Fitness: {best_fitness_global}")
    print(f"Best Sequence: {best_seq_global}\n")

    # Save generation-by-generation best fitness plot
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


# -----------------------------------------------------------
# 4. Main (Example usage)
# -----------------------------------------------------------
if __name__ == "__main__":
    # (1) Load config.json
    config = load_config()  # CONFIG_FILE = "config.json"

    # (2) Target unitary (e.g., single-qubit Hadamard gate)
    unitary = (1 / np.sqrt(2)) * np.array([
        [1, 1],
        [1, -1]
    ])

    # (3) Check the fitness of an example sequence
    from utils import fitness  # re-import for demonstration
    seq_example = [1, 1, 2, 2, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1,
                   0, 0, 3, 3, 2, 2, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1]
    fitness_example = fitness(seq_example, unitary, generators)
    print(f"Fitness for the example sequence: {fitness_example}\n")

    # (4) Run the GA
    best_sequence, best_fitness_val, log_dir = genetic_algorithm(
        unitary, config)

    # (5) Also save the used config.json to the log folder
    config_copy_path = os.path.join(log_dir, 'config_used.json')
    with open(config_copy_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print(f"\nResults are saved in: {log_dir}")
    print(f"Best Sequence Found: {best_sequence}")
    print(f"Best Fitness: {best_fitness_val}")
