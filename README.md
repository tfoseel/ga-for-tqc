# Genetic Algorithm for Topological Quantum Compilation

This repository provides a code of using a Genetic Algorithm (GA) to approximate a single-qubit unitary gate using Topological Anyons (Fibonacci anyons).
The project is divided into two main parts:

- ```utils.py``` – Contains all crossover and mutation functions, as well as the fitness function.
- ```main.py``` – Runs the genetic algorithm using configurations specified in config.json, logs results (in CSV format), and plots fitness evolution over generations.

## Features
- Multiple Crossover Methods: Single-Point, Two-Point, Uniform, Block-Based.
- Multiple Mutation Methods: Point, Swap, Inversion, Shuffle, Block, (Placeholder for Guided).
- Configuration via config.json: Set parameters like population size, number of generations, crossover/mutation types, etc.
- Logging: Logs each generation’s best fitness and best sequence into a CSV file.
- Plotting: Automatically saves a plot of the best fitness over generations.
- Reproducibility: Saves the used ```config.json``` as ```config_used.json``` in the logs folder to keep track of all parameters.

## Repository Structure

```bash
.
├── config.json
├── main.py
├── utils.py
├── logs/
│   └── ... (Generated logs and plots)
└── README.md
```

### ```config.json```

An example config file might look like:
```json
{
  "pop_size": 500,
  "parents_ratio": 0.02,
  "generations": 50,
  "seq_length": 30,
  "mutation_rate": 0.2,
  "save_log_path": "logs",
  
  "crossover_type": "two_point",
  "mutation_type": "swap",
  
  "block_size": 5
}

```

You can customize:

- ```pop_size```: Size of the population.
- ```parents_ratio```: Fraction of the best individuals to preserve each generation (elitism).
- ```generations```: Number of evolution steps.
- ```seq_length```: Length of the chromosome (i.e., the sequence of anyon operators).
- ```mutation_rate```: Probability for mutation events.
- ```save_log_path```: Directory where logs and plots are stored.
crossover_type: Which crossover method to use. (e.g., "single_point", "two_point", "uniform", "block_based")
- ```mutation_type```: Which mutation method to use. (e.g., "point", "swap", "inversion", "shuffle", "block", "guided")
- ```block_size```: Size of a block for block-based crossover or block mutation.

## How to Run

- Clone the repository (or download the files).
- Install dependencies:
```bash
pip install -r requirements.txt
```
- Update ```config.json``` to your desired settings.
- Run the main script:
```bash
python main.py
```

## Results
- Logs Folder: After each run, a folder named with the format ```YYYYMMDD_HHMMSS``` is automatically created under ```logs/```.
- ```log.csv```: Stores generation-by-generation logs.
- ```fitness_plot.png```: A line plot showing the best fitness value over generations.
- ```config_used.json```: A copy of the config settings used for that run.

## Customization

### Adding/Modifying Crossover or Mutation Methods
In ```utils.py```:

- Crossover: Implement a new function (e.g., ```my_new_crossover```) and reference it in the crossover dispatcher:

```python
def crossover(seq1, seq2, config):
    crossover_type = config.get("crossover_type", "single_point")
    if crossover_type == "my_new_crossover":
        return my_new_crossover(seq1, seq2)
    ...
```
- Mutation: Similarly, implement a new function (e.g., ```my_new_mutation```) and reference it in the mutate dispatcher:

```python
def mutate(seq, mutation_rate, num_generators, config):
    mutation_type = config.get("mutation_type", "point")
    if mutation_type == "my_new_mutation":
        return my_new_mutation(seq, mutation_rate, num_generators)
    ...
```

Update ```config.json``` with the new type (e.g., ```"crossover_type": "my_new_crossover"```).

### Changing the Target Unitary
In ```main.py```, look for the line where unitary is defined:

```python
unitary = (1 / np.sqrt(2)) * np.array([
    [1, 1],
    [1, -1]
])
```
Replace it with your desired matrix ($2\times2$ for a single qubit, or larger if you adapt the code for multi-qubit systems).

### Using Different Anyon Generators
In ```main.py```, we define ```generators = [sigma_1, sigma_2, sigma_1_inv, sigma_2_inv]```.
If you want to use different anyon sets or additional matrices, modify this list accordingly, and make sure ```NUM_GENERATORS``` is updated to match.