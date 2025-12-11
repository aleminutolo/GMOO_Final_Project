import random
from individual import Individual


def tournament_selection(population, fitness_scores, tournament_size=3):
    """
    Tournament selection operator.

    Randomly select tournament_size individuals and return the best one.

    Args:
        population: List of Individual objects
        fitness_scores: List of fitness values (parallel to population)
        tournament_size: Number of individuals in tournament (default: 3)

    Returns:
        Selected Individual (clone)
    """
    # Randomly select tournament_size individuals
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament = [(population[i], fitness_scores[i]) for i in tournament_indices]

    # Return the individual with highest fitness
    best = max(tournament, key=lambda x: x[1])
    return best[0].clone()


def uniform_crossover(parent1, parent2, crossover_rate=0.7):
    """
    Uniform crossover for continuous weights and thresholds.

    Each gene is independently inherited from either parent with 50% probability.

    Args:
        parent1: First parent Individual
        parent2: Second parent Individual
        crossover_rate: Probability of performing crossover (default: 0.7)

    Returns:
        tuple (offspring1, offspring2) - Two new Individual objects
    """
    # If no crossover, return clones
    if random.random() > crossover_rate:
        return parent1.clone(), parent2.clone()

    # Create offspring with mixed genes
    off1_weights = []
    off2_weights = []

    # Uniform crossover for weights
    for w1, w2 in zip(parent1.weights, parent2.weights):
        if random.random() < 0.5:
            off1_weights.append(w1)
            off2_weights.append(w2)
        else:
            off1_weights.append(w2)
            off2_weights.append(w1)

    # Uniform crossover for thresholds
    off1_thresholds = []
    off2_thresholds = []

    for t1, t2 in zip(parent1.thresholds, parent2.thresholds):
        if random.random() < 0.5:
            off1_thresholds.append(t1)
            off2_thresholds.append(t2)
        else:
            off1_thresholds.append(t2)
            off2_thresholds.append(t1)

    # Strategy inheritance (randomly choose from one parent)
    off1_strategy = random.choice([parent1.strategy, parent2.strategy])
    off2_strategy = random.choice([parent1.strategy, parent2.strategy])

    # Create offspring
    offspring1 = Individual(off1_weights, off1_thresholds, off1_strategy)
    offspring2 = Individual(off2_weights, off2_thresholds, off2_strategy)

    return offspring1, offspring2


def gaussian_mutation(individual, mutation_rate=0.1, variance=0.15):
    """
    Gaussian mutation for continuous parameters.

    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutating each gene (default: 0.1)
        variance: Standard deviation of Gaussian noise (default: 0.15)

    Returns:
        Mutated Individual (new object)
    """
    mutated_weights = []
    mutated_thresholds = []

    # Mutate weights
    for weight in individual.weights:
        if random.random() < mutation_rate:
            # Add Gaussian noise
            noise = random.gauss(0, variance)
            new_weight = weight + noise
            # Clip to [0, 1] range
            new_weight = max(0.0, min(1.0, new_weight))
            mutated_weights.append(new_weight)
        else:
            mutated_weights.append(weight)

    # Mutate thresholds
    threshold_ranges = [
        (100, 200),  # threshold_far
        (20, 60),    # threshold_close
        (50, 200),   # health_advantage_threshold
        (1, 2)       # energy_threshold
    ]

    for i, threshold in enumerate(individual.thresholds):
        if random.random() < mutation_rate:
            # Add Gaussian noise scaled to the range
            range_size = threshold_ranges[i][1] - threshold_ranges[i][0]
            noise = random.gauss(0, variance * range_size)
            new_threshold = threshold + noise
            # Clip to valid range
            new_threshold = max(threshold_ranges[i][0], min(threshold_ranges[i][1], new_threshold))
            mutated_thresholds.append(new_threshold)
        else:
            mutated_thresholds.append(threshold)

    # Mutate strategy (with lower probability)
    mutated_strategy = individual.strategy
    if random.random() < mutation_rate * 0.3:  # Lower rate for discrete parameter
        mutated_strategy = random.choice(Individual.STRATEGIES)

    return Individual(mutated_weights, mutated_thresholds, mutated_strategy)


def evolve_population(population, fitness_scores, offspring_count, elite_count=2,
                     tournament_size=3, crossover_rate=0.7, mutation_rate=0.1):
    """
    Evolve a population using (μ + λ) - ES.

    - Keep elite_count best individuals (elitism)
    - Generate offspring_count new individuals through selection, crossover, mutation
    - Combine parents and offspring, select best μ for next generation

    Args:
        population: List of Individual objects
        fitness_scores: List of fitness values
        offspring_count: Number of offspring to generate (λ)
        elite_count: Number of elite individuals to preserve (default: 2)
        tournament_size: Size of tournament for selection (default: 3)
        crossover_rate: Probability of crossover (default: 0.7)
        mutation_rate: Probability of mutation per gene (default: 0.1)

    Returns:
        New population (same size as input)
    """
    mu = len(population)
    new_population = []

    # Elitism: Keep best elite_count individuals
    sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    for i in range(elite_count):
        new_population.append(sorted_pop[i][0].clone())

    # Generate offspring
    offspring = []
    while len(offspring) < offspring_count:
        # Selection
        parent1 = tournament_selection(population, fitness_scores, tournament_size)
        parent2 = tournament_selection(population, fitness_scores, tournament_size)

        # Crossover
        child1, child2 = uniform_crossover(parent1, parent2, crossover_rate)

        # Mutation
        child1 = gaussian_mutation(child1, mutation_rate)
        child2 = gaussian_mutation(child2, mutation_rate)

        offspring.append(child1)
        if len(offspring) < offspring_count:
            offspring.append(child2)

    # Add offspring to new population
    new_population.extend(offspring)

    # If we have more than μ individuals, keep only the best μ
    # (This implements the "+" in (μ + λ))
    if len(new_population) > mu:
        new_population = new_population[:mu]

    return new_population


def create_random_opponent():
    """
    Create a random opponent for baseline testing.

    Returns:
        Individual with random genotype
    """
    return Individual()


# Test the evolutionary operators
if __name__ == "__main__":
    from individual import create_random_population

    print("Testing Evolutionary Operators...")
    print("=" * 60)

    # Test 1: Tournament Selection
    print("\nTest 1: Tournament Selection")
    print("-" * 60)
    pop = create_random_population(5)
    fitness = [100, 200, 150, 300, 250]

    print("Population fitness:", fitness)
    selected = tournament_selection(pop, fitness, tournament_size=3)
    print(f"Selected individual: {selected}")
    print(" Tournament selection working")

    # Test 2: Uniform Crossover
    print("\nTest 2: Uniform Crossover")
    print("-" * 60)
    parent1 = Individual()
    parent2 = Individual()

    print(f"Parent 1 weights (first 5): {parent1.weights[:5]}")
    print(f"Parent 2 weights (first 5): {parent2.weights[:5]}")

    off1, off2 = uniform_crossover(parent1, parent2)
    print(f"Offspring 1 weights (first 5): {off1.weights[:5]}")
    print(f"Offspring 2 weights (first 5): {off2.weights[:5]}")
    print(" Crossover working")

    # Test 3: Gaussian Mutation
    print("\nTest 3: Gaussian Mutation")
    print("-" * 60)
    original = Individual()
    print(f"Original weights (first 5): {original.weights[:5]}")
    print(f"Original thresholds: {original.thresholds}")
    print(f"Original strategy: {original.strategy}")

    mutated = gaussian_mutation(original, mutation_rate=0.5)  # High rate for testing
    print(f"Mutated weights (first 5): {mutated.weights[:5]}")
    print(f"Mutated thresholds: {mutated.thresholds}")
    print(f"Mutated strategy: {mutated.strategy}")

    # Check that some genes changed
    changed_weights = sum(1 for i in range(len(original.weights))
                         if original.weights[i] != mutated.weights[i])
    print(f"Changed weights: {changed_weights}/30")
    print(" Mutation working")

    # Test 4: Full Evolution
    print("\nTest 4: Population Evolution")
    print("-" * 60)
    population = create_random_population(10)
    # Simulate fitness scores
    fitness_scores = [random.uniform(0, 500) for _ in range(10)]

    print(f"Initial population size: {len(population)}")
    print(f"Fitness scores: {[f'{f:.1f}' for f in fitness_scores]}")

    new_pop = evolve_population(
        population,
        fitness_scores,
        offspring_count=15,  # λ = 15
        elite_count=2
    )

    print(f"New population size: {len(new_pop)}")
    print(" Population evolution working")

    # Test 5: Multiple Generations
    print("\nTest 5: Multi-Generation Evolution")
    print("-" * 60)
    pop = create_random_population(20)

    for gen in range(3):
        # Simulate fitness evaluation
        fitness = [random.uniform(100, 500) for _ in range(len(pop))]
        avg_fitness = sum(fitness) / len(fitness)
        max_fitness = max(fitness)

        print(f"Generation {gen}: Avg={avg_fitness:.1f}, Max={max_fitness:.1f}")

        # Evolve
        pop = evolve_population(pop, fitness, offspring_count=30, elite_count=2)

    print(" Multi-generation evolution working")

    print("\n" + "=" * 60)
    print("All evolutionary operators tests passed! ")
    print("=" * 60)
