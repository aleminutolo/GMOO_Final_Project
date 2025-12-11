import random


class Individual:
    """
    Represents an AI controller's genotype for a fighting game character.

    The genotype consists of:
    - Weights: Continuous values that determine action probabilities in different situations
    - Thresholds: Continuous values that define distance categories and decision boundaries
    - Strategy: Discrete parameter that defines the general behavior style      ## NOT USED FOR NOW!!

    This uses a weighted decision tree approach where different game
    situations trigger different action sets with evolved weights.
    """

    # Strategy types (discrete parameter) ### NOT USED FOR NOW.
    STRATEGY_AGGRESSIVE = 'aggressive'
    STRATEGY_DEFENSIVE = 'defensive'
    STRATEGY_BALANCED = 'balanced'

    STRATEGIES = [STRATEGY_AGGRESSIVE, STRATEGY_DEFENSIVE, STRATEGY_BALANCED]

    def __init__(self, weights=None, thresholds=None, strategy=None):
        """
        Initialize an individual with genotype parameters.

        Args:
            weights: List of continuous values for decision making (default: random)
            thresholds: List of continuous threshold values (default: random)
            strategy: Discrete strategy type (default: random choice)
        """
        # Initialize weights for different situations
        # Total: 30 weights organized in groups
        if weights is None:
            self.weights = self._initialize_random_weights()
        else:
            self.weights = list(weights)

        # Initialize thresholds for distance categories and decision boundaries
        # [threshold_far, threshold_close, health_advantage_threshold, energy_threshold]
        if thresholds is None:
            self.thresholds = self._initialize_random_thresholds()
        else:
            self.thresholds = list(thresholds)

        # Initialize strategy
        if strategy is None:
            self.strategy = random.choice(self.STRATEGIES)
        else:
            self.strategy = strategy

        # Fitness tracking
        self.fitness = 0.0
        self.matches_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def _initialize_random_weights(self):
        """
        Initialize random weights for decision making.

        Weight groups (30 total):
        - FAR + health advantage: [FIREBALL, FORWARD, JUMP] (3 weights)
        - FAR + health disadvantage: [BLOCK, BACKWARD, FORWARD] (3 weights)
        - CLOSE + has energy: [SPECIAL, THROW, A_ATTACK] (3 weights)
        - CLOSE + no energy: [A_ATTACK, B_ATTACK, BLOCK] (3 weights)
        - MEDIUM range: [FORWARD, A_ATTACK, B_ATTACK] (3 weights)
        - JUMPING: [A_ATTACK, B_ATTACK, NO_ATTACK] (3 weights)
        - OPPONENT attacking: [BLOCK, BACKWARD, JUMP] (3 weights)
        - OPPONENT blocking: [THROW, FORWARD, BACKWARD] (3 weights)
        - LOW health (<30%): [BLOCK, BACKWARD, SPECIAL] (3 weights)
        - HIGH energy (>=3 bars): [SPECIAL, HYPER, FORWARD] (3 weights)

        Returns:
            List of 30 random weights in range [0, 1]
        """
        return [random.random() for _ in range(30)]

    def _initialize_random_thresholds(self):
        """
        Initialize random thresholds for categorization.

        Thresholds (4 total):
        - threshold_far: Distance considered "far" (range: 100-200 pixels)
        - threshold_close: Distance considered "close" (range: 20-60 pixels)
        - health_advantage_threshold: Health difference to consider advantage (range: 50-200)
        - energy_threshold: Energy level to use special moves (range: 1-2 bars)

        Returns:
            List of 4 random thresholds
        """
        return [
            random.uniform(100, 200),  # threshold_far
            random.uniform(20, 60),     # threshold_close
            random.uniform(50, 200),    # health_advantage_threshold
            random.uniform(1, 2)        # energy_threshold
        ]

    def get_weights_for_situation(self, situation_index):
        """
        Get the weights for a specific situation.

        Args:
            situation_index: Index of the situation (0-9)

        Returns:
            np.array of 3 weights for that situation
        """
        start_idx = situation_index * 3
        end_idx = start_idx + 3
        return self.weights[start_idx:end_idx]

    def normalize_weights(self, weights):
        """
        Normalize weights to sum to 1 (for probability distribution).

        Args:
            weights: List of weights

        Returns:
            Normalized weights as list
        """
        total = sum(weights)
        if total == 0:
            # If all weights are 0, return uniform distribution
            return [1.0 / len(weights)] * len(weights)
        return [w / total for w in weights]

    def update_fitness(self, match_result, health_remaining, damage_dealt,
                      avg_distance=None, damage_taken=None):
        """
        Update fitness based on match results.

        IMPROVED FITNESS FUNCTION:
        - Rewards winning heavily
        - Rewards dealing damage (encourages aggression)
        - Rewards getting close to opponent (encourages engagement)
        - Small penalty for taking damage (encourages defense)

        Args:
            match_result: 'win', 'loss', or 'draw'
            health_remaining: Health points remaining after match
            damage_dealt: Total damage dealt in the match
            avg_distance: Average distance to opponent during match (optional)
            damage_taken: Damage received during match (optional)
        """
        self.matches_played += 1

        if match_result == 'win':
            self.wins += 1
            match_score = 200  # Increased from 100 - winning is very important
        elif match_result == 'loss':
            self.losses += 1
            match_score = 0  # Changed from -50 - don't heavily penalize losses in early evolution
        else:  # draw
            self.draws += 1
            match_score = 50  # Increased from 25 - draws with damage are better than nothing

        # New improved fitness function
        # Heavily reward damage dealing (main objective)
        damage_bonus = damage_dealt * 2.0  # Was 0.3, now 2.0 - MUCH more important

        # Reward survival but less than damage
        health_bonus = health_remaining * 0.3  # Was 0.5

        # Bonus for getting close (if tracked)
        # Lower average distance = higher bonus
        proximity_bonus = 0
        if avg_distance is not None:
            # Max 50 points for staying close (avg distance < 50 pixels)
            # 0 points for staying far (avg distance > 200 pixels)
            proximity_bonus = max(0, 50 - avg_distance * 0.25)

        # Small penalty for taking damage (encourages defense)
        damage_penalty = 0
        if damage_taken is not None:
            damage_penalty = damage_taken * 0.2

        # Composite fitness
        match_fitness = (match_score + damage_bonus + health_bonus +
                        proximity_bonus - damage_penalty)

        # Update cumulative fitness (will be averaged later)
        self.fitness += match_fitness

    def get_average_fitness(self):
        """
        Get the average fitness across all matches played.

        Returns:
            Average fitness score
        """
        if self.matches_played == 0:
            return 0.0
        return self.fitness / self.matches_played

    def reset_fitness(self):
        """Reset fitness tracking for a new generation."""
        self.fitness = 0.0
        self.matches_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def clone(self):
        """
        Create a deep copy of this individual.

        Returns:
            New Individual with same genotype
        """
        return Individual(
            weights=self.weights.copy(),
            thresholds=self.thresholds.copy(),
            strategy=self.strategy
        )

    def __str__(self):
        """String representation of the individual."""
        return (f"Individual(strategy={self.strategy}, "
                f"fitness={self.get_average_fitness():.2f}, "
                f"wins={self.wins}, losses={self.losses}, draws={self.draws})")

    def __repr__(self):
        return self.__str__()


def create_random_population(size):
    """
    Create a population of random individuals.

    Args:
        size: Number of individuals to create

    Returns:
        List of Individual objects
    """
    return [Individual() for _ in range(size)]


if __name__ == "__main__":
    # Test the Individual class
    print("Testing Individual class...")

    # Create random individual
    ind = Individual()
    print(f"\nCreated: {ind}")
    print(f"Weights count: {len(ind.weights)}")
    print(f"Thresholds: {ind.thresholds}")
    print(f"Strategy: {ind.strategy}")

    # Test fitness update
    ind.update_fitness('win', 800, 500)
    ind.update_fitness('loss', 200, 300)
    ind.update_fitness('win', 600, 600)
    print(f"\nAfter 3 matches: {ind}")
    print(f"Average fitness: {ind.get_average_fitness():.2f}")

    # Test population creation
    pop = create_random_population(5)
    print(f"\nCreated population of {len(pop)} individuals:")
    for i, individual in enumerate(pop):
        print(f"  {i+1}. {individual}")

    print("\n✓ Individual class working correctly!")
