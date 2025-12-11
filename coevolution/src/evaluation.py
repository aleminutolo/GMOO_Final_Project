import pygame
import sys
import os
import random

# Add game path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Street_pyghter/src'))

from Round import Player, Background, STATECONST, KEYCONST, UI
from game import Point, Vector
import config

from controller import AIController, GameState
from individual import Individual


class FightSimulator:
    """
    Manages fight simulation between two AI individuals.
    Can run in headless mode for faster execution.
    """

    def __init__(self, headless=True):
        """
        Initialize fight simulator.

        Args:
            headless: If True, run without rendering (much faster)
        """
        self.headless = headless
        self.pygame_initialized = False

    def _initialize_pygame(self):
        """Initialize pygame if not already done."""
        if not self.pygame_initialized:
            # Save and change directory for resource loading
            self.original_dir = os.getcwd()
            game_src_dir = os.path.join(os.path.dirname(__file__), '../../Street_pyghter/src')
            self.game_src_dir = os.path.abspath(game_src_dir)
            os.chdir(self.game_src_dir)

            # Set SDL to use dummy video driver for headless mode
            if self.headless:
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
                os.environ['SDL_AUDIODRIVER'] = 'dummy'

            pygame.init()
            if not self.headless:
                pygame.mixer.init()

            self.pygame_initialized = True

    def run_fight(self, individual1, individual2, character1='Ken', character2='Zangief',
                  max_time=60, render=False):
        """
        Run a fight between two individuals.

        Args:
            individual1: First individual (player 1)
            individual2: Second individual (player 2)
            character1: Character name for player 1
            character2: Character name for player 2
            max_time: Maximum fight duration in seconds (default: 60)
            render: Whether to render the fight (only works if not headless)

        Returns:
            dict with fight results:
            {
                'winner': 1, 2, or 0 (draw),
                'p1_health': final health of player 1,
                'p2_health': final health of player 2,
                'p1_damage': damage dealt by player 1,
                'p2_damage': damage dealt by player 2,
                'frames': number of frames
            }
        """
        self._initialize_pygame()

        # Create screen
        if self.headless:
            # In headless mode, we still need to set a display mode for pygame to work
            screen = pygame.display.set_mode((320, 240))
        elif not render:
            screen = pygame.Surface((320, 240), 0, 32)
        else:
            screen = pygame.display.set_mode((640, 480), 0, 32)
            pygame.display.set_caption(f"{character1} vs {character2}")

        # Load characters
        player1 = Player(character1, 120, 100)
        player2 = Player(character2, 120, 100, Player2=True, alt_color=True)

        # Load background
        background = Background('../res/Background/Bckgrnd0.png')

        # Create AI controllers
        controller1 = AIController(individual1, character1)
        controller2 = AIController(individual2, character2)

        # Initialize positions
        player1.reinit(Point(40, 195), Point(280, 195))
        player2.reinit(Point(280, 195), Point(40, 195))
        player2.facingRight = False

        # Track initial health
        initial_hp1 = player1.health.hp
        initial_hp2 = player2.health.hp

        # Create UI with proper health bars and timer
        # For render mode, use full UI; for headless, use simple timer only
        if render and not self.headless:
            # Convert max_time to UI time index: 99s=3, 60s=2, 30s=1
            if max_time >= 99:
                time_index = 3
            elif max_time >= 60:
                time_index = 2
            else:
                time_index = 1
            ui = UI(time=time_index, rounds=0)
        else:
            # Simple UI for headless/non-render mode (timer tracking only)
            class SimpleTimer:
                def __init__(self, max_time):
                    self.time = max_time * 60  # Convert to frames (60 FPS)
                    self.tick_counter = 0

                def tick_me(self, rate):
                    self.tick_counter += 1
                    if self.tick_counter >= 60 * rate:  # Every second
                        self.tick_counter = 0
                        if self.time > 0:
                            self.time -= 1

            class SimpleUI:
                def __init__(self):
                    self.timer = SimpleTimer(max_time)

                def tick_me(self, rate):
                    self.timer.tick_me(rate)

            ui = SimpleUI()

        # Main fight loop
        frame_count = 0
        max_frames = max_time * 60  # 60 FPS
        clock = pygame.time.Clock()

        # Track distance for fitness calculation
        total_distance = 0
        distance_samples = 0

        while frame_count < max_frames:
            frame_count += 1

            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    break

            # Get AI inputs
            game_state1 = GameState(player1, player2, ui)
            game_state2 = GameState(player2, player1, ui)

            # Track distance every 10 frames
            if frame_count % 10 == 0:
                distance = abs(player1.position.x - player2.position.x)
                total_distance += distance
                distance_samples += 1

            stick1, btn1 = controller1.decide_action(game_state1)
            stick2, btn2 = controller2.decide_action(game_state2)

            # Set inputs
            player1.setInputs(stick1, btn1)
            player2.setInputs(stick2, btn2)

            # Update game logic
            player1.action(player2, ui.timer.time)
            player2.action(player1, ui.timer.time)

            # Tick animations to advance frames (required for movement!)
            player1.tick_me(2)
            player2.tick_me(2)

            # Update background scrolling
            center = (player1.position + player2.position) / 2
            scrolling = background.action(center)

            # Move characters (pass scrolling vector!)
            player1.move(scrolling, player2.position, player2.getState())
            player2.move(scrolling, player1.position, player1.getState())

            # Check attacks
            player1.attack(player2)
            player2.attack(player1)

            # Render if requested
            if render and not self.headless:
                screen.fill((0, 0, 0))
                background.print_me(screen)
                player1.print_me(screen)
                player2.print_me(screen)

                # Render UI with health bars and timer
                ui.print_me(screen, player1.health, player2.health,
                           player1.combo_count, player2.combo_count)

                # Scale and display
                scaled = pygame.transform.scale2x(screen)
                display_screen = pygame.display.get_surface()
                if display_screen:
                    display_screen.blit(scaled, (0, 0))
                    pygame.display.flip()

            # Check win conditions
            if player1.health.amIdead() or player2.health.amIdead() or ui.timer.time == 0:
                break

            # Tick timer
            ui.tick_me(1)

            # Control framerate (faster in headless mode)
            if not self.headless and render:
                clock.tick(60)
            else:
                clock.tick(120)  # Faster simulation

        # Determine winner
        p1_hp = player1.health.hp
        p2_hp = player2.health.hp

        if p1_hp > p2_hp:
            winner = 1
        elif p2_hp > p1_hp:
            winner = 2
        else:
            winner = 0

        # Calculate damage dealt and taken
        p1_damage = initial_hp2 - p2_hp
        p2_damage = initial_hp1 - p1_hp

        # Calculate average distance
        avg_distance = total_distance / distance_samples if distance_samples > 0 else 200

        result = {
            'winner': winner,
            'p1_health': p1_hp,
            'p2_health': p2_hp,
            'p1_damage': p1_damage,
            'p2_damage': p2_damage,
            'avg_distance': avg_distance,
            'frames': frame_count
        }

        # Close display if opened
        if render and not self.headless:
            pygame.display.quit()

        return result

    def cleanup(self):
        """Restore original directory."""
        if self.pygame_initialized:
            os.chdir(self.original_dir)


def evaluate_individual(individual, opponents, simulator, character='Ken', opponent_character='Zangief'):
    """
    Evaluate an individual against multiple opponents (K-fold evaluation).

    Args:
        individual: Individual to evaluate
        opponents: List of opponent Individuals (or single opponent)
        simulator: FightSimulator instance
        character: Character name for the individual
        opponent_character: Character name for opponents

    Returns:
        float: Fitness score
    """
    # Handle single opponent
    if isinstance(opponents, Individual):
        opponents = [opponents]

    # Reset individual fitness
    individual.reset_fitness()

    # Fight against each opponent
    for opponent in opponents:
        result = simulator.run_fight(
            individual,
            opponent,
            character1=character,
            character2=opponent_character,
            max_time=30,  # 30 seconds per match
            render=False
        )

        # Update fitness based on result
        if result['winner'] == 1:
            match_result = 'win'
        elif result['winner'] == 2:
            match_result = 'loss'
        else:
            match_result = 'draw'

        individual.update_fitness(
            match_result,
            result['p1_health'],
            result['p1_damage'],
            avg_distance=result['avg_distance'],
            damage_taken=result['p2_damage']
        )

    return individual.get_average_fitness()


def evaluate_population(population, opponents, simulator, character='Ken',
                       opponent_character='Rick', k_matches=5):
    """
    Evaluate an entire population using K-fold evaluation.

    Each individual fights against k_matches randomly selected opponents.

    Args:
        population: List of Individuals to evaluate
        opponents: List of opponent Individuals
        simulator: FightSimulator instance
        character: Character for population
        opponent_character: Character for opponents
        k_matches: Number of matches per individual (K-fold)

    Returns:
        List of fitness scores (parallel to population)
    """
    fitness_scores = []

    for i, individual in enumerate(population):
        # Select k random opponents
        selected_opponents = random.sample(opponents, min(k_matches, len(opponents)))

        # Evaluate
        fitness = evaluate_individual(
            individual,
            selected_opponents,
            simulator,
            character,
            opponent_character
        )

        fitness_scores.append(fitness)

        # Progress indicator
        if (i + 1) % 5 == 0 or (i + 1) == len(population):
            print(f"  Evaluated {i+1}/{len(population)} individuals...")

    return fitness_scores


# Test evaluation
if __name__ == "__main__":
    from individual import Individual, create_random_population

    print("=" * 60)
    print("Testing Evaluation Module")
    print("=" * 60)

    # Test 1: Single Fight
    print("\nTest 1: Single Fight Simulation")
    print("-" * 60)

    ind1 = Individual()
    ind2 = Individual()

    simulator = FightSimulator(headless=True)

    print("Running fight between two random individuals...")
    result = simulator.run_fight(ind1, ind2, character1='Ken', character2='Rick', max_time=10)

    print(f"Winner: Player {result['winner']}")
    print(f"Player 1: {result['p1_health']} HP, {result['p1_damage']} damage dealt")
    print(f"Player 2: {result['p2_health']} HP, {result['p2_damage']} damage dealt")
    print(f"Frames: {result['frames']}")
    print(" Single fight working")

    # Test 2: Individual Evaluation
    print("\nTest 2: Individual Evaluation (K-fold)")
    print("-" * 60)

    individual = Individual()
    opponents = create_random_population(3)

    print(f"Evaluating individual against {len(opponents)} opponents...")
    fitness = evaluate_individual(individual, opponents, simulator)

    print(f"Individual: {individual}")
    print(f"Fitness: {fitness:.2f}")
    print(" Individual evaluation working")

    # Test 3: Population Evaluation
    print("\nTest 3: Population Evaluation")
    print("-" * 60)

    population = create_random_population(5)
    opponents = create_random_population(5)

    print(f"Evaluating population of {len(population)} against {len(opponents)} opponents...")
    fitness_scores = evaluate_population(population, opponents, simulator, k_matches=2)

    print("\nFitness scores:")
    for i, (ind, fit) in enumerate(zip(population, fitness_scores)):
        print(f"  Individual {i+1}: {fit:.2f} (W:{ind.wins} L:{ind.losses} D:{ind.draws})")

    print(" Population evaluation working")

    # Cleanup
    simulator.cleanup()

    print("\n" + "=" * 60)
    print("All evaluation tests passed! ")
    print("=" * 60)
