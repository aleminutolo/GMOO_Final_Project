import sys
import os
import time
import json

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from individual import Individual, create_random_population
from evolution import evolve_population
from evaluation import FightSimulator, evaluate_population


def run_coevolution_experiment(
    pop_size=20,
    generations=20,
    offspring_count=30,
    k_matches=3,
    tournament_size=3,
    character_a='Ken',
    character_b='Zangief'
):
    """
    Run two-population competitive coevolution.

    Population A and Population B evolve simultaneously, each trying to beat the other.

    Args:
        pop_size: Size of each population (μ)
        generations: Number of generations to evolve
        offspring_count: Offspring per generation (λ)
        k_matches: K-fold evaluation matches
        tournament_size: Tournament size for selection (default: 3)
        character_a: Character for population A
        character_b: Character for population B

    Returns:
        Dictionary with experiment results
    """
    print("=" * 80)
    print("TWO-POPULATION COMPETITIVE COEVOLUTION")
    print("=" * 80)
    print(f"\nPopulation A: {character_a} (μ={pop_size}, λ={offspring_count})")
    print(f"Population B: {character_b} (μ={pop_size}, λ={offspring_count})")
    print(f"Generations: {generations}")
    print(f"K-fold matches: {k_matches}")
    print(f"Tournament size: {tournament_size}")
    print("=" * 80)

    # Initialize populations
    print("\nInitializing populations...")
    population_a = create_random_population(pop_size)
    population_b = create_random_population(pop_size)

    # Create simulator
    simulator = FightSimulator(headless=True)

    # Track statistics (enhanced for analysis)
    stats = {
        'pop_size': pop_size,
        'generations_total': generations,
        'k_matches': k_matches,
        'tournament_size': tournament_size,
        'character1': character_a,
        'character2': character_b,
        'history_a': [],
        'history_b': [],
        'best_individual_a': None,
        'best_individual_b': None,
        'best_fitness_a': float('-inf'),
        'best_fitness_b': float('-inf'),
        'total_time': 0
    }

    print(f"\n{'='*80}")
    print("Starting Coevolution...")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Main coevolution loop
    for gen in range(generations):
        gen_start = time.time()

        print(f"\n{'='*80}")
        print(f"Generation {gen+1}/{generations}")
        print(f"{'='*80}")

        # Evaluate Population A against Population B
        print(f"\nEvaluating {character_a} (Pop A) vs {character_b} (Pop B)...")
        fitness_a = evaluate_population(
            population_a,
            population_b,
            simulator,
            character=character_a,
            opponent_character=character_b,
            k_matches=k_matches
        )

        # Evaluate Population B against Population A
        print(f"\nEvaluating {character_b} (Pop B) vs {character_a} (Pop A)...")
        fitness_b = evaluate_population(
            population_b,
            population_a,
            simulator,
            character=character_b,
            opponent_character=character_a,
            k_matches=k_matches
        )

        # Calculate statistics
        avg_fitness_a = sum(fitness_a) / len(fitness_a)
        max_fitness_a = max(fitness_a)
        best_idx_a = fitness_a.index(max_fitness_a)
        best_individual_a = population_a[best_idx_a]

        avg_fitness_b = sum(fitness_b) / len(fitness_b)
        max_fitness_b = max(fitness_b)
        best_idx_b = fitness_b.index(max_fitness_b)
        best_individual_b = population_b[best_idx_b]

        # Track best overall
        if max_fitness_a > stats['best_fitness_a']:
            stats['best_fitness_a'] = max_fitness_a
            stats['best_individual_a'] = {
                'strategy': best_individual_a.strategy,
                'thresholds': best_individual_a.thresholds,
                'weights': best_individual_a.weights,
                'fitness': max_fitness_a,
                'wins': best_individual_a.wins,
                'losses': best_individual_a.losses,
                'draws': best_individual_a.draws
            }

        if max_fitness_b > stats['best_fitness_b']:
            stats['best_fitness_b'] = max_fitness_b
            stats['best_individual_b'] = {
                'strategy': best_individual_b.strategy,
                'thresholds': best_individual_b.thresholds,
                'weights': best_individual_b.weights,
                'fitness': max_fitness_b,
                'wins': best_individual_b.wins,
                'losses': best_individual_b.losses,
                'draws': best_individual_b.draws
            }

        # Count strategies for diversity metrics
        strategies_a = {}
        for ind in population_a:
            strategies_a[ind.strategy] = strategies_a.get(ind.strategy, 0) + 1

        strategies_b = {}
        for ind in population_b:
            strategies_b[ind.strategy] = strategies_b.get(ind.strategy, 0) + 1

        # Calculate win rates for entire population
        total_wins_a = sum(ind.wins for ind in population_a)
        total_losses_a = sum(ind.losses for ind in population_a)
        total_draws_a = sum(ind.draws for ind in population_a)
        total_matches_a = sum(ind.matches_played for ind in population_a)
        win_rate_a = total_wins_a / total_matches_a if total_matches_a > 0 else 0

        total_wins_b = sum(ind.wins for ind in population_b)
        total_losses_b = sum(ind.losses for ind in population_b)
        total_draws_b = sum(ind.draws for ind in population_b)
        total_matches_b = sum(ind.matches_played for ind in population_b)
        win_rate_b = total_wins_b / total_matches_b if total_matches_b > 0 else 0

        # Store per-generation statistics
        stats['history_a'].append({
            'generation': gen + 1,
            'avg_fitness': avg_fitness_a,
            'max_fitness': max_fitness_a,
            'min_fitness': min(fitness_a),
            'strategies': strategies_a,
            'win_rate': win_rate_a,
            'total_wins': total_wins_a,
            'total_losses': total_losses_a,
            'total_draws': total_draws_a,
            'total_matches': total_matches_a
        })

        stats['history_b'].append({
            'generation': gen + 1,
            'avg_fitness': avg_fitness_b,
            'max_fitness': max_fitness_b,
            'min_fitness': min(fitness_b),
            'strategies': strategies_b,
            'win_rate': win_rate_b,
            'total_wins': total_wins_b,
            'total_losses': total_losses_b,
            'total_draws': total_draws_b,
            'total_matches': total_matches_b
        })

        # Print generation summary
        gen_time = time.time() - gen_start
        print(f"\n{'-'*80}")
        print(f"Generation {gen+1} Results:")
        print(f"{'-'*80}")
        print(f"Population A ({character_a}):")
        print(f"  Avg Fitness: {avg_fitness_a:.2f}")
        print(f"  Max Fitness: {max_fitness_a:.2f}")
        print(f"  Win Rate: {win_rate_a:.1%} (W:{total_wins_a} L:{total_losses_a} D:{total_draws_a})")
        print(f"  Best: {best_individual_a}")
        print(f"\nPopulation B ({character_b}):")
        print(f"  Avg Fitness: {avg_fitness_b:.2f}")
        print(f"  Max Fitness: {max_fitness_b:.2f}")
        print(f"  Win Rate: {win_rate_b:.1%} (W:{total_wins_b} L:{total_losses_b} D:{total_draws_b})")
        print(f"  Best: {best_individual_b}")
        print(f"\nTime: {gen_time:.1f}s")
        print(f"{'-'*80}")

        # Evolve both populations
        if gen < generations - 1:
            print(f"\nEvolving populations...")

            population_a = evolve_population(
                population_a,
                fitness_a,
                offspring_count=offspring_count,
                elite_count=2,
                tournament_size=tournament_size,
                crossover_rate=0.7,
                mutation_rate=0.1
            )

            population_b = evolve_population(
                population_b,
                fitness_b,
                offspring_count=offspring_count,
                elite_count=2,
                tournament_size=tournament_size,
                crossover_rate=0.7,
                mutation_rate=0.1
            )

    total_time = time.time() - start_time
    stats['total_time'] = total_time

    # Cleanup
    simulator.cleanup()

    # Save results to JSON
    results_file = f'results_coevolution_{character_a}vs{character_b}_{generations}gen.json'
    with open(results_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Final summary
    print(f"\n{'='*80}")
    print("COEVOLUTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Avg time per generation: {total_time/generations:.1f}s")

    hist_a = stats['history_a']
    hist_b = stats['history_b']

    print(f"\n{'-'*80}")
    print(f"Population A ({character_a}) Evolution:")
    print(f"{'-'*80}")
    print(f"  Initial: Avg={hist_a[0]['avg_fitness']:.2f}, Max={hist_a[0]['max_fitness']:.2f}")
    print(f"  Final:   Avg={hist_a[-1]['avg_fitness']:.2f}, Max={hist_a[-1]['max_fitness']:.2f}")
    print(f"  Change:  Avg={hist_a[-1]['avg_fitness'] - hist_a[0]['avg_fitness']:+.2f}, "
          f"Max={hist_a[-1]['max_fitness'] - hist_a[0]['max_fitness']:+.2f}")
    print(f"  Best Overall: {stats['best_fitness_a']:.2f}")

    print(f"\n{'-'*80}")
    print(f"Population B ({character_b}) Evolution:")
    print(f"{'-'*80}")
    print(f"  Initial: Avg={hist_b[0]['avg_fitness']:.2f}, Max={hist_b[0]['max_fitness']:.2f}")
    print(f"  Final:   Avg={hist_b[-1]['avg_fitness']:.2f}, Max={hist_b[-1]['max_fitness']:.2f}")
    print(f"  Change:  Avg={hist_b[-1]['avg_fitness'] - hist_b[0]['avg_fitness']:+.2f}, "
          f"Max={hist_b[-1]['max_fitness'] - hist_b[0]['max_fitness']:+.2f}")
    print(f"  Best Overall: {stats['best_fitness_b']:.2f}")

    # Check for arms race (both improving)
    a_improved = hist_a[-1]['avg_fitness'] > hist_a[0]['avg_fitness']
    b_improved = hist_b[-1]['avg_fitness'] > hist_b[0]['avg_fitness']

    print(f"\n{'='*80}")
    if a_improved and b_improved:
        print("Arms race detected! Both populations improved.")
    elif a_improved or b_improved:
        improved = character_a if a_improved else character_b
        print(f"{improved} improved significantly.")
    else:
        print("Minimal improvement in both populations.")
    print(f"{'='*80}\n")

    return stats


def plot_coevolution(stats):
    """
    Print ASCII plot of both populations' fitness evolution.

    Args:
        stats: Statistics dictionary from experiment
    """
    print("\nCoevolution Progress (ASCII Plot):")
    print("-" * 80)

    hist_a = stats['history_a']
    hist_b = stats['history_b']

    pop_a_avg = [h['avg_fitness'] for h in hist_a]
    pop_a_max = [h['max_fitness'] for h in hist_a]
    pop_b_avg = [h['avg_fitness'] for h in hist_b]
    pop_b_max = [h['max_fitness'] for h in hist_b]

    # Combine all values for scaling
    all_values = pop_a_avg + pop_a_max + pop_b_avg + pop_b_max
    min_val = min(all_values)
    max_val = max(all_values)
    value_range = max_val - min_val if max_val > min_val else 1

    plot_height = 15
    plot_width = len(hist_a)

    # Create plot
    for row in range(plot_height, -1, -1):
        line = ""
        for col in range(plot_width):
            a_avg = int((pop_a_avg[col] - min_val) / value_range * plot_height)
            a_max = int((pop_a_max[col] - min_val) / value_range * plot_height)
            b_avg = int((pop_b_avg[col] - min_val) / value_range * plot_height)
            b_max = int((pop_b_max[col] - min_val) / value_range * plot_height)

            if row == a_max:
                line += "▲"  # Pop A max
            elif row == a_avg:
                line += "●"  # Pop A avg
            elif row == b_max:
                line += "△"  # Pop B max
            elif row == b_avg:
                line += "○"  # Pop B avg
            elif row == 0:
                line += "─"
            else:
                line += " "

        # Y-axis label
        val = min_val + (row / plot_height) * value_range
        print(f"{val:6.1f} │ {line}")

    # X-axis
    print("       └" + "─" * plot_width)
    x_labels = "".join(str((i+1) % 10) for i in range(plot_width))
    print(f"        {x_labels}")
    print(f"\n  Legend:")
    print(f"    Population A: ● = Avg, ▲ = Max")
    print(f"    Population B: ○ = Avg, △ = Max")
    print("-" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run two-population competitive coevolution')
    parser.add_argument('--pop-size', type=int, default=10,
                       help='Population size (default: 10)')
    parser.add_argument('--generations', type=int, default=5,
                       help='Number of generations (default: 5)')
    parser.add_argument('--offspring', type=int, default=15,
                       help='Offspring count (default: 15)')
    parser.add_argument('--k-matches', type=int, default=2,
                       help='K-fold matches (default: 2)')
    parser.add_argument('--tournament-size', type=int, default=3,
                       help='Tournament size for selection (default: 3)')
    parser.add_argument('--char-a', type=str, default='Ken',
                       help='Character for population A (default: Ken)')
    parser.add_argument('--char-b', type=str, default='Rick',
                       help='Character for population B (default: Rick)')

    args = parser.parse_args()

    # Run experiment
    stats = run_coevolution_experiment(
        pop_size=args.pop_size,
        generations=args.generations,
        offspring_count=args.offspring,
        k_matches=args.k_matches,
        tournament_size=args.tournament_size,
        character_a=args.char_a,
        character_b=args.char_b
    )

    # Plot results
    plot_coevolution(stats)

    print("Coevolution experiment complete!")
