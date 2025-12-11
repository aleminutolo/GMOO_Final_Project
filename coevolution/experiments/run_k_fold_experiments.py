import sys
import os
import time
import json
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from coevolution import run_coevolution_experiment


def run_k_fold_sweep(
    k_values=[3, 4, 5],
    pop_size=20,
    generations=20,
    offspring_count=30,
    tournament_size=3,
    character_a='Ken',
    character_b='Zangief'
):
    """
    Run coevolution experiments for different k-fold values.

    Args:
        k_values: List of k-fold values to test
        pop_size: Population size for each experiment
        generations: Number of generations
        offspring_count: Offspring per generation
        tournament_size: Tournament size for selection
        character_a: Character for population A
        character_b: Character for population B

    Returns:
        Dictionary mapping k values to their results
    """
    print("=" * 80)
    print("K-FOLD PARAMETER SWEEP")
    print("=" * 80)
    print(f"\nTesting k values: {k_values}")
    print(f"Population size: {pop_size}")
    print(f"Generations: {generations}")
    print(f"Tournament size: {tournament_size}")
    print(f"Characters: {character_a} vs {character_b}")
    print("=" * 80)

    all_results = {}

    for k in k_values:
        print(f"\n\n{'='*80}")
        print(f"RUNNING EXPERIMENT: k={k}")
        print(f"{'='*80}\n")

        experiment_start = time.time()

        # Run experiment
        stats = run_coevolution_experiment(
            pop_size=pop_size,
            generations=generations,
            offspring_count=offspring_count,
            k_matches=k,
            tournament_size=tournament_size,
            character_a=character_a,
            character_b=character_b
        )

        experiment_time = time.time() - experiment_start

        # Store results
        all_results[k] = stats

        # Save individual results
        results_file = f'results_k_fold_sweep_k{k}_{character_a}vs{character_b}.json'
        with open(results_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n✓ Experiment k={k} complete in {experiment_time:.1f}s")
        print(f"  Saved to: {results_file}")

    # Save combined results
    combined_file = f'results_k_fold_sweep_combined_{character_a}vs{character_b}.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"\n✓ Combined results saved to: {combined_file}")

    return all_results


def plot_k_fold_comparison(all_results, save_file='k_fold_comparison.png'):
    """
    Create comparison plots for different k values.

    Args:
        all_results: Dictionary mapping k values to experiment results
        save_file: Filename for saving the plot
    """
    print(f"\nGenerating comparison plots...")

    k_values = sorted(all_results.keys())

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('K-Fold Parameter Sweep Results', fontsize=16, fontweight='bold')

    # Plot 1: Final fitness vs k
    ax1 = axes[0, 0]
    final_fitness_a = []
    final_fitness_b = []

    for k in k_values:
        stats = all_results[k]
        final_fitness_a.append(stats['history_a'][-1]['max_fitness'])
        final_fitness_b.append(stats['history_b'][-1]['max_fitness'])

    ax1.plot(k_values, final_fitness_a, 'o-', label=f"Pop A ({all_results[k_values[0]]['character1']})",
             linewidth=2, markersize=8)
    ax1.plot(k_values, final_fitness_b, 's-', label=f"Pop B ({all_results[k_values[0]]['character2']})",
             linewidth=2, markersize=8)
    ax1.set_xlabel('K-Fold Value', fontsize=12)
    ax1.set_ylabel('Final Max Fitness', fontsize=12)
    ax1.set_title('Final Fitness vs K-Fold', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Win rate vs k
    ax2 = axes[0, 1]
    final_win_rate_a = []
    final_win_rate_b = []

    for k in k_values:
        stats = all_results[k]
        final_win_rate_a.append(stats['history_a'][-1]['win_rate'] * 100)
        final_win_rate_b.append(stats['history_b'][-1]['win_rate'] * 100)

    ax2.plot(k_values, final_win_rate_a, 'o-', label=f"Pop A ({all_results[k_values[0]]['character1']})",
             linewidth=2, markersize=8)
    ax2.plot(k_values, final_win_rate_b, 's-', label=f"Pop B ({all_results[k_values[0]]['character2']})",
             linewidth=2, markersize=8)
    ax2.set_xlabel('K-Fold Value', fontsize=12)
    ax2.set_ylabel('Win Rate (%)', fontsize=12)
    ax2.set_title('Final Win Rate vs K-Fold', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    # Plot 3: Fitness improvement vs k
    ax3 = axes[1, 0]
    improvement_a = []
    improvement_b = []

    for k in k_values:
        stats = all_results[k]
        init_a = stats['history_a'][0]['max_fitness']
        final_a = stats['history_a'][-1]['max_fitness']
        init_b = stats['history_b'][0]['max_fitness']
        final_b = stats['history_b'][-1]['max_fitness']

        improvement_a.append(final_a - init_a)
        improvement_b.append(final_b - init_b)

    ax3.bar([k - 0.2 for k in k_values], improvement_a, width=0.4,
            label=f"Pop A ({all_results[k_values[0]]['character1']})", alpha=0.8)
    ax3.bar([k + 0.2 for k in k_values], improvement_b, width=0.4,
            label=f"Pop B ({all_results[k_values[0]]['character2']})", alpha=0.8)
    ax3.set_xlabel('K-Fold Value', fontsize=12)
    ax3.set_ylabel('Fitness Improvement', fontsize=12)
    ax3.set_title('Fitness Improvement vs K-Fold', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Plot 4: Evolution curves for all k values
    ax4 = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))

    for i, k in enumerate(k_values):
        stats = all_results[k]
        generations = [h['generation'] for h in stats['history_a']]
        fitness_a = [h['max_fitness'] for h in stats['history_a']]

        ax4.plot(generations, fitness_a, '-', color=colors[i],
                label=f'k={k}', linewidth=2, alpha=0.7)

    ax4.set_xlabel('Generation', fontsize=12)
    ax4.set_ylabel('Max Fitness (Pop A)', fontsize=12)
    ax4.set_title(f'Evolution Curves - Pop A ({all_results[k_values[0]]["character1"]})',
                  fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to: {save_file}")

    # Also create a summary table
    print("\n" + "="*80)
    print("K-FOLD COMPARISON SUMMARY")
    print("="*80)
    print(f"{'K':>3} | {'Pop A Final':>12} | {'Pop B Final':>12} | {'A WinRate':>10} | {'B WinRate':>10} | {'Time (s)':>8}")
    print("-"*80)

    for k in k_values:
        stats = all_results[k]
        final_a = stats['history_a'][-1]['max_fitness']
        final_b = stats['history_b'][-1]['max_fitness']
        wr_a = stats['history_a'][-1]['win_rate'] * 100
        wr_b = stats['history_b'][-1]['win_rate'] * 100
        total_time = stats['total_time']

        print(f"{k:3d} | {final_a:12.2f} | {final_b:12.2f} | {wr_a:9.1f}% | {wr_b:9.1f}% | {total_time:8.1f}")

    print("="*80)


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description='Run k-fold parameter sweep')
    parser.add_argument('--k-min', type=int, default=3,
                       help='Minimum k value (default: 3)')
    parser.add_argument('--k-max', type=int, default=8,
                       help='Maximum k value (default: 8)')
    parser.add_argument('--pop-size', type=int, default=20,
                       help='Population size (default: 20)')
    parser.add_argument('--generations', type=int, default=20,
                       help='Number of generations (default: 20)')
    parser.add_argument('--offspring', type=int, default=30,
                       help='Offspring count (default: 30)')
    parser.add_argument('--tournament-size', type=int, default=3,
                       help='Tournament size (default: 3)')
    parser.add_argument('--char-a', type=str, default='Ken',
                       help='Character for population A (default: Ken)')
    parser.add_argument('--char-b', type=str, default='Zangief',
                       help='Character for population B (default: Zangief)')

    args = parser.parse_args()

    # Generate k values
    k_values = list(range(args.k_min, args.k_max + 1))

    # Run experiments
    all_results = run_k_fold_sweep(
        k_values=k_values,
        pop_size=args.pop_size,
        generations=args.generations,
        offspring_count=args.offspring,
        tournament_size=args.tournament_size,
        character_a=args.char_a,
        character_b=args.char_b
    )

    # Generate comparison plots
    plot_k_fold_comparison(all_results)

    print("\n✓ K-fold parameter sweep complete!")
