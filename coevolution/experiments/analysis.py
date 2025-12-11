import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import List, Dict, Tuple


class CoevolutionAnalyzer:
    """
    Analyze coevolution experiments based on course metrics.

    Key Metrics (from Lecture 8):
    1. Fitness Evolution - Track avg/max/min fitness over generations
    2. Arms Race Detection - Both populations improving = successful coevolution
    3. Diversity Maintenance - Ensure populations don't lose diversity
    4. Win Rate Tracking - Competition balance (50% = balanced, >70% = domination)
    5. Convergence - Detect when evolution plateaus
    6. Baseline Performance - Measure absolute skill, not just relative
    """

    def __init__(self, results_file: str):
        """Load coevolution results from JSON file."""
        with open(results_file, 'r') as f:
            self.data = json.load(f)

        self.generations = len(self.data['history_a'])

    def plot_fitness_curves(self, save_path: str = None):
        """
        Plot fitness evolution for both populations.

        Shows:
        - Average fitness (main indicator)
        - Max fitness (best individual)
        - Min fitness (worst individual)

        Arms race: Both populations should show upward trends.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        gens = range(self.generations)

        # Population A
        hist_a = self.data['history_a']
        avg_a = [h['avg_fitness'] for h in hist_a]
        max_a = [h['max_fitness'] for h in hist_a]
        min_a = [h['min_fitness'] for h in hist_a]

        ax1.plot(gens, avg_a, 'b-', linewidth=2, label='Average')
        ax1.plot(gens, max_a, 'g--', linewidth=1.5, label='Max')
        ax1.plot(gens, min_a, 'r:', linewidth=1.5, label='Min')
        ax1.fill_between(gens, min_a, max_a, alpha=0.2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title(f'Population A ({self.data["character1"]}) Fitness Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Population B
        hist_b = self.data['history_b']
        avg_b = [h['avg_fitness'] for h in hist_b]
        max_b = [h['max_fitness'] for h in hist_b]
        min_b = [h['min_fitness'] for h in hist_b]

        ax2.plot(gens, avg_b, 'b-', linewidth=2, label='Average')
        ax2.plot(gens, max_b, 'g--', linewidth=1.5, label='Max')
        ax2.plot(gens, min_b, 'r:', linewidth=1.5, label='Min')
        ax2.fill_between(gens, min_b, max_b, alpha=0.2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness')
        ax2.set_title(f'Population B ({self.data["character2"]}) Fitness Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved fitness curves to {save_path}")
        else:
            plt.show()

        return fig

    def plot_arms_race(self, save_path: str = None):
        """
        Visualize the arms race between populations.

        Both populations improving = successful competitive coevolution
        One improving, one declining = dominance (not ideal)
        Both flat = stagnation/cycling
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        gens = range(self.generations)

        hist_a = self.data['history_a']
        hist_b = self.data['history_b']

        avg_a = [h['avg_fitness'] for h in hist_a]
        avg_b = [h['avg_fitness'] for h in hist_b]

        ax.plot(gens, avg_a, 'b-', linewidth=2.5, marker='o', markersize=6,
                label=f'Pop A ({self.data["character1"]})')
        ax.plot(gens, avg_b, 'r-', linewidth=2.5, marker='s', markersize=6,
                label=f'Pop B ({self.data["character2"]})')

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Average Fitness', fontsize=12)
        ax.set_title('Arms Race: Competitive Coevolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Detect arms race
        improvement_a = avg_a[-1] - avg_a[0]
        improvement_b = avg_b[-1] - avg_b[0]

        if improvement_a > 0 and improvement_b > 0:
            status = "✓ ARMS RACE DETECTED"
            color = 'green'
        elif improvement_a > 0 and improvement_b < 0:
            status = "⚠ Population A Dominating"
            color = 'orange'
        elif improvement_a < 0 and improvement_b > 0:
            status = "⚠ Population B Dominating"
            color = 'orange'
        else:
            status = "✗ Stagnation/Cycling"
            color = 'red'

        ax.text(0.5, 0.95, status, transform=ax.transAxes,
                fontsize=12, fontweight='bold', color=color,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved arms race plot to {save_path}")
        else:
            plt.show()

        return fig

    def analyze_diversity(self) -> Dict[str, any]:
        """
        Analyze genetic diversity in populations.

        From course: Loss of diversity is a major problem in coevolution.

        Metrics:
        - Weight variance (genotype diversity)
        - Strategy distribution (phenotype diversity)
        - Unique individuals
        """
        diversity_a = []
        diversity_b = []

        for gen in self.data['history_a']:
            # Strategy distribution
            strategies = gen.get('strategies', {})
            # Shannon entropy for diversity
            total = sum(strategies.values())
            if total > 0:
                entropy = -sum((count/total) * np.log2(count/total + 1e-10)
                              for count in strategies.values() if count > 0)
            else:
                entropy = 0
            diversity_a.append(entropy)

        for gen in self.data['history_b']:
            strategies = gen.get('strategies', {})
            total = sum(strategies.values())
            if total > 0:
                entropy = -sum((count/total) * np.log2(count/total + 1e-10)
                              for count in strategies.values() if count > 0)
            else:
                entropy = 0
            diversity_b.append(entropy)

        return {
            'diversity_a': diversity_a,
            'diversity_b': diversity_b,
            'avg_diversity_a': np.mean(diversity_a),
            'avg_diversity_b': np.mean(diversity_b),
            'final_diversity_a': diversity_a[-1],
            'final_diversity_b': diversity_b[-1]
        }

    def plot_diversity(self, save_path: str = None):
        """Plot diversity metrics over time."""
        diversity = self.analyze_diversity()

        fig, ax = plt.subplots(figsize=(10, 6))

        gens = range(self.generations)
        ax.plot(gens, diversity['diversity_a'], 'b-', linewidth=2,
                marker='o', label='Pop A')
        ax.plot(gens, diversity['diversity_b'], 'r-', linewidth=2,
                marker='s', label='Pop B')

        ax.set_xlabel('Generation')
        ax.set_ylabel('Diversity (Shannon Entropy)')
        ax.set_title('Population Diversity Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Warning if diversity drops too low
        if diversity['final_diversity_a'] < 0.5 or diversity['final_diversity_b'] < 0.5:
            ax.text(0.5, 0.05, '⚠ Low diversity detected!',
                   transform=ax.transAxes, fontsize=10, color='red',
                   ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved diversity plot to {save_path}")
        else:
            plt.show()

        return fig

    def plot_win_rates(self, save_path: str = None):
        """
        Plot win rates over generations for both populations.

        Shows:
        - Win rate percentage for each population
        - Balanced competition = both populations around 50%
        - One population dominating = win rate significantly different from 50%

        This metric complements fitness by showing actual match outcomes.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        gens = range(self.generations)
        hist_a = self.data['history_a']
        hist_b = self.data['history_b']

        # Extract win rates (convert to percentage)
        win_rates_a = [h.get('win_rate', 0) * 100 for h in hist_a]
        win_rates_b = [h.get('win_rate', 0) * 100 for h in hist_b]

        # Plot win rates
        ax.plot(gens, win_rates_a, 'b-', linewidth=2.5, marker='o', markersize=6,
                label=f'Pop A ({self.data["character1"]})')
        ax.plot(gens, win_rates_b, 'r-', linewidth=2.5, marker='s', markersize=6,
                label=f'Pop B ({self.data["character2"]})')

        # Add 50% reference line (balanced competition)
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='Balanced (50%)')

        # Shade the "balanced region" (40-60%)
        ax.axhspan(40, 60, alpha=0.1, color='green', label='Balanced region')

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title('Win Rates Over Generations', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Analyze balance
        final_wr_a = win_rates_a[-1]
        final_wr_b = win_rates_b[-1]
        avg_wr_a = np.mean(win_rates_a)
        avg_wr_b = np.mean(win_rates_b)

        # Determine competition balance
        if 40 <= final_wr_a <= 60 and 40 <= final_wr_b <= 60:
            status = "✓ BALANCED COMPETITION"
            color = 'green'
        elif final_wr_a > 70:
            status = f"⚠ Pop A Dominating ({final_wr_a:.1f}%)"
            color = 'orange'
        elif final_wr_b > 70:
            status = f"⚠ Pop B Dominating ({final_wr_b:.1f}%)"
            color = 'orange'
        else:
            status = "Moderate Imbalance"
            color = 'blue'

        # Add status text
        ax.text(0.5, 0.95, status, transform=ax.transAxes,
                fontsize=12, fontweight='bold', color=color,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add average win rate annotations
        ax.text(0.02, 0.98, f'Avg: {avg_wr_a:.1f}%', transform=ax.transAxes,
                fontsize=9, color='blue', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.text(0.02, 0.92, f'Avg: {avg_wr_b:.1f}%', transform=ax.transAxes,
                fontsize=9, color='red', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved win rate plot to {save_path}")
        else:
            plt.show()

        return fig

    def detect_convergence(self, window: int = 5, threshold: float = 5.0) -> Dict:
        """
        Detect when evolution has converged (plateaued).

        Convergence = fitness change over window < threshold
        Helps determine optimal number of generations.
        """
        hist_a = self.data['history_a']
        hist_b = self.data['history_b']

        avg_a = [h['avg_fitness'] for h in hist_a]
        avg_b = [h['avg_fitness'] for h in hist_b]

        def find_convergence(fitness_history):
            for i in range(window, len(fitness_history)):
                window_data = fitness_history[i-window:i]
                change = max(window_data) - min(window_data)
                if change < threshold:
                    return i - window
            return None

        conv_a = find_convergence(avg_a)
        conv_b = find_convergence(avg_b)

        return {
            'converged_a': conv_a is not None,
            'convergence_gen_a': conv_a,
            'converged_b': conv_b is not None,
            'convergence_gen_b': conv_b,
            'both_converged': conv_a is not None and conv_b is not None,
            'convergence_gen': max(conv_a or 0, conv_b or 0)
        }

    def generate_report(self, output_dir: str = 'analysis_results'):
        """
        Generate comprehensive analysis report with all plots and metrics.
        """
        os.makedirs(output_dir, exist_ok=True)

        print("="*70)
        print("COEVOLUTION ANALYSIS REPORT")
        print("="*70)

        # 1. Basic statistics
        print("\n1. EXPERIMENT OVERVIEW")
        print("-" * 70)
        print(f"Population size: {self.data['pop_size']}")
        print(f"Generations: {self.generations}")
        print(f"K-matches: {self.data['k_matches']}")
        print(f"Characters: {self.data['character1']} vs {self.data['character2']}")
        print(f"Total time: {self.data['total_time']:.1f}s")

        # 2. Fitness evolution
        print("\n2. FITNESS EVOLUTION")
        print("-" * 70)
        hist_a = self.data['history_a']
        hist_b = self.data['history_b']

        print(f"\nPopulation A ({self.data['character1']}):")
        print(f"  Initial: Avg={hist_a[0]['avg_fitness']:.2f}, Max={hist_a[0]['max_fitness']:.2f}")
        print(f"  Final:   Avg={hist_a[-1]['avg_fitness']:.2f}, Max={hist_a[-1]['max_fitness']:.2f}")
        print(f"  Change:  Avg={hist_a[-1]['avg_fitness']-hist_a[0]['avg_fitness']:+.2f}")

        print(f"\nPopulation B ({self.data['character2']}):")
        print(f"  Initial: Avg={hist_b[0]['avg_fitness']:.2f}, Max={hist_b[0]['max_fitness']:.2f}")
        print(f"  Final:   Avg={hist_b[-1]['avg_fitness']:.2f}, Max={hist_b[-1]['max_fitness']:.2f}")
        print(f"  Change:  Avg={hist_b[-1]['avg_fitness']-hist_b[0]['avg_fitness']:+.2f}")

        # 3. Arms race detection
        print("\n3. ARMS RACE ANALYSIS")
        print("-" * 70)
        improvement_a = hist_a[-1]['avg_fitness'] - hist_a[0]['avg_fitness']
        improvement_b = hist_b[-1]['avg_fitness'] - hist_b[0]['avg_fitness']

        if improvement_a > 0 and improvement_b > 0:
            print("✓ ARMS RACE DETECTED: Both populations improving")
            print("  This indicates successful competitive coevolution.")
        else:
            print("⚠ No clear arms race detected")

        # 4. Diversity analysis
        print("\n4. DIVERSITY ANALYSIS")
        print("-" * 70)
        diversity = self.analyze_diversity()
        print(f"Population A diversity: {diversity['final_diversity_a']:.3f}")
        print(f"Population B diversity: {diversity['final_diversity_b']:.3f}")

        if diversity['final_diversity_a'] < 0.5:
            print("⚠ Warning: Population A has low diversity")
        if diversity['final_diversity_b'] < 0.5:
            print("⚠ Warning: Population B has low diversity")

        # 5. Convergence analysis
        print("\n5. CONVERGENCE ANALYSIS")
        print("-" * 70)
        conv = self.detect_convergence()
        if conv['both_converged']:
            print(f"✓ Both populations converged around generation {conv['convergence_gen']}")
            print(f"  Recommendation: Use ~{conv['convergence_gen']+5} generations for this setup")
        else:
            print("⚠ Populations have not fully converged")
            print("  Recommendation: Run more generations")

        # 6. Win rate analysis (if available)
        if 'win_rate' in hist_a[0]:
            print("\n6. WIN RATE ANALYSIS")
            print("-" * 70)
            final_wr_a = hist_a[-1]['win_rate'] * 100
            final_wr_b = hist_b[-1]['win_rate'] * 100
            avg_wr_a = np.mean([h['win_rate'] for h in hist_a]) * 100
            avg_wr_b = np.mean([h['win_rate'] for h in hist_b]) * 100

            print(f"Population A ({self.data['character1']}):")
            print(f"  Average win rate: {avg_wr_a:.1f}%")
            print(f"  Final win rate: {final_wr_a:.1f}%")

            print(f"\nPopulation B ({self.data['character2']}):")
            print(f"  Average win rate: {avg_wr_b:.1f}%")
            print(f"  Final win rate: {final_wr_b:.1f}%")

            if 40 <= final_wr_a <= 60 and 40 <= final_wr_b <= 60:
                print("\n✓ Balanced competition achieved (both populations ~50% win rate)")
            elif final_wr_a > 70:
                print(f"\n⚠ Population A dominates with {final_wr_a:.1f}% win rate")
            elif final_wr_b > 70:
                print(f"\n⚠ Population B dominates with {final_wr_b:.1f}% win rate")

        # 7. Generate plots
        print("\n7. GENERATING PLOTS")
        print("-" * 70)
        self.plot_fitness_curves(os.path.join(output_dir, 'fitness_curves.png'))
        self.plot_arms_race(os.path.join(output_dir, 'arms_race.png'))
        self.plot_diversity(os.path.join(output_dir, 'diversity.png'))

        # Generate win rate plot if data is available
        if 'win_rate' in hist_a[0]:
            self.plot_win_rates(os.path.join(output_dir, 'win_rates.png'))

        print("\n" + "="*70)
        print(f"✓ Analysis complete! Results saved to {output_dir}/")
        print("="*70)


def visualize_best_battle(results_file: str, headless: bool = False):
    """
    Visualize a fight between the best individuals from each population.

    Args:
        results_file: Path to coevolution results JSON
        headless: If False, show the fight visually
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

    from individual import Individual
    from evaluation import FightSimulator
    import json

    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Reconstruct best individuals
    best_a_data = data['best_individual_a']
    best_b_data = data['best_individual_b']

    best_a = Individual(
        strategy=best_a_data['strategy'],
        thresholds=best_a_data['thresholds'],
        weights=best_a_data['weights']
    )

    best_b = Individual(
        strategy=best_b_data['strategy'],
        thresholds=best_b_data['thresholds'],
        weights=best_b_data['weights']
    )

    print("="*70)
    print("BATTLE: BEST vs BEST")
    print("="*70)
    print(f"\nChampion A ({data['character1']}): {best_a}")
    print(f"Champion B ({data['character2']}): {best_b}")
    print(f"\nFinal fitness: A={best_a_data['fitness']:.2f}, B={best_b_data['fitness']:.2f}")
    print("\nStarting fight...\n")

    # Run fight
    simulator = FightSimulator(headless=headless)
    result = simulator.run_fight(
        best_a, best_b,
        character1=data['character1'],
        character2=data['character2'],
        max_time=99,  # Full match
        render=not headless
    )

    print("="*70)
    print("FIGHT RESULT:")
    print("="*70)
    print(f"Winner: {'Champion A' if result['winner']==1 else 'Champion B' if result['winner']==2 else 'DRAW'}")
    print(f"{data['character1']}: {result['p1_health']} HP, {result['p1_damage']} damage dealt")
    print(f"{data['character2']}: {result['p2_health']} HP, {result['p2_damage']} damage dealt")
    print(f"Average distance: {result['avg_distance']:.1f} pixels")
    print("="*70)

    simulator.cleanup()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze coevolution results')
    parser.add_argument('results_file', help='Path to results JSON file')
    parser.add_argument('--output-dir', default='analysis_results',
                       help='Directory for output plots')
    parser.add_argument('--watch-battle', action='store_true',
                       help='Watch a fight between best individuals')

    args = parser.parse_args()

    # Run analysis
    analyzer = CoevolutionAnalyzer(args.results_file)
    analyzer.generate_report(args.output_dir)

    # Optionally watch best battle
    if args.watch_battle:
        print("\n")
        visualize_best_battle(args.results_file, headless=False)
