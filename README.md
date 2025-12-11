# Competitive Coevolution for Fighting Game AI

Two-population competitive coevolution using genetic algorithms. AI controllers for Street Fighter characters evolve against each other in an arms race.

Built on [Street_pyghter](https://github.com/Freddy875/Street_pyghter) game engine. In order to run the code Street_pyghter must be downloaded.

## Example

```bash
# Run coevolution
cd experiments
python coevolution.py --pop-size 20 --generations 20 --k-matches 3

# Analyze results
python analysis.py results_coevolution_KenvsZangief_20gen.json --watch-battle
```

## Structure

- `src/` - Core implementation (genotype, evolution, controller, evaluation)
- `experiments/` - Experiment runners and analysis tools
