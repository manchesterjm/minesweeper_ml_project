# Minesweeper ML

AI agents for solving Minesweeper using **constraint satisfaction (AC-3)** and **reinforcement learning (DQN)**.

## Results

| Agent | Win Rate | Strategy |
|-------|----------|----------|
| **Hybrid (trained)** | **88%** | AC-3 logic + neural network for guessing |
| Logic (AC-3) | 81% | Pure constraint propagation |
| DQN (trained) | 1% | Deep Q-Network (20k episodes) |
| Random | 0% | Baseline |

The **Hybrid agent** achieves near-theoretical maximum (~85-90%) for beginner Minesweeper.

## Key Insight

Pure DQN struggles with Minesweeper because:
- Huge state space (2^81 possible boards)
- Sparse rewards (win/lose at end)
- Requires 500k+ episodes to learn basic patterns

The **Hybrid approach** works better:
- **AC-3 constraint propagation** handles ~93% of moves (definite deductions)
- **Neural network** handles ~7% of moves (forced guesses)
- Network learns patterns for when to guess, achieving 86% guess accuracy

## Agents

### LogicAgent (AC-3)
- Builds constraints from revealed numbers
- Iteratively propagates to find definite safe/mine cells
- Applies subset reduction for advanced deductions
- Falls back to probability estimation when guessing

### HybridAgent
- Uses LogicAgent's AC-3 for certain moves
- Trains a neural network specifically for "forced guess" situations
- Network learns which cells are safer when logic can't help

### DQNAgent
- Standard Deep Q-Network with experience replay
- Dueling architecture, Double DQN
- Works but requires massive training (not recommended)

## Usage

```bash
# Activate conda environment
conda activate minesweeper-gpu

# Train hybrid agent (recommended)
python main.py train-hybrid --episodes 10000 --eval

# Evaluate agents
python main.py evaluate --agent hybrid --games 100
python main.py evaluate --agent logic --games 100

# Compare all agents
python main.py compare --games 200

# Watch agent play
python demo.py --size 9 --games 5 --delay 0.3

# Large board demo
python demo.py --size 50 --games 3 --delay 0.02
```

## Project Structure

```
minesweeper_ml_project/
├── main.py                 # CLI entry point
├── demo.py                 # Visual demo script
├── src/
│   ├── game/
│   │   ├── board.py        # Minesweeper board logic
│   │   ├── cell.py         # Cell state management
│   │   └── environment.py  # Gym-style environment
│   ├── agents/
│   │   ├── base_agent.py   # Abstract agent interface
│   │   ├── random_agent.py # Random baseline
│   │   ├── logic_agent.py  # AC-3 constraint solver
│   │   ├── dqn_agent.py    # Deep Q-Network
│   │   └── hybrid_agent.py # AC-3 + neural guessing
│   └── training/
│       └── trainer.py      # Training loops, evaluation
├── tests/                  # Unit tests (63 tests)
├── checkpoints/            # Saved models
└── requirements.txt
```

## Requirements

- Python 3.12+
- PyTorch 2.7+ (with CUDA 12.8 for GPU)
- NumPy

```bash
# CPU only
pip install -r requirements.txt

# GPU (RTX 50-series / Blackwell)
conda create -n minesweeper-gpu python=3.12 pytorch=2.7.0=gpu_cuda128_py312* -c pkgs/main
conda activate minesweeper-gpu
pip install gymnasium pytest
```

## Technical Details

### AC-3 Constraint Propagation

Each revealed number creates a constraint:
- Variable: hidden neighbor cells
- Domain: {safe, mine}
- Constraint: sum of mines = revealed number

The algorithm:
1. Build constraints from all revealed numbers
2. Propagate: if remaining_mines == 0, all hidden are safe
3. Propagate: if remaining_mines == hidden_count, all are mines
4. Subset reduction: compare constraint pairs for additional deductions
5. Repeat until no changes (fixpoint)

### Hybrid Training

The neural network only trains on "forced guess" situations:
- Input: normalized board state
- Output: safety score per cell
- Loss: binary cross-entropy (was the guess safe?)
- Only ~7% of moves require the network

This focused training is why Hybrid learns quickly (10k episodes) while pure DQN struggles even at 100k episodes.
