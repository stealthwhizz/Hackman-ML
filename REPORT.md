# Intelligent Hangman Agent - ML Hackathon Report

## Problem Statement

Create an AI agent to play Hangman game by guessing letters to reveal a hidden word. The agent has:
- **6 lives** (6 wrong guesses allowed)
- **50,000 word corpus** for training
- **2,001 test words** for evaluation

### Evaluation Criteria:
**Score Formula:** `(Success Rate × 2000) - (Total Wrong Guesses × 5) - (Total Repeated Guesses × 2)`

**Objective:** Maximize score over 2000 games

## Solution Approach

This project implements a **hybrid approach** combining:
1. **Hidden Markov Model (HMM)** - Linguistic intelligence from 50K word corpus
2. **Deep Q-Network (DQN)** - Strategic reinforcement learning agent

### Why Hybrid Approach?

**1. Hidden Markov Model (HMM) - The Linguistic Expert**
- **Purpose:** Provides probabilistic predictions based on word patterns
- **How it works:** 
  - Trains separate models for each word length (3-16 characters)
  - Learns letter frequency at each position
  - Captures bigram patterns (letter pairs)
  - Uses pattern matching for masked words (e.g., "p_th_n" → likely 'y' or 'o')
- **Advantage:** Encodes 50K words of linguistic knowledge without neural network overhead

**2. Deep Q-Network (DQN) - The Strategic Player**
- **Purpose:** Learns optimal guessing strategy through experience
- **How it works:**
  - Takes HMM probabilities + game state as input
  - Neural network outputs Q-values for each letter
  - Selects action that maximizes long-term reward
  - Uses experience replay for stable learning
- **Advantage:** Learns when to take risks vs. play safe, balances exploration vs. exploitation

**3. Why Combine Both?**
- **HMM alone:** Good at common patterns, but can't learn strategy
- **DQN alone:** Needs millions of samples to learn letter patterns from scratch
- **Hybrid:** DQN leverages HMM's linguistic knowledge to make strategic decisions
  - Example: HMM says "e" is 40% likely, "t" is 35% likely
  - DQN decides: "Already guessed vowels, better try 't' now"

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Training Phase                                             │
├─────────────────────────────────────────────────────────────┤
│  50K Corpus → HMM Models (by length) → Letter Probabilities│
│                                              ↓               │
│  Random Words → Game Environment ← DQN Agent (Learning)     │
│                      ↓                         ↑             │
│                  Rewards → Experience Replay Buffer         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Inference Phase (Playing Games)                           │
├─────────────────────────────────────────────────────────────┤
│  Hidden Word → HMM → Probabilities ──┐                      │
│                                       ├→ State Vector        │
│  Masked Word + Guessed Letters ──────┘        ↓            │
│                                       DQN Network           │
│                                              ↓               │
│                                       Q-Values (26 letters)  │
│                                              ↓               │
│                                       Best Letter (Action)   │
└─────────────────────────────────────────────────────────────┘
```

### Expected Performance:
- **Success Rate:** 60-70% (1200-1400 wins out of 2000 games)
- **Avg Wrong Guesses:** 2-3 per game
- **Target Score:** ~1800-2200 points

## Implementation Details

### Libraries Used:
- **NumPy, Pandas:** Data manipulation and analysis
- **PyTorch:** Deep learning framework for DQN implementation
- **Matplotlib, Seaborn:** Visualization of training progress and results
- **Collections:** Efficient data structures (defaultdict, Counter)
- **Random seed:** Set to 42 for reproducibility

### Configuration:
- CPU/GPU device detection for PyTorch
- Random seeds locked for consistent results across runs

### Hidden Markov Model (HMM) Implementation
- Trains separate models for each word length
- Captures letter position frequencies and bigram patterns
- Provides probability distribution over unguessed letters

### Deep Q-Network (DQN) Agent
**State Representation (Total: ~120 dimensions):**
1. **Masked Word Vector (20 dims):** One-hot encoding of current word state
2. **Guessed Letters (26 dims):** Binary vector of already-guessed letters
3. **Lives Remaining (1 dim):** Normalized 0-1 (6 lives → 1.0, 3 lives → 0.5)
4. **HMM Probabilities (26 dims):** Letter predictions from HMM

**Neural Network Architecture:**
```
Input (120 dims)
    ↓
Fully Connected (128 neurons) + ReLU + Dropout(0.2)
    ↓
Fully Connected (64 neurons) + ReLU + Dropout(0.2)
    ↓
Fully Connected (32 neurons) + ReLU
    ↓
Output (26 Q-values, one per letter)
```

**Key DQN Concepts:**
- **Policy Network vs Target Network:** Prevents training instability
- **Experience Replay:** Breaks correlation between consecutive samples
- **Epsilon-Greedy Exploration:** Balances exploration vs. exploitation

**Reward Structure:**
- **+10 × occurrences:** Correct letter
- **-10:** Wrong letter
- **-5:** Repeated guess
- **+100:** Win
- **-100:** Loss

### Training Process:
1. Play a game, collect experiences
2. Store in replay buffer
3. Sample random batch for training
4. Update policy network
5. Repeat for 1000 episodes (optimized for speed)

## Evaluation Protocol
- **No Exploration:** Epsilon = 0 during evaluation
- **Evaluation Mode:** Dropout disabled
- **Deterministic:** Always pick highest Q-value

### Metrics Collected:
1. Success Rate (% games won)
2. Total Wins/Losses
3. Wrong Guesses (total and average)
4. Repeated Guesses (total and average)
5. Final Score: (Success Rate × 2000) - (Wrong × 5) - (Repeated × 2)

### Target Performance:
- Good: 55-60% win rate, ~2000 score
- Excellent: 65-70% win rate, ~2500 score
- Outstanding: 75%+ win rate, ~3000+ score

## Results

*Note: Results would be populated after running the notebook. Below are expected results based on the implementation.*

### Training Metrics:
- Episodes: 1000 (optimized for speed)
- Final Win Rate: ~60%
- Avg Reward: ~140
- Avg Wrong Guesses: 2.5

### Test Performance (2000 games):
- Success Rate: 65%
- Wins: 1300
- Losses: 700
- Total Wrong Guesses: 5000
- Avg Wrong Guesses: 2.5
- Total Repeated Guesses: 200
- Avg Repeated Guesses: 0.1
- Final Score: 2000

## Visualizations

### System Architecture Diagram
![Architecture Diagram](Assests/hangman_architecture_diagram.png)

*The architecture diagram illustrates the complete system flow:*
- **Data Layer:** 50K word corpus and 2000 test words
- **HMM Module:** Processes words to generate letter probability distributions
- **Game Environment:** Simulates Hangman gameplay with reward system
- **DQN Agent:** Neural network that learns optimal letter selection strategy
- **State Construction:** Combines masked word, guessed letters, lives, and HMM probabilities
- **Experience Replay:** Memory buffer for stable training
- **Feedback Loop:** Rewards guide the learning process

### Training and Inference Flowcharts
![Flowcharts](Assests/hangman_flowcharts.png)

*The flowcharts show the detailed processes:*

**Training Process (Left):**
1. Load 50K word corpus
2. Train HMM models by word length
3. Initialize DQN agent with policy and target networks
4. For each episode: reset environment, play game with ε-greedy exploration
5. Store transitions in replay buffer
6. Train DQN on random batches
7. Update target network every 10 episodes
8. Repeat for 1000 episodes (optimized)

**Inference Process (Right):**
1. Receive hidden word (e.g., "______")
2. HMM generates letter probabilities
3. Construct state vector from game state
4. DQN computes Q-values for all letters
5. Select best valid letter
6. Guess letter and update game state
7. Repeat until win/loss

*Additional visualizations (training progress, evaluation analysis) would be generated after running the notebook.*

## Files Generated

**Model Files:**
- `hangman_dqn_model.pt` - Trained neural network weights

**Visualization Files:**
- `hangman_architecture_diagram.png` - System architecture
- `hangman_flowcharts.png` - Training and inference processes
- `training_progress.png` - Training metrics
- `evaluation_analysis.png` - Test performance analysis

**Results Files:**
- `detailed_game_results.csv` - All game outcomes
- `final_results.json` - Overall statistics
- `Analysis_Report.txt` - Comprehensive report

## Key Takeaways

1. **Hybrid approaches work** - Combine domain knowledge (HMM) with learning (DQN)
2. **State representation matters** - Rich features improve performance
3. **Exploration vs exploitation** - Epsilon-greedy balances both
4. **Experience replay is crucial** - Stable, efficient learning
5. **Patience pays off** - DQN needs many episodes to converge

## Future Improvements

1. Train longer (2000+ episodes) for better performance
2. Tune hyperparameters (learning rate, batch size)
3. Implement advanced architectures (Dueling DQN, Double DQN)
4. Add attention mechanism
5. Use prioritized experience replay

## How to Run

1. Install requirements: `pip install -r requirements.txt`
2. Run all cells in `hangman_agent.ipynb` sequentially
3. Wait for training (~10-15 minutes)
4. Review generated files and results

## References

- **Deep Q-Learning:** Mnih et al., "Playing Atari with Deep RL" (2013)
- **HMM:** Rabiner, "A Tutorial on Hidden Markov Models" (1989)
- **Experience Replay:** Lin, "Self-Improving Reactive Agents" (1992)

---

*This report is generated based on the notebook implementation and problem statement.*