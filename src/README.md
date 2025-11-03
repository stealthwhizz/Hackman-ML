# Source Code (src/)

This folder contains all the Python source code for the Hangman ML project.

## 📁 Files

### Core Module
- **`hangman_dqn_model.py`** - Main model implementation
  - `HangmanHMM` - Hidden Markov Model for letter prediction
  - `HangmanDQNAgent` - Deep Q-Network reinforcement learning agent
  - `HangmanEnv` - Hangman game environment
  - `DQN` - Neural network architecture
  - Utility functions: `load_data()`, `save_model()`, `load_model()`

### Training Scripts
- **`train_quick_dqn.py`** - Quick training (~30 minutes)
  - 10,000 episodes
  - Curriculum learning
  - Target accuracy: 55-65%
  
- **`train_improved_dqn.py`** - Full training (~2-3 hours)
  - 50,000 episodes
  - Checkpointing every 5,000 episodes
  - Maximum accuracy optimization

### Testing & GUI
- **`test_model.py`** - Model evaluation script
  - Tests saved models on test dataset
  - Generates performance reports
  
- **`hangman_gui.py`** - Interactive Streamlit GUI
  - Play against AI or watch AI play
  - Real-time visualization
  - Statistics tracking

## 🚀 Usage

### Training a Model
```bash
# Quick training (30 minutes)
cd src
python train_quick_dqn.py

# Full training (2-3 hours)
python train_improved_dqn.py
```

### Testing a Model
```bash
cd src
python test_model.py
```

### Running the GUI
```bash
# From project root
streamlit run src/hangman_gui.py

# Or from src folder
cd src
streamlit run hangman_gui.py
```

## 📦 Package Structure

The `__init__.py` file makes this directory a Python package, allowing imports like:
```python
from src import HangmanHMM, HangmanDQNAgent
```

## 🔧 Dependencies

All required packages are listed in the root `requirements.txt`:
- PyTorch
- Streamlit
- NumPy
- Pandas
- Matplotlib
- tqdm
