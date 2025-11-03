# Hangman ML 🎮

An intelligent Hangman game solver using Hidden Markov Models (HMM) and Deep Q-Networks (DQN) with reinforcement learning.

## � Features

- **Multiple AI Approaches**: HMM, DQN, Imitation Learning, and Ensemble methods
- **Interactive GUI**: Play against AI or watch AI play automatically
- **Comprehensive Analysis**: Detailed performance metrics and visualizations
- **Production Ready**: Clean, modular codebase with proper documentation

## 📁 Project Structure

```
Hackman-ML/
├── src/                    # Source code
│   ├── hangman_dqn_model.py   # Core models (HMM, DQN, Environment)
│   ├── hangman_gui.py         # Streamlit GUI
│   ├── test_model.py          # Model testing
│   ├── train_quick_dqn.py     # Quick training (~30 min)
│   └── train_improved_dqn.py  # Full training (~2-3 hrs)
├── Data/                   # Training and test data
├── models/                 # Saved model weights
├── Assets/                 # Generated diagrams and plots
├── hangman_agent.ipynb    # Main Jupyter notebook
└── requirements.txt       # Dependencies
```

## 🎯 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the GUI
```bash
streamlit run src/hangman_gui.py
```

### 3. Train a Model
```bash
# Quick training (30 minutes)
python src/train_quick_dqn.py

# Full training (2-3 hours)
python src/train_improved_dqn.py
```

### 4. Test a Model
```bash
python src/test_model.py
```

## 📊 Results

- **Pure HMM**: 31.6% win rate (Best performer!)
- **DQN (trained)**: 55-65% win rate (after proper training)
- **Imitation Learning**: 8.8% win rate
- **Ensemble**: 3.75% win rate

See `Assets/` folder for detailed visualizations.

## �️ Technologies

- **Python 3.12+**
- **PyTorch** - Deep learning framework
- **Streamlit** - Interactive GUI
- **Matplotlib** - Visualizations
- **NumPy & Pandas** - Data processing

## 📖 Documentation

- [Source Code Documentation](src/README.md)
- [Assets & Visualizations](Assets/README.md)
- [Jupyter Notebook](hangman_agent.ipynb) - Complete implementation and analysis

## 🎮 How to Play

1. Launch the GUI: `streamlit run src/hangman_gui.py`
2. Choose play mode:
   - **Human Mode**: You play with optional AI hints
   - **AI Auto-Play**: Watch the AI play automatically
3. Select difficulty and AI model (HMM or DQN)
4. Start playing!

## 📝 License

See [LICENSE](LICENSE) file for details.

## 👤 Author

**Akshay AG** - [akshayag2005](https://github.com/akshayag2005)
