"""
Hangman ML - Source Code Package
================================
This package contains all the source code for the Hangman ML project.

Modules:
- hangman_dqn_model: Core model implementations (HMM, DQN, Environment)
- hangman_gui: Streamlit GUI for playing Hangman
- test_model: Script for testing trained models
- train_quick_dqn: Quick training script (~30 minutes)
- train_improved_dqn: Full training script (~2-3 hours)
"""

__version__ = "1.0.0"
__author__ = "Akshay AG"

# You can import key classes here for easier access
from .hangman_dqn_model import (
    HangmanHMM,
    HangmanDQNAgent,
    HangmanEnv,
    DQN
)

__all__ = [
    'HangmanHMM',
    'HangmanDQNAgent', 
    'HangmanEnv',
    'DQN'
]
