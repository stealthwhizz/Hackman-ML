"""
Hangman DQN Model
=================
Deep Q-Network (DQN) Agent for playing Hangman game.
Combines Hidden Markov Model (HMM) with Deep Reinforcement Learning.

Author: Akshay AG
Date: November 3, 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pickle
import re
from collections import defaultdict, Counter
from pathlib import Path


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class HangmanHMM:
    """Hidden Markov Model for Hangman letter prediction."""
    
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.models_by_length = {}  # Separate model for each word length
        
    def train(self, words):
        """Train HMM on corpus words."""
        print("Training HMM models...")
        
        # Group words by length
        words_by_length = defaultdict(list)
        for word in words:
            words_by_length[len(word)].append(word)
        
        # Train model for each length
        for length, word_list in words_by_length.items():
            if len(word_list) < 5:  # Skip lengths with too few examples
                continue
                
            model = {
                'position_freq': defaultdict(Counter),  # Letter freq at each position
                'bigram_freq': defaultdict(Counter),    # Letter pair frequencies
                'overall_freq': Counter(),              # Overall letter frequency
                'word_list': word_list                  # Store words for pattern matching
            }
            
            # Collect statistics
            for word in word_list:
                for i, letter in enumerate(word):
                    model['position_freq'][i][letter] += 1
                    model['overall_freq'][letter] += 1
                    
                    if i > 0:
                        model['bigram_freq'][word[i-1]][letter] += 1
            
            self.models_by_length[length] = model
        
        print(f"✓ Trained models for {len(self.models_by_length)} word lengths")
    
    def get_letter_probabilities(self, masked_word, guessed_letters):
        """Get probability distribution over remaining letters."""
        word_length = len(masked_word)
        
        # Get model for this word length
        if word_length not in self.models_by_length:
            return self._fallback_probabilities(guessed_letters)
        
        model = self.models_by_length[word_length]
        letter_scores = Counter()
        
        # Filter words matching the pattern
        matching_words = self._filter_matching_words(masked_word, model['word_list'])
        
        if matching_words:
            # Count letters in matching words
            for word in matching_words:
                for i, letter in enumerate(word):
                    if masked_word[i] == '_' and letter not in guessed_letters:
                        letter_scores[letter] += 1
        else:
            # Use position-based frequencies
            for i, char in enumerate(masked_word):
                if char == '_':
                    for letter in self.alphabet:
                        if letter not in guessed_letters:
                            letter_scores[letter] += model['position_freq'][i].get(letter, 0)
        
        # Normalize to probabilities
        total = sum(letter_scores.values()) or 1
        probabilities = {letter: score / total for letter, score in letter_scores.items()}
        
        # Ensure all unguessed letters have some probability
        for letter in self.alphabet:
            if letter not in guessed_letters and letter not in probabilities:
                probabilities[letter] = 1e-6
        
        return probabilities
    
    def _filter_matching_words(self, masked_word, word_list):
        """Filter words that match the masked pattern."""
        pattern = masked_word.replace('_', '.')
        regex = re.compile(f"^{pattern}$")
        return [word for word in word_list if regex.match(word)]
    
    def _fallback_probabilities(self, guessed_letters):
        """Fallback to uniform distribution over unguessed letters."""
        remaining = [l for l in self.alphabet if l not in guessed_letters]
        prob = 1.0 / len(remaining) if remaining else 0
        return {letter: prob for letter in remaining}
    
    def save(self, filepath):
        """Save HMM model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.models_by_length, f)
        print(f"✓ HMM model saved to {filepath}")
    
    def load(self, filepath):
        """Load HMM model from file."""
        with open(filepath, 'rb') as f:
            self.models_by_length = pickle.load(f)
        print(f"✓ HMM model loaded from {filepath}")


class HangmanEnv:
    """Hangman game environment for RL training and testing."""
    
    def __init__(self, word_list, max_wrong=6):
        self.word_list = word_list
        self.max_wrong = max_wrong
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.reset()
    
    def reset(self, word=None):
        """Reset environment for a new game."""
        self.word = word if word else random.choice(self.word_list)
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        self.done = False
        return self.get_state()
    
    def get_state(self):
        """Get current game state."""
        masked = ''.join([l if l in self.guessed_letters else '_' for l in self.word])
        return {
            'masked_word': masked,
            'guessed_letters': self.guessed_letters.copy(),
            'wrong_guesses': self.wrong_guesses,
            'lives_left': self.max_wrong - self.wrong_guesses,
            'word_length': len(self.word)
        }
    
    def step(self, letter):
        """Take an action (guess a letter) and return (state, reward, done, info)."""
        reward = 0
        info = {}
        
        # Check if letter already guessed
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            reward = -5
            info['repeated'] = True
        else:
            self.guessed_letters.add(letter)
            
            # Check if letter is in word
            if letter in self.word:
                occurrences = self.word.count(letter)
                reward = 10 * occurrences
                info['correct'] = True
            else:
                self.wrong_guesses += 1
                reward = -10
                info['correct'] = False
        
        # Check win/loss conditions
        masked = ''.join([l if l in self.guessed_letters else '_' for l in self.word])
        
        if '_' not in masked:
            self.done = True
            reward += 100
            info['won'] = True
        elif self.wrong_guesses >= self.max_wrong:
            self.done = True
            reward = -100
            info['won'] = False
        
        info['word'] = self.word
        return self.get_state(), reward, self.done, info
    
    def get_valid_actions(self):
        """Get list of letters that haven't been guessed yet."""
        return [l for l in self.alphabet if l not in self.guessed_letters]


class DQN(nn.Module):
    """Deep Q-Network for Hangman."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 128, 64]):
        super(DQN, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class HangmanDQNAgent:
    """DQN Agent for playing Hangman."""
    
    def __init__(self, hmm, max_word_length=20, gamma=0.95, lr=0.0005, 
                 epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=0.9995):
        self.hmm = hmm
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.max_word_length = max_word_length
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # State dimension: masked word + guessed letters + lives + HMM probs
        self.state_dim = max_word_length + 26 + 1 + 26
        self.action_dim = 26
        
        # Networks
        self.policy_net = DQN(self.state_dim, self.action_dim).to(device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.memory = []
        self.memory_size = 20000
    
    def state_to_vector(self, state):
        """Convert game state to neural network input vector."""
        masked_word = state['masked_word']
        guessed_letters = state['guessed_letters']
        lives_left = state['lives_left']
        
        # One-hot encode masked word (use 27 for _)
        masked_vector = np.zeros(self.max_word_length)
        for i, char in enumerate(masked_word[:self.max_word_length]):
            if char == '_':
                masked_vector[i] = 27
            else:
                masked_vector[i] = ord(char) - ord('a') + 1
        
        # Binary vector for guessed letters
        guessed_vector = np.array([1 if l in guessed_letters else 0 for l in self.alphabet])
        
        # Normalized lives
        lives_vector = np.array([lives_left / 6.0])
        
        # HMM probabilities
        hmm_probs = self.hmm.get_letter_probabilities(masked_word, guessed_letters)
        hmm_vector = np.array([hmm_probs.get(l, 0) for l in self.alphabet])
        
        # Concatenate all features
        state_vector = np.concatenate([masked_vector, guessed_vector, lives_vector, hmm_vector])
        return torch.FloatTensor(state_vector).to(device)
    
    def select_action(self, state, valid_actions):
        """Select action using epsilon-greedy policy with HMM bias."""
        if random.random() < self.epsilon:
            # Weighted exploration using HMM probabilities
            hmm_probs = self.hmm.get_letter_probabilities(state['masked_word'], state['guessed_letters'])
            valid_probs = [hmm_probs.get(a, 0.01) for a in valid_actions]
            prob_sum = sum(valid_probs)
            if prob_sum > 0:
                normalized_probs = [p/prob_sum for p in valid_probs]
                action = np.random.choice(valid_actions, p=normalized_probs)
            else:
                action = random.choice(valid_actions)
        else:
            # Exploit: best action from Q-network
            with torch.no_grad():
                state_vector = self.state_to_vector(state)
                q_values = self.policy_net(state_vector)
                
                # Mask invalid actions
                valid_indices = [self.alphabet.index(a) for a in valid_actions]
                masked_q = q_values.clone()
                invalid_mask = torch.ones_like(masked_q) * float('-inf')
                invalid_mask[valid_indices] = 0
                masked_q = masked_q + invalid_mask
                
                action_idx = masked_q.argmax().item()
                action = self.alphabet[action_idx]
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def train_step(self, batch_size=128):
        """Train the network on a batch of experiences."""
        if len(self.memory) < batch_size:
            return 0
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_batch = torch.stack([self.state_to_vector(s) for s in states])
        action_batch = torch.LongTensor([self.alphabet.index(a) for a in actions]).to(device)
        reward_batch = torch.FloatTensor(rewards).to(device)
        next_state_batch = torch.stack([self.state_to_vector(s) for s in next_states])
        done_batch = torch.FloatTensor(dones).to(device)
        
        # Compute Q-values
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values (Double DQN)
        with torch.no_grad():
            next_q_policy = self.policy_net(next_state_batch)
            next_actions = next_q_policy.argmax(1)
            next_q_target = self.target_net(next_state_batch)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath, save_hmm=True):
        """
        Save the DQN model and optionally the HMM model.
        
        Args:
            filepath: Path to save the model (e.g., 'models/dqn_agent.pt')
            save_hmm: Whether to save HMM model separately
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory[-1000:],  # Save last 1000 experiences
            'hyperparameters': {
                'max_word_length': self.max_word_length,
                'gamma': self.gamma,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"✓ DQN model saved to {filepath}")
        
        if save_hmm:
            hmm_path = filepath.parent / f"{filepath.stem}_hmm.pkl"
            self.hmm.save(hmm_path)
    
    def load_model(self, filepath, load_hmm=True):
        """
        Load the DQN model and optionally the HMM model.
        
        Args:
            filepath: Path to load the model from
            load_hmm: Whether to load HMM model separately
        """
        filepath = Path(filepath)
        
        checkpoint = torch.load(filepath, map_location=device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.memory = checkpoint.get('memory', [])
        
        print(f"✓ DQN model loaded from {filepath}")
        print(f"  - Epsilon: {self.epsilon:.4f}")
        print(f"  - Memory size: {len(self.memory)}")
        
        if load_hmm:
            hmm_path = filepath.parent / f"{filepath.stem}_hmm.pkl"
            if hmm_path.exists():
                self.hmm.load(hmm_path)
            else:
                print(f"⚠ Warning: HMM file not found at {hmm_path}")


def evaluate_agent(agent, test_words, num_games=2000, verbose=True):
    """
    Evaluate agent on test set.
    
    Args:
        agent: HangmanDQNAgent instance
        test_words: List of test words
        num_games: Number of games to play
        verbose: Whether to print progress
    
    Returns:
        Dictionary with evaluation metrics
    """
    agent.policy_net.eval()
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during evaluation
    
    wins = 0
    total_wrong_guesses = 0
    total_repeated_guesses = 0
    game_results = []
    
    env = HangmanEnv(test_words)
    
    if verbose:
        print("\n" + "="*80)
        print(f"EVALUATING DQN AGENT ON {num_games} GAMES")
        print("="*80)
    
    for game_num in range(num_games):
        state = env.reset()
        
        while not env.done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            state, reward, done, info = env.step(action)
        
        # Record results
        won = info.get('won', False)
        if won:
            wins += 1
        
        total_wrong_guesses += env.wrong_guesses
        total_repeated_guesses += env.repeated_guesses
        
        game_results.append({
            'game': game_num + 1,
            'word': env.word,
            'won': won,
            'wrong_guesses': env.wrong_guesses,
            'repeated_guesses': env.repeated_guesses
        })
        
        # Progress update
        if verbose and (game_num + 1) % 200 == 0:
            print(f"Played {game_num + 1}/{num_games} games | "
                  f"Current Win Rate: {wins/(game_num+1):.2%}")
    
    # Calculate final score
    success_rate = wins / num_games
    final_score = (success_rate * num_games) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)
    
    results = {
        'success_rate': success_rate,
        'wins': wins,
        'losses': num_games - wins,
        'total_wrong_guesses': total_wrong_guesses,
        'total_repeated_guesses': total_repeated_guesses,
        'avg_wrong_guesses': total_wrong_guesses / num_games,
        'avg_repeated_guesses': total_repeated_guesses / num_games,
        'final_score': final_score,
        'game_results': game_results
    }
    
    if verbose:
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"Games Played: {num_games}")
        print(f"Wins: {wins}")
        print(f"Losses: {num_games - wins}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"\nTotal Wrong Guesses: {total_wrong_guesses}")
        print(f"Avg Wrong Guesses per Game: {results['avg_wrong_guesses']:.2f}")
        print(f"\nTotal Repeated Guesses: {total_repeated_guesses}")
        print(f"Avg Repeated Guesses per Game: {results['avg_repeated_guesses']:.2f}")
        print(f"\n{'='*80}")
        print(f"FINAL SCORE: {final_score:.2f}")
        print(f"{'='*80}")
    
    agent.policy_net.train()
    agent.epsilon = original_epsilon
    
    return results


def load_data(corpus_path='Data/corpus.txt', test_path='Data/test.txt'):
    """Load training and test data."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_words = [word.strip().lower() for word in f.readlines()]
    
    with open(test_path, 'r', encoding='utf-8') as f:
        test_words = [word.strip().lower() for word in f.readlines()]
    
    print(f"✓ Loaded {len(corpus_words)} corpus words")
    print(f"✓ Loaded {len(test_words)} test words")
    
    return corpus_words, test_words


if __name__ == "__main__":
    print("="*80)
    print("HANGMAN DQN MODEL MODULE")
    print("="*80)
    print("\nThis module contains:")
    print("  • HangmanHMM - Hidden Markov Model for letter prediction")
    print("  • HangmanEnv - Game environment")
    print("  • DQN - Deep Q-Network architecture")
    print("  • HangmanDQNAgent - Complete DQN agent with save/load")
    print("  • evaluate_agent() - Testing function")
    print("  • load_data() - Data loading utility")
    print("\n" + "="*80)
