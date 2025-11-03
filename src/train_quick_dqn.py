"""
Quick Improved DQN Training (10K episodes - ~30 minutes)
========================================================
Faster version for testing improvements before full 50K training.
"""

import numpy as np
import torch
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from hangman_dqn_model import (
    HangmanHMM,
    HangmanDQNAgent,
    HangmanEnv,
    evaluate_agent,
    load_data,
    device
)

# Set random seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def quick_train(agent, env, num_episodes=10000):
    """Quick training with optimal settings."""
    
    # Curriculum: word lengths by difficulty
    words_by_length = defaultdict(list)
    for word in env.word_list:
        words_by_length[len(word)].append(word)
    
    # Simpler curriculum for quick training
    curriculum = [
        (0, 3000, [3, 4, 5]),           # Short words
        (3000, 7000, [4, 5, 6, 7, 8]),  # Medium words
        (7000, 10000, list(range(3, 16)))  # All words
    ]
    
    episode_wins = []
    
    print(f"🚀 Quick Training: {num_episodes:,} episodes")
    print(f"   Device: {device}")
    print(f"   Curriculum stages: {len(curriculum)}")
    
    pbar = tqdm(range(num_episodes), desc="Training")
    
    for episode in pbar:
        # Select word by curriculum
        stage_lengths = [3, 4, 5]
        for start, end, lengths in curriculum:
            if start <= episode < end:
                stage_lengths = [l for l in lengths if l in words_by_length]
                break
        
        if stage_lengths:
            length = random.choice(stage_lengths)
            word = random.choice(words_by_length[length])
        else:
            word = random.choice(env.word_list)
        
        state = env.reset(word=word)
        
        while not env.done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train 3 times per step
            for _ in range(3):
                agent.train_step(batch_size=256)
            
            state = next_state
        
        episode_wins.append(1 if info.get('won', False) else 0)
        agent.decay_epsilon()
        
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Update progress
        if len(episode_wins) >= 100:
            recent_wr = np.mean(episode_wins[-100:])
            pbar.set_postfix({'Win%': f'{recent_wr*100:.1f}', 'ε': f'{agent.epsilon:.3f}'})
    
    pbar.close()
    
    final_wr = np.mean(episode_wins[-1000:]) if len(episode_wins) >= 1000 else np.mean(episode_wins)
    print(f"\n✅ Training complete!")
    print(f"   Final win rate (last 1000): {final_wr*100:.2f}%")
    
    return episode_wins


def main():
    print("="*80)
    print("QUICK IMPROVED DQN TRAINING (10K episodes)")
    print("="*80)
    
    # Load data
    print("\n[1/4] Loading data...")
    corpus_words, test_words = load_data()
    
    # Train HMM
    print("\n[2/4] Training HMM...")
    hmm = HangmanHMM()
    hmm.train(corpus_words)
    
    # Initialize agent with better hyperparameters
    print("\n[3/4] Initializing agent...")
    agent = HangmanDQNAgent(
        hmm=hmm,
        gamma=0.98,
        lr=0.0003,
        epsilon_start=0.95,
        epsilon_end=0.05,
        epsilon_decay=0.9995
    )
    print(f"   Parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    
    # Train
    print("\n[4/4] Training (this will take ~20-30 minutes)...")
    env = HangmanEnv(corpus_words)
    wins = quick_train(agent, env, num_episodes=10000)
    
    # Save model
    print("\n💾 Saving model...")
    model_path = Path('models/dqn_quick_10k.pt')
    agent.save_model(model_path, save_hmm=True)
    print(f"✓ Model saved: {model_path}")
    
    # Evaluate
    print("\n📊 Evaluating on 500 test games...")
    results = evaluate_agent(agent, test_words, num_games=500, verbose=True)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Training: 10,000 episodes")
    print(f"Test Win Rate: {results['success_rate']*100:.2f}%")
    print(f"Test Score: {results['final_score']:.2f}")
    print(f"\nComparison to HMM (31.6%):")
    if results['success_rate'] > 0.316:
        print(f"✅ DQN is better by {(results['success_rate']-0.316)*100:.2f} points!")
    else:
        gap = (0.316 - results['success_rate']) * 100
        print(f"📈 DQN needs {gap:.2f} more points")
        print(f"   Try running train_improved_dqn.py for 50K episodes")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
