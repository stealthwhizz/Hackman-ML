"""
Improved DQN Training Script for Hangman
=========================================
This script trains the DQN agent with optimized hyperparameters and longer training
to achieve better performance (target: 60%+ win rate).

Key Improvements:
1. Longer training (50,000 episodes instead of 5,000)
2. Curriculum learning (start with easier words)
3. Better exploration schedule
4. Enhanced reward shaping
5. Regular checkpointing
6. Performance tracking
"""

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json
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

print("="*80)
print("IMPROVED DQN TRAINING FOR HANGMAN")
print("="*80)


def train_improved_dqn(agent, env, num_episodes=50000, checkpoint_interval=5000):
    """
    Train DQN agent with curriculum learning and improved strategies.
    
    Improvements:
    - Curriculum learning: start with short words, gradually increase difficulty
    - Adaptive batch size
    - Regular checkpointing
    - Performance tracking
    """
    
    # Separate words by length for curriculum
    words_by_length = defaultdict(list)
    for word in env.word_list:
        words_by_length[len(word)].append(word)
    
    print(f"\n📚 Curriculum Learning Setup:")
    print(f"   Word lengths available: {sorted(words_by_length.keys())}")
    print(f"   Total training episodes: {num_episodes:,}")
    
    # Curriculum stages (progressive difficulty)
    curriculum = [
        (0, 5000, [3, 4, 5]),                  # Stage 1: Very short words
        (5000, 15000, [4, 5, 6, 7]),           # Stage 2: Short-medium words
        (15000, 30000, [5, 6, 7, 8, 9]),       # Stage 3: Medium words
        (30000, 40000, [6, 7, 8, 9, 10, 11]),  # Stage 4: Medium-long words
        (40000, 50000, list(range(3, 16)))     # Stage 5: All words
    ]
    
    print("\n🎯 Training Stages:")
    for i, (start, end, lengths) in enumerate(curriculum, 1):
        print(f"   Stage {i}: Episodes {start:,}-{end:,} | Lengths {lengths}")
    
    # Training metrics
    episode_rewards = []
    episode_wins = []
    episode_lengths = []
    losses = []
    
    # Best model tracking
    best_win_rate = 0
    best_episode = 0
    
    # Create checkpoints directory
    checkpoint_dir = Path('models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting training...")
    print(f"   Device: {device}")
    print(f"   Initial epsilon: {agent.epsilon:.3f}")
    print(f"   Memory size: {agent.memory_size:,}")
    
    # Progress bar
    pbar = tqdm(range(num_episodes), desc="Training", unit="episode")
    
    for episode in pbar:
        # Select word based on curriculum
        stage_lengths = [3, 4, 5]  # default
        for start, end, lengths in curriculum:
            if start <= episode < end:
                stage_lengths = [l for l in lengths if l in words_by_length]
                break
        
        if stage_lengths:
            length = random.choice(stage_lengths)
            selected_word = random.choice(words_by_length[length])
        else:
            selected_word = random.choice(env.word_list)
        
        state = env.reset(word=selected_word)
        episode_reward = 0
        step_count = 0
        
        while not env.done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train more frequently in later stages
            train_frequency = 3 if episode > 20000 else 2
            for _ in range(train_frequency):
                loss = agent.train_step(batch_size=256)
                if loss > 0:
                    losses.append(loss)
            
            episode_reward += reward
            step_count += 1
            state = next_state
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_wins.append(1 if info.get('won', False) else 0)
        episode_lengths.append(step_count)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update target network
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Update progress bar
        if len(episode_wins) >= 100:
            recent_win_rate = np.mean(episode_wins[-100:])
            recent_reward = np.mean(episode_rewards[-100:])
            pbar.set_postfix({
                'Win%': f'{recent_win_rate*100:.1f}',
                'Reward': f'{recent_reward:.0f}',
                'ε': f'{agent.epsilon:.3f}'
            })
            
            # Track best model
            if recent_win_rate > best_win_rate:
                best_win_rate = recent_win_rate
                best_episode = episode
        
        # Checkpoint every N episodes
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f'dqn_ep{episode+1}.pt'
            agent.save_model(checkpoint_path, save_hmm=False)
            
            recent_win_rate = np.mean(episode_wins[-1000:]) if len(episode_wins) >= 1000 else np.mean(episode_wins)
            
            print(f"\n✓ Checkpoint saved: {checkpoint_path.name}")
            print(f"  Episode {episode+1:,}/{num_episodes:,}")
            print(f"  Win rate (last 1000): {recent_win_rate*100:.2f}%")
            print(f"  Memory size: {len(agent.memory):,}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
    
    pbar.close()
    
    print(f"\n✅ Training complete!")
    print(f"   Best win rate: {best_win_rate*100:.2f}% (episode {best_episode:,})")
    print(f"   Final epsilon: {agent.epsilon:.4f}")
    print(f"   Final memory size: {len(agent.memory):,}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_wins': episode_wins,
        'episode_lengths': episode_lengths,
        'losses': losses,
        'best_win_rate': best_win_rate,
        'best_episode': best_episode
    }


def plot_training_progress(metrics, save_path='training_progress_improved.png'):
    """Plot comprehensive training metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Improved DQN Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Win rate over time
    ax1 = axes[0, 0]
    wins = metrics['episode_wins']
    window = 500
    if len(wins) >= window:
        win_rate = np.convolve(wins, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(wins)), win_rate * 100, linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title(f'Win Rate (Moving Average, window={window})')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='r', linestyle='--', label='Target: 50%')
    ax1.legend()
    
    # 2. Episode rewards
    ax2 = axes[0, 1]
    rewards = metrics['episode_rewards']
    if len(rewards) >= window:
        avg_reward = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(rewards)), avg_reward, linewidth=2, color='orange')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.set_title(f'Episode Reward (Moving Average, window={window})')
    ax2.grid(True, alpha=0.3)
    
    # 3. Training loss
    ax3 = axes[1, 0]
    losses = metrics['losses']
    if len(losses) > 1000:
        # Sample losses for plotting (too many points)
        step = max(1, len(losses) // 10000)
        sampled_losses = losses[::step]
        ax3.plot(sampled_losses, alpha=0.6, linewidth=1)
        # Add moving average
        loss_window = min(100, len(sampled_losses) // 10)
        if len(sampled_losses) >= loss_window:
            avg_loss = np.convolve(sampled_losses, np.ones(loss_window)/loss_window, mode='valid')
            ax3.plot(range(loss_window-1, len(sampled_losses)), avg_loss, 
                    linewidth=2, color='red', label='Moving Avg')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Episode length distribution
    ax4 = axes[1, 1]
    lengths = metrics['episode_lengths']
    if len(lengths) >= 1000:
        # Show recent distribution
        recent_lengths = lengths[-5000:]
        ax4.hist(recent_lengths, bins=30, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Episode Length (Steps)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Episode Length Distribution (Last 5000 episodes)')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Training progress plot saved: {save_path}")
    plt.close()


def main():
    """Main training function."""
    
    # Configuration
    NUM_TRAINING_EPISODES = 50000  # Much longer training
    CHECKPOINT_INTERVAL = 5000
    FINAL_TEST_GAMES = 2000
    
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Training Episodes: {NUM_TRAINING_EPISODES:,}")
    print(f"Checkpoint Interval: {CHECKPOINT_INTERVAL:,}")
    print(f"Final Test Games: {FINAL_TEST_GAMES:,}")
    print(f"Device: {device}")
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    corpus_words, test_words = load_data()
    
    # Step 2: Train HMM
    print("\n[2/5] Training HMM...")
    hmm = HangmanHMM()
    hmm.train(corpus_words)
    
    # Step 3: Initialize improved DQN agent
    print("\n[3/5] Initializing improved DQN agent...")
    agent = HangmanDQNAgent(
        hmm=hmm,
        max_word_length=20,
        gamma=0.98,  # Increased from 0.95
        lr=0.0003,   # Decreased from 0.0005
        epsilon_start=0.95,  # Start higher
        epsilon_end=0.05,
        epsilon_decay=0.9998  # Slower decay
    )
    
    print(f"  State dimension: {agent.state_dim}")
    print(f"  Action dimension: {agent.action_dim}")
    print(f"  Network parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    print(f"  Gamma: {agent.gamma}")
    print(f"  Learning rate: 0.0003")
    
    # Step 4: Train with improvements
    print("\n[4/5] Training improved DQN agent...")
    env = HangmanEnv(corpus_words)
    
    metrics = train_improved_dqn(
        agent, 
        env, 
        num_episodes=NUM_TRAINING_EPISODES,
        checkpoint_interval=CHECKPOINT_INTERVAL
    )
    
    # Plot training progress
    plot_training_progress(metrics)
    
    # Save final model
    print("\n💾 Saving final model...")
    final_model_path = Path('models/dqn_agent_improved_final.pt')
    agent.save_model(final_model_path, save_hmm=True)
    print(f"✓ Final model saved: {final_model_path}")
    
    # Save training metrics
    metrics_path = Path('models/training_metrics_improved.json')
    with open(metrics_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_metrics = {
            'best_win_rate': float(metrics['best_win_rate']),
            'best_episode': int(metrics['best_episode']),
            'final_win_rate': float(np.mean(metrics['episode_wins'][-1000:])),
            'total_episodes': len(metrics['episode_wins'])
        }
        json.dump(json_metrics, f, indent=2)
    print(f"✓ Training metrics saved: {metrics_path}")
    
    # Step 5: Final evaluation
    print("\n[5/5] Final evaluation on test set...")
    results = evaluate_agent(agent, test_words, num_games=FINAL_TEST_GAMES, verbose=True)
    
    # Save test results
    results_path = Path('models/test_results_improved.json')
    with open(results_path, 'w') as f:
        json_results = {
            'success_rate': results['success_rate'],
            'wins': results['wins'],
            'losses': results['losses'],
            'total_wrong_guesses': results['total_wrong_guesses'],
            'total_repeated_guesses': results['total_repeated_guesses'],
            'avg_wrong_guesses': results['avg_wrong_guesses'],
            'avg_repeated_guesses': results['avg_repeated_guesses'],
            'final_score': results['final_score']
        }
        json.dump(json_results, f, indent=2)
    print(f"\n✓ Test results saved: {results_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)
    print(f"Training Episodes: {NUM_TRAINING_EPISODES:,}")
    print(f"Best Training Win Rate: {metrics['best_win_rate']*100:.2f}% (episode {metrics['best_episode']:,})")
    print(f"\nFinal Test Results:")
    print(f"  Win Rate: {results['success_rate']*100:.2f}%")
    print(f"  Games Won: {results['wins']}/{FINAL_TEST_GAMES}")
    print(f"  Avg Wrong Guesses: {results['avg_wrong_guesses']:.2f}")
    print(f"  Final Score: {results['final_score']:.2f}")
    
    # Compare with HMM baseline
    print(f"\n📊 Comparison with HMM Baseline:")
    print(f"  HMM Win Rate: 31.6%")
    print(f"  Improved DQN: {results['success_rate']*100:.2f}%")
    if results['success_rate'] > 0.316:
        print(f"  🎉 DQN surpassed HMM by {(results['success_rate']-0.316)*100:.2f} percentage points!")
    else:
        print(f"  📈 DQN needs {(0.316-results['success_rate'])*100:.2f} more percentage points to match HMM")
    
    print("\n" + "="*80)
    print("All files saved in 'models/' directory")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
