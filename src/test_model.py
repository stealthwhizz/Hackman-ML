"""
Test Script for Hangman DQN Model
==================================
This script demonstrates how to:
1. Load a trained DQN model
2. Test it on the test dataset
3. Display performance metrics
"""

import torch
from hangman_dqn_model import (
    HangmanHMM, 
    HangmanDQNAgent, 
    evaluate_agent, 
    load_data
)
from pathlib import Path


def main():
    print("\n" + "="*80)
    print("HANGMAN DQN MODEL - TESTING SCRIPT")
    print("="*80)
    
    # Load data
    print("\n[1/4] Loading data...")
    corpus_words, test_words = load_data(test_path='Data/test_data.txt')
    
    # Initialize HMM
    print("\n[2/4] Initializing and training HMM...")
    hmm = HangmanHMM()
    hmm.train(corpus_words)
    
    # Initialize agent
    print("\n[3/4] Loading trained DQN model...")
    agent = HangmanDQNAgent(hmm)
    
    # Check if model file exists
    model_path = Path('models/dqn_agent_final.pt')
    
    if model_path.exists():
        agent.load_model(model_path, load_hmm=False)
        print(f"✓ Model loaded successfully from {model_path}")
    else:
        print(f"⚠ Warning: Model file not found at {model_path}")
        print("   The agent will use randomly initialized weights.")
        print("   To test with a trained model, please train first or")
        print("   update the model_path variable.")
    
    # Evaluate
    print("\n[4/4] Evaluating on test set...")
    results = evaluate_agent(agent, test_words, num_games=2000, verbose=True)
    
    # Display detailed results
    print("\n" + "="*80)
    print("DETAILED METRICS")
    print("="*80)
    print(f"Win Rate: {results['success_rate']*100:.2f}%")
    print(f"Average Wrong Guesses: {results['avg_wrong_guesses']:.2f}")
    print(f"Average Repeated Guesses: {results['avg_repeated_guesses']:.2f}")
    print(f"\n🏆 FINAL SCORE: {results['final_score']:.2f}")
    print("="*80)
    
    # Save results
    import json
    results_path = Path('results/test_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
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
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    results = main()
