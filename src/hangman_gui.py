"""
Hangman Game - Interactive GUI
===============================
Play Hangman against AI agents (HMM or DQN) with a beautiful interface.

Run with: streamlit run hangman_gui.py
"""

import streamlit as st
import random
import sys
from pathlib import Path

# Import our models
from hangman_dqn_model import (
    HangmanHMM,
    HangmanDQNAgent,
    HangmanEnv,
    load_data,
    device
)

# Page configuration
st.set_page_config(
    page_title="Hangman AI Game",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2ecc71;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .word-display {
        font-size: 4rem;
        text-align: center;
        letter-spacing: 1rem;
        font-family: monospace;
        color: #3498db;
        margin: 2rem 0;
    }
    .letter-button {
        font-size: 1.5rem;
        margin: 0.2rem;
    }
    .game-over-win {
        font-size: 2rem;
        color: #2ecc71;
        text-align: center;
        font-weight: bold;
    }
    .game-over-lose {
        font-size: 2rem;
        color: #e74c3c;
        text-align: center;
        font-weight: bold;
    }
    .stats-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.hmm = None
    st.session_state.dqn_agent = None
    st.session_state.corpus_words = None
    st.session_state.test_words = None

if 'game_active' not in st.session_state:
    st.session_state.game_active = False
    st.session_state.word = ""
    st.session_state.masked_word = ""
    st.session_state.guessed_letters = set()
    st.session_state.wrong_guesses = 0
    st.session_state.max_wrong = 6
    st.session_state.game_won = False
    st.session_state.game_lost = False
    st.session_state.total_games = 0
    st.session_state.total_wins = 0
    st.session_state.ai_playing = False
    st.session_state.play_mode = "Human"  # "Human" or "AI"


@st.cache_resource
def load_models():
    """Load HMM and DQN models (cached)."""
    try:
        # Load data
        corpus_words, test_words = load_data()
        
        # Load HMM
        hmm = HangmanHMM()
        hmm.train(corpus_words)
        
        # Try to load trained DQN
        dqn_agent = HangmanDQNAgent(hmm)
        model_path = Path('models/dqn_agent_final.pt')
        
        if model_path.exists():
            dqn_agent.load_model(model_path, load_hmm=False)
            dqn_status = "✅ Trained model loaded"
        else:
            dqn_status = "⚠️ Untrained (train first)"
        
        return hmm, dqn_agent, corpus_words, test_words, dqn_status
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, "❌ Error"


def draw_hangman(wrong_guesses):
    """Draw hangman ASCII art based on wrong guesses."""
    stages = [
        # Stage 0: Empty
        """
           ┌─────┐
           │     
           │     
           │     
           │     
           │     
        ═══════════
        """,
        # Stage 1: Head
        """
           ┌─────┐
           │     │
           │     O
           │     
           │     
           │     
        ═══════════
        """,
        # Stage 2: Body
        """
           ┌─────┐
           │     │
           │     O
           │     │
           │     
           │     
        ═══════════
        """,
        # Stage 3: Left arm
        """
           ┌─────┐
           │     │
           │     O
           │    ╱│
           │     
           │     
        ═══════════
        """,
        # Stage 4: Right arm
        """
           ┌─────┐
           │     │
           │     O
           │    ╱│╲
           │     
           │     
        ═══════════
        """,
        # Stage 5: Left leg
        """
           ┌─────┐
           │     │
           │     O
           │    ╱│╲
           │    ╱ 
           │     
        ═══════════
        """,
        # Stage 6: Right leg (Game Over)
        """
           ┌─────┐
           │     │
           │     O
           │    ╱│╲
           │    ╱ ╲
           │     
        ═══════════
        """
    ]
    return stages[min(wrong_guesses, 6)]


def start_new_game(mode, difficulty, play_mode="Human"):
    """Start a new game."""
    # Select word based on difficulty
    if difficulty == "Easy (3-5 letters)":
        words = [w for w in st.session_state.test_words if 3 <= len(w) <= 5]
    elif difficulty == "Medium (6-8 letters)":
        words = [w for w in st.session_state.test_words if 6 <= len(w) <= 8]
    elif difficulty == "Hard (9-12 letters)":
        words = [w for w in st.session_state.test_words if 9 <= len(w) <= 12]
    else:  # Very Hard
        words = [w for w in st.session_state.test_words if len(w) >= 13]
    
    if not words:
        words = st.session_state.test_words
    
    word = random.choice(words).lower()
    
    st.session_state.game_active = True
    st.session_state.word = word
    st.session_state.masked_word = "_" * len(word)
    st.session_state.guessed_letters = set()
    st.session_state.wrong_guesses = 0
    st.session_state.game_won = False
    st.session_state.game_lost = False
    st.session_state.game_mode = mode
    st.session_state.play_mode = play_mode
    st.session_state.ai_playing = (play_mode == "AI")


def make_guess(letter):
    """Process a letter guess."""
    if letter in st.session_state.guessed_letters:
        return "Already guessed!"
    
    st.session_state.guessed_letters.add(letter)
    
    if letter in st.session_state.word:
        # Correct guess - update masked word
        new_masked = ""
        for i, char in enumerate(st.session_state.word):
            if char in st.session_state.guessed_letters:
                new_masked += char
            else:
                new_masked += "_"
        st.session_state.masked_word = new_masked
        
        # Check win
        if "_" not in st.session_state.masked_word:
            st.session_state.game_won = True
            st.session_state.game_active = False
            st.session_state.total_games += 1
            st.session_state.total_wins += 1
        
        return "Correct! ✅"
    else:
        # Wrong guess
        st.session_state.wrong_guesses += 1
        
        # Check loss
        if st.session_state.wrong_guesses >= st.session_state.max_wrong:
            st.session_state.game_lost = True
            st.session_state.game_active = False
            st.session_state.total_games += 1
        
        return "Wrong! ❌"


def get_ai_hint(mode):
    """Get AI suggestion for next letter."""
    if mode == "HMM":
        probs = st.session_state.hmm.get_letter_probabilities(
            st.session_state.masked_word,
            st.session_state.guessed_letters
        )
        # Get top 3
        top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        return top_3
    else:  # DQN
        env = HangmanEnv([st.session_state.word])
        state = {
            'masked_word': st.session_state.masked_word,
            'guessed_letters': st.session_state.guessed_letters,
            'wrong_guesses': st.session_state.wrong_guesses,
            'lives_left': st.session_state.max_wrong - st.session_state.wrong_guesses,
            'word_length': len(st.session_state.word)
        }
        valid_actions = [l for l in 'abcdefghijklmnopqrstuvwxyz' 
                        if l not in st.session_state.guessed_letters]
        
        if valid_actions:
            suggestion = st.session_state.dqn_agent.select_action(state, valid_actions)
            return [(suggestion, 1.0)]
        return []


def ai_make_move(mode):
    """AI makes the next move automatically."""
    hints = get_ai_hint(mode)
    if hints:
        letter = hints[0][0]
        return make_guess(letter), letter
    return None, None


def main():
    """Main app function."""
    
    # Header
    st.markdown('<p class="main-header">🎮 HANGMAN AI GAME 🎮</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Game Settings")
        
        # Load models button
        if not st.session_state.initialized:
            if st.button("🔄 Load AI Models", use_container_width=True):
                with st.spinner("Loading models..."):
                    hmm, dqn, corpus, test, dqn_status = load_models()
                    if hmm:
                        st.session_state.hmm = hmm
                        st.session_state.dqn_agent = dqn
                        st.session_state.corpus_words = corpus
                        st.session_state.test_words = test
                        st.session_state.initialized = True
                        st.success("✅ Models loaded!")
                        st.info(f"DQN Status: {dqn_status}")
                        st.rerun()
        
        if st.session_state.initialized:
            st.success("✅ AI Models Ready")
            
            st.markdown("---")
            
            # Play mode selection
            play_mode = st.radio(
                "🎮 Play Mode:",
                ["Human (You Play)", "AI Auto-Play (Watch AI)"],
                help="Choose who plays the game"
            )
            
            # Game mode selection (AI to use)
            game_mode = st.radio(
                "🤖 AI Model:",
                ["HMM (Pattern Matching)", "DQN (Neural Network)", "No AI (Pure Human)"],
                help="Choose which AI model to use"
            )
            
            # Difficulty selection
            difficulty = st.selectbox(
                "🎯 Difficulty:",
                ["Easy (3-5 letters)", "Medium (6-8 letters)", 
                 "Hard (9-12 letters)", "Very Hard (13+ letters)"]
            )
            
            st.markdown("---")
            
            # New game button
            play_mode_val = "AI" if "AI Auto-Play" in play_mode else "Human"
            if st.button("🎲 New Game", use_container_width=True):
                start_new_game(game_mode, difficulty, play_mode_val)
                st.rerun()
            
            st.markdown("---")
            
            # Statistics
            st.subheader("📊 Statistics")
            if st.session_state.total_games > 0:
                win_rate = st.session_state.total_wins / st.session_state.total_games
                st.metric("Total Games", st.session_state.total_games)
                st.metric("Wins", st.session_state.total_wins)
                st.metric("Win Rate", f"{win_rate:.1%}")
            else:
                st.info("Play games to see statistics")
            
            st.markdown("---")
            
            # How to play
            with st.expander("📖 How to Play"):
                st.markdown("""
                **Objective:** Guess the hidden word letter by letter!
                
                **Rules:**
                - You have 6 wrong guesses allowed
                - Each wrong guess adds a body part to the hangman
                - Win by revealing the whole word
                - Lose if hangman is complete
                
                **AI Assistance:**
                - **HMM:** Uses statistical patterns from 50K words
                - **DQN:** Uses trained neural network
                - **No AI:** Pure human skill!
                
                **Tips:**
                - Start with common vowels (e, a, i, o, u)
                - Then try common consonants (r, t, n, s, l)
                - Use AI hints wisely!
                """)
    
    # Main game area
    if not st.session_state.initialized:
        st.info("👈 Click 'Load AI Models' in the sidebar to start!")
        
        # Show demo info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🧠 HMM Agent")
            st.write("Pattern matching using statistical analysis of 50,000 words")
            st.success("Win Rate: 31.6%")
        
        with col2:
            st.markdown("### 🤖 DQN Agent")
            st.write("Deep Q-Network trained with reinforcement learning")
            st.info("Train for better accuracy!")
        
        with col3:
            st.markdown("### 🎮 Play Mode")
            st.write("Use AI hints or play solo - your choice!")
            st.warning("Challenge yourself!")
        
        return
    
    if not st.session_state.game_active and not st.session_state.game_won and not st.session_state.game_lost:
        st.info("👈 Start a new game from the sidebar!")
        return
    
    # Display game state
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Word display
        display_word = " ".join(st.session_state.masked_word.upper())
        st.markdown(f'<p class="word-display">{display_word}</p>', unsafe_allow_html=True)
        
        # Game status
        if st.session_state.game_won:
            winner = "AI WON" if st.session_state.play_mode == "AI" else "YOU WON"
            st.markdown(f'<p class="game-over-win">🎉 {winner}! 🎉<br>The word was: {st.session_state.word.upper()}</p>', 
                       unsafe_allow_html=True)
            st.balloons()
        elif st.session_state.game_lost:
            loser = "AI LOST" if st.session_state.play_mode == "AI" else "GAME OVER"
            st.markdown(f'<p class="game-over-lose">😔 {loser}!<br>The word was: {st.session_state.word.upper()}</p>', 
                       unsafe_allow_html=True)
        
        # AI Auto-Play or Human controls
        if st.session_state.game_active:
            if st.session_state.play_mode == "AI":
                # AI is playing - show next move button
                st.markdown("### 🤖 AI is Playing...")
                
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    if st.button("▶️ AI Make Next Move", use_container_width=True, type="primary"):
                        mode = "HMM" if "HMM" in st.session_state.game_mode else "DQN"
                        result, letter = ai_make_move(mode)
                        if result:
                            st.success(f"AI guessed: **{letter.upper()}** - {result}")
                        st.rerun()
                
                with col_b:
                    if st.button("⏩ Auto-Complete Game", use_container_width=True):
                        mode = "HMM" if "HMM" in st.session_state.game_mode else "DQN"
                        moves = []
                        while st.session_state.game_active:
                            result, letter = ai_make_move(mode)
                            if result and letter:
                                moves.append((letter, result))
                            else:
                                break
                        
                        # Show all moves
                        st.info(f"AI made {len(moves)} moves")
                        st.rerun()
            else:
                # Human is playing - show letter buttons
                st.markdown("### Select a Letter:")
                
                # Create 3 rows of letter buttons
                alphabet = 'abcdefghijklmnopqrstuvwxyz'
                rows = [alphabet[:9], alphabet[9:18], alphabet[18:]]
                
                for row in rows:
                    cols = st.columns(len(row))
                    for i, letter in enumerate(row):
                        with cols[i]:
                            disabled = letter in st.session_state.guessed_letters
                            if st.button(letter.upper(), key=f"btn_{letter}", 
                                       disabled=disabled, use_container_width=True):
                                result = make_guess(letter)
                                st.rerun()
    
    with col2:
        # Hangman drawing
        st.text(draw_hangman(st.session_state.wrong_guesses))
        
        # Lives remaining
        lives = st.session_state.max_wrong - st.session_state.wrong_guesses
        st.markdown(f"### ❤️ Lives: {lives}/{st.session_state.max_wrong}")
        
        # Progress bar
        st.progress((st.session_state.max_wrong - st.session_state.wrong_guesses) / st.session_state.max_wrong)
        
        st.markdown("---")
        
        # Guessed letters
        if st.session_state.guessed_letters:
            st.markdown("### 📝 Guessed Letters:")
            guessed_sorted = sorted(list(st.session_state.guessed_letters))
            st.write(" ".join([l.upper() for l in guessed_sorted]))
        
        st.markdown("---")
        
        # Show current play mode
        if st.session_state.play_mode == "AI":
            st.info("🤖 AI is playing this game")
        else:
            st.info("👤 You are playing this game")
        
        st.markdown("---")
        
        # AI Hint (only for human players with AI assistance)
        if st.session_state.game_active and st.session_state.play_mode == "Human" and st.session_state.game_mode != "No AI (Pure Human)":
            if st.button("💡 Get AI Hint", use_container_width=True):
                mode = "HMM" if "HMM" in st.session_state.game_mode else "DQN"
                hints = get_ai_hint(mode)
                
                st.markdown("### 🤖 AI Suggests:")
                if hints:
                    if mode == "HMM":
                        for letter, prob in hints:
                            st.write(f"**{letter.upper()}** - {prob:.1%} confidence")
                    else:
                        st.write(f"**{hints[0][0].upper()}** - Best choice")
                else:
                    st.write("No suggestions available")


if __name__ == "__main__":
    main()
