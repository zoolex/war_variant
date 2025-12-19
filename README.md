# War Variant

**Alex Zhou**

# What problem is being solved?

The goal of this project is to design an autonomous agent that can learn to play a competitve, turn based card game using reinforcement learning.

Specifically, the problem is to enable an AI to:
- Decide which card to play from their hidden hand
- Decide how much to bet under uncertainty
- Decide whether to call or fold when facing an opponent's bet

# Game Overview
This is a strategic variant of the classic card game War:
- Each player starts with 13 random cards (values 2-14, where 11-14 = J, Q, K, A)
- Each player also starts with 1,000 coins
- Each round:
    - Player 1 bets first (AI goes first)
    - The other player may call or fold
    - Both players pay a fixed buy-in (50 coins)
    - Both players pick a card to place
    - Higher card wins the round and the pot
- After 13 rounds (or bankruptcy), the higher balance wins

# How to Run

1) Train the AI (Optional)

Run the training script first:
python train_model.py

You will be prompted for the number of training games (default - 100, recommended - >10,000)
- The Ai plays against a randomized opponent
- The model is saved to war_ai_model.pkl

2) Play against the AI

Run the file:
Python main.py

- Type quit at any input to exit immediately
- Displays balances, cards, bets, and outcomes



# AI learning Approach
The AI uses Q-learning, a model-free reinforcement learning algorithm.

The agent learns on optimal policy by interacting with the game environment and receiving rewards based on round outcome. Training can be resumed across sessions, and learned behavior persists.

**State Representation**
The game state is categorized as such:
- Balance status (losing/even/winning)
- Game phase (early/mid/late)
- Hand strength (weak/medium/strong)
- Bet size category (small/medium/large)

**Action Spaces**
- Card Selection (low/medium/high card)
- Betting (small/medium/large/bluff)
- Response (call/fold)

# Example Training Output
<img width="450" height="171" alt="image" src="https://github.com/user-attachments/assets/c0709b28-de89-49f2-a4e4-fbd3a42c3a10" />

# Description of the Software

**Software Components**
The system is divided into two main programs:
1) Training System (train_model.py)
   - AI plays against a randomized opponent
   - Q-tables are updated after every round
   - Learned values are saved
2) Gameplay system (main.py)
   - Loads the trained Q-tables
   - Compete against the AI
   - Uses a low exploration rate to favor learned behavior

# Programming Langaage & Libraries
- Language: Python 3
- Libraries:
  - random
  - pickle
  - collections.defaultdict
  - sys, os
 No external ML libraries are used. ALl learning logic is implemented manually.

# Evaluation
1) Self-play training performance
   - AI trained against a random baseline opponent
   - Win/lose statistics tracked across thousands of games
   - Average final balance
   - Rewards accumulated

# Quantitative Results
- After training, the AI achieves a win rate significantly above 50%
- Average final balance increases with training duration
- Trained Ai consistetly outperforms untrained AI version

# Lessons Learned
- Implementation of game logic using Python
- How to create and implement a reinforcement model
- How to use a randomized opponent to simulate matches
- Q-learning is effective for learning strategic behavior
- Seperation of training and gameplay file to improve maintainability

# Future Improvements
- Increase state granularity (exact card ranks rather than categories of rank)
- Train against adaptive opponents
- Implement Deep Q-Networks
- Add visualizations of learning curves and Q-values
- Implement web-based user interface



