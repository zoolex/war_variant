import random
import sys
import pickle
import os
from collections import defaultdict

# Utility Functions
def generate_hand():
    return [random.randint(2, 14) for _ in range(13)]

def card_name(v):
    names = {11: "J", 12: "Q", 13: "K", 14: "A"}
    return names.get(v, str(v))

# Random Opponent AI
class RandomAI:
    """Simple AI that plays randomly - used as opponent for training"""
    
    def select_card(self, hand):
        """Randomly select a card"""
        card = random.choice(hand)
        hand.remove(card)
        return card
    
    def select_bet(self):
        """Random bet between 50-350"""
        return random.randint(50, 350)
    
    def decide_call_or_fold(self, card, bet_size):
        """Random decision with slight bias toward calling with strong cards"""
        if card >= 12:  # Strong cards (Q, K, A)
            return "call" if random.random() < 0.8 else "fold"
        elif card >= 9:  # Medium cards
            return "call" if random.random() < 0.5 else "fold"
        else:  # Weak cards
            return "call" if random.random() < 0.3 else "fold"


# Q-Learning AI Strategy
class QLearningAI:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.2):
        # Q-Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Q-tables: state -> action -> value
        self.q_card_selection = defaultdict(lambda: defaultdict(float))
        self.q_betting = defaultdict(lambda: defaultdict(float))
        self.q_call_fold = defaultdict(lambda: defaultdict(float))
        
        # Experience tracking
        self.last_state = None
        self.last_action = None
        self.last_action_type = None
        
        # Statistics
        self.games_played = 0
        self.total_reward = 0
        self.wins = 0
        self.losses = 0
        
    def get_state(self, hand, round_num, balance_diff, cards_remaining, opponent_bet=None):
        """Convert game state to a hashable state representation"""
        sorted_hand = sorted(hand)
        
        # Discretize continuous values for state representation
        balance_category = (
            'losing_bad' if balance_diff < -300 else
            'losing' if balance_diff < -100 else
            'even' if abs(balance_diff) <= 100 else
            'winning' if balance_diff < 300 else
            'winning_big'
        )
        
        cards_category = (
            'early' if cards_remaining > 9 else
            'mid' if cards_remaining > 5 else
            'late'
        )
        
        # Hand strength (average of top 3 cards)
        top_cards = sorted_hand[-3:] if len(sorted_hand) >= 3 else sorted_hand
        hand_strength = sum(top_cards) / len(top_cards)
        strength_category = (
            'weak' if hand_strength < 8 else
            'medium' if hand_strength < 11 else
            'strong'
        )
        
        if opponent_bet is not None:
            bet_category = (
                'small' if opponent_bet <= 100 else
                'medium' if opponent_bet <= 200 else
                'large'
            )
            return (balance_category, cards_category, strength_category, bet_category)
        
        return (balance_category, cards_category, strength_category)
    
    def select_card(self, hand, round_num, balance_diff, cards_remaining):
        # Use Q-learning to select a card
        state = self.get_state(hand, round_num, balance_diff, cards_remaining)
        sorted_hand = sorted(hand)
        
        # Map cards to action categories (low, medium, high)
        actions = []
        for card in sorted_hand:
            if card <= 8:
                action_category = 'low'
            elif card <= 11:
                action_category = 'medium'
            else:
                action_category = 'high'
            actions.append((card, action_category))
        
        # Epsilon-greedy: explore vs exploit
        if random.random() < self.epsilon:
            chosen_card, action_category = random.choice(actions)
        else:
            best_value = float('-inf')
            best_action = actions[0]
            
            for card, action_category in actions:
                q_value = self.q_card_selection[state][action_category]
                if q_value > best_value:
                    best_value = q_value
                    best_action = (card, action_category)
            
            chosen_card, action_category = best_action
        
        # Store for learning
        self.last_state = state
        self.last_action = action_category
        self.last_action_type = 'card_selection'
        
        hand.remove(chosen_card)
        return chosen_card
    
    def select_bet(self, card, balance_diff, cards_remaining):
        # Use Q-learning to determine bet size
        card_strength = 'weak' if card < 8 else 'medium' if card < 11 else 'strong'
        balance_category = 'losing' if balance_diff < -200 else 'even' if abs(balance_diff) <= 200 else 'winning'
        cards_category = 'early' if cards_remaining > 9 else 'mid' if cards_remaining > 5 else 'late'
        
        state = (card_strength, balance_category, cards_category)
        
        bet_actions = ['small', 'medium', 'large', 'bluff']
        
        if random.random() < self.epsilon:
            action = random.choice(bet_actions)
        else:
            action = max(bet_actions, key=lambda a: self.q_betting[state][a])
        
        self.last_state = state
        self.last_action = action
        self.last_action_type = 'betting'
        
        bet_mapping = {
            'small': random.randint(50, 120),
            'medium': random.randint(120, 200),
            'large': random.randint(200, 350),
            'bluff': random.randint(250, 400)
        }
        
        return bet_mapping[action]
    
    def decide_call_or_fold(self, card, bet_size, balance_diff, cards_remaining):
        # Use Q-learning to decide call or fold
        card_strength = 'weak' if card < 8 else 'medium' if card < 11 else 'strong'
        bet_category = 'small' if bet_size <= 100 else 'medium' if bet_size <= 200 else 'large'
        balance_category = 'losing' if balance_diff < -200 else 'even' if abs(balance_diff) <= 200 else 'winning'
        cards_category = 'early' if cards_remaining > 9 else 'mid' if cards_remaining > 5 else 'late'
        
        state = (card_strength, bet_category, balance_category, cards_category)
        
        actions = ['call', 'fold']
        
        if random.random() < self.epsilon:
            action = random.choice(actions)
        else:
            action = max(actions, key=lambda a: self.q_call_fold[state][a])
        
        self.last_state = state
        self.last_action = action
        self.last_action_type = 'call_fold'
        
        return action
    
    def update_q_value(self, reward, next_state=None):
        # Update Q-value based on received reward
        if self.last_state is None or self.last_action is None:
            return
        
        if self.last_action_type == 'card_selection':
            q_table = self.q_card_selection
        elif self.last_action_type == 'betting':
            q_table = self.q_betting
        else:
            q_table = self.q_call_fold
        
        current_q = q_table[self.last_state][self.last_action]
        
        if next_state is not None:
            future_value = max(q_table[next_state].values()) if q_table[next_state] else 0
        else:
            future_value = 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * future_value - current_q
        )
        
        q_table[self.last_state][self.last_action] = new_q
        
        self.total_reward += reward
    
    def learn_from_outcome(self, won, amount_won_or_lost):
        # Learn from round outcome
        reward = amount_won_or_lost if won else -amount_won_or_lost
        reward = max(-500, min(500, reward)) / 100
        self.update_q_value(reward)
    
    def save_model(self, filename='war_ai_model.pkl'):
        # Save learned Q-tables to file
        model_data = {
            'q_card_selection': dict(self.q_card_selection),
            'q_betting': dict(self.q_betting),
            'q_call_fold': dict(self.q_call_fold),
            'games_played': self.games_played,
            'total_reward': self.total_reward,
            'wins': self.wins,
            'losses': self.losses
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='war_ai_model.pkl'):
        # Load learned Q-tables from file
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            self.q_card_selection = defaultdict(lambda: defaultdict(float), model_data['q_card_selection'])
            self.q_betting = defaultdict(lambda: defaultdict(float), model_data['q_betting'])
            self.q_call_fold = defaultdict(lambda: defaultdict(float), model_data['q_call_fold'])
            self.games_played = model_data['games_played']
            self.total_reward = model_data['total_reward']
            self.wins = model_data.get('wins', 0)
            self.losses = model_data.get('losses', 0)
            return True
        return False


# Game Loop (AI vs AI)
def play_game_ai_vs_ai(learning_ai, opponent_ai, verbose=False):
    # Play a game between learning AI and random opponent
    learning_hand = generate_hand()
    opponent_hand = generate_hand()

    learning_balance = 1000
    opponent_balance = 1000
    buyin = 50

    for round_num in range(1, 14):
        learning_bets_first = (round_num % 2 == 1)
        balance_diff = learning_balance - opponent_balance
        cards_remaining = len(learning_hand)

        if verbose:
            print(f"\n--- Round {round_num} ---")
            print(f"Learning AI: {learning_balance} | Opponent: {opponent_balance}")

        if learning_bets_first:
            # Learning AI bets first
            learning_card = learning_ai.select_card(learning_hand, round_num, balance_diff, cards_remaining)
            learning_bet = learning_ai.select_bet(learning_card, balance_diff, cards_remaining)

            opponent_decision = opponent_ai.decide_call_or_fold(
                max(opponent_hand) if opponent_hand else 2,  # Opponent peeks at best card
                learning_bet
            )

            if opponent_decision == "fold":
                learning_balance += buyin
                opponent_balance -= buyin
                learning_ai.learn_from_outcome(True, buyin)
                if verbose:
                    print(f"Learning AI bet {learning_bet}, Opponent folded")
                continue

            opponent_card = opponent_ai.select_card(opponent_hand)

            if verbose:
                print(f"Learning AI: {card_name(learning_card)} | Opponent: {card_name(opponent_card)}")

            if learning_card > opponent_card:
                total = learning_bet + buyin * 2
                learning_balance += total
                opponent_balance -= learning_bet + buyin
                learning_ai.learn_from_outcome(True, learning_bet + buyin)
                if verbose:
                    print("Learning AI wins!")
            elif opponent_card > learning_card:
                total = learning_bet + buyin * 2
                opponent_balance += total
                learning_balance -= learning_bet + buyin
                learning_ai.learn_from_outcome(False, learning_bet + buyin)
                if verbose:
                    print("Opponent wins!")
            else:
                learning_balance -= buyin
                opponent_balance -= buyin
                learning_ai.learn_from_outcome(False, buyin)
                if verbose:
                    print("Tie!")

        else:
            # Opponent bets first
            opponent_card = opponent_ai.select_card(opponent_hand)
            opponent_bet = opponent_ai.select_bet()

            learning_card = learning_ai.select_card(learning_hand, round_num, -balance_diff, cards_remaining)
            learning_decision = learning_ai.decide_call_or_fold(learning_card, opponent_bet, balance_diff, cards_remaining)

            if learning_decision == "fold":
                opponent_balance += buyin
                learning_balance -= buyin
                learning_ai.learn_from_outcome(False, buyin)
                if verbose:
                    print(f"Opponent bet {opponent_bet}, Learning AI folded")
                continue

            if verbose:
                print(f"Learning AI: {card_name(learning_card)} | Opponent: {card_name(opponent_card)}")

            if learning_card > opponent_card:
                total = opponent_bet + buyin * 2
                learning_balance += total
                opponent_balance -= opponent_bet + buyin
                learning_ai.learn_from_outcome(True, opponent_bet + buyin)
                if verbose:
                    print("Learning AI wins!")
            elif opponent_card > learning_card:
                total = opponent_bet + buyin * 2
                opponent_balance += total
                learning_balance -= opponent_bet + buyin
                learning_ai.learn_from_outcome(False, opponent_bet + buyin)
                if verbose:
                    print("Opponent wins!")
            else:
                learning_balance -= buyin
                opponent_balance -= buyin
                learning_ai.learn_from_outcome(False, buyin)
                if verbose:
                    print("Tie!")

        if learning_balance <= 0 or opponent_balance <= 0:
            break

    learning_ai.games_played += 1
    won = learning_balance > opponent_balance
    
    if won:
        learning_ai.wins += 1
    else:
        learning_ai.losses += 1
    
    return won, learning_balance, opponent_balance

# Training Function
def train_ai(num_games=100, verbose_every=10, save_every=20):
    # Train the AI by playing multiple games against random opponent
    print("=" * 50)
    print("TRAINING MODE: AI vs Random Opponent")
    print("=" * 50)
    
    learning_ai = QLearningAI()
    opponent_ai = RandomAI()
    
    # Try to load existing model
    if learning_ai.load_model():
        print(f"Continuing training from {learning_ai.games_played} games")
        print(f"Current record: {learning_ai.wins}W - {learning_ai.losses}L")
    
    print(f"\nTraining for {num_games} games...\n")
    
    wins = 0
    total_learning_balance = 0
    total_opponent_balance = 0
    
    for i in range(num_games):
        verbose = (i % verbose_every == 0)
        
        won, learning_bal, opponent_bal = play_game_ai_vs_ai(learning_ai, opponent_ai, verbose)
        
        if won:
            wins += 1
        
        total_learning_balance += learning_bal
        total_opponent_balance += opponent_bal
        
        if (i + 1) % verbose_every == 0:
            win_rate = wins / (i + 1) * 100
            avg_balance = total_learning_balance / (i + 1)
            print(f"\nProgress: {i + 1}/{num_games} games")
            print(f"Win rate: {win_rate:.1f}% ({wins}/{i + 1})")
            print(f"Avg final balance: {avg_balance:.0f}")
            print(f"Total reward: {learning_ai.total_reward:.2f}")
        
        if (i + 1) % save_every == 0:
            learning_ai.save_model()
    
    # Final save
    learning_ai.save_model()
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Total games: {learning_ai.games_played}")
    print(f"Overall record: {learning_ai.wins}W - {learning_ai.losses}L")
    print(f"This session: {wins}W - {num_games - wins}L ({wins/num_games*100:.1f}% win rate)")
    print(f"Avg final balance: {total_learning_balance/num_games:.0f}")
    print(f"Total reward accumulated: {learning_ai.total_reward:.2f}")

# Main Entry Point
def main():
    print("\n" + "=" * 50)
    print("WAR GAME - Q-LEARNING AI TRAINER")
    print("=" * 50)
    print("\nThis will train an AI to play the card game War")
    print("by having it play against a random opponent.\n")
    
    while True:
        try:
            num_games = input("How many games to train? (default 100): ").strip()
            num_games = int(num_games) if num_games else 100
            break
        except ValueError:
            print("Please enter a valid number")
    
    train_ai(num_games=num_games)
    
    print("\nThe AI has been trained and saved to 'war_ai_model.pkl'")

if __name__ == "__main__":
    main()