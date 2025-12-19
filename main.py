import random
import sys
import pickle
import os
from collections import defaultdict

# Utility Functions
def safe_input(prompt):
    text = input(prompt)
    if text.strip().lower() == "quit":
        print("\nGame ended.")
        sys.exit()
    return text

def generate_hand():
    return [random.randint(2, 14) for _ in range(13)]

def card_name(v):
    names = {11: "J", 12: "Q", 13: "K", 14: "A"}
    return names.get(v, str(v))


# Load Q-Learning AI
class QLearningAI:
    def __init__(self, epsilon=0.05):  # Lower epsilon = more exploitation
        self.epsilon = epsilon
        self.q_card_selection = defaultdict(lambda: defaultdict(float))
        self.q_betting = defaultdict(lambda: defaultdict(float))
        self.q_call_fold = defaultdict(lambda: defaultdict(float))
        self.games_played = 0
        self.total_reward = 0
        self.wins = 0
        self.losses = 0
        
    def get_state(self, hand, round_num, balance_diff, cards_remaining, opponent_bet=None):
        sorted_hand = sorted(hand)
        
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
        state = self.get_state(hand, round_num, balance_diff, cards_remaining)
        sorted_hand = sorted(hand)
        
        actions = []
        for card in sorted_hand:
            if card <= 8:
                action_category = 'low'
            elif card <= 11:
                action_category = 'medium'
            else:
                action_category = 'high'
            actions.append((card, action_category))
        
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
        
        hand.remove(chosen_card)
        return chosen_card
    
    def select_bet(self, card, balance_diff, cards_remaining):
        card_strength = 'weak' if card < 8 else 'medium' if card < 11 else 'strong'
        balance_category = 'losing' if balance_diff < -200 else 'even' if abs(balance_diff) <= 200 else 'winning'
        cards_category = 'early' if cards_remaining > 9 else 'mid' if cards_remaining > 5 else 'late'
        
        state = (card_strength, balance_category, cards_category)
        bet_actions = ['small', 'medium', 'large', 'bluff']
        
        if random.random() < self.epsilon:
            action = random.choice(bet_actions)
        else:
            action = max(bet_actions, key=lambda a: self.q_betting[state][a])
        
        bet_mapping = {
            'small': random.randint(50, 120),
            'medium': random.randint(120, 200),
            'large': random.randint(200, 350),
            'bluff': random.randint(250, 400)
        }
        
        return bet_mapping[action]
    
    def decide_call_or_fold(self, card, bet_size, balance_diff, cards_remaining):
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
        
        return action
    
    def load_model(self, filename='war_ai_model.pkl'):
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

# Game Loop (Human vs AI)
def play_game(ai_brain):
    player_hand = generate_hand()
    ai_hand = generate_hand()

    player_balance = 1000
    ai_balance = 1000
    buyin = 50
    
    player_folds = 0
    ai_folds = 0

    print("\n==============================")
    print(" WAR VARIANT — YOU vs TRAINED AI")
    print(" Type 'quit' at ANY time to exit")
    print("==============================\n")

    for round_num in range(1, 14):
        ai_bets_first = (round_num % 2 == 1)
        balance_diff = ai_balance - player_balance
        cards_remaining = len(ai_hand)

        print(f"\n========= ROUND {round_num} =========")
        print(f"Your balance: {player_balance}")
        print(f"AI balance: {ai_balance}")
        print(f"Cards remaining: {cards_remaining}")
        print(f"BUY-IN: {buyin} (each player must pay to participate)\n")

        if ai_bets_first:
            # AI BETS FIRST
            print("Your hand:", ", ".join(card_name(c) for c in player_hand))
            
            ai_card = ai_brain.select_card(ai_hand, round_num, balance_diff, cards_remaining)
            ai_bet = ai_brain.select_bet(ai_card, balance_diff, cards_remaining)

            print(f"AI bets: {ai_bet}")

            choice = safe_input("Do you match or fold? (m/f): ").lower().strip()

            print("\n------------------------")

            if choice == "f":
                print("You folded. AI wins the round.")
                print(f"You lost {buyin} buy-in")
                player_balance -= buyin
                ai_balance += buyin
                player_folds += 1
                continue

            print("\nSelect a card to play:")
            print(", ".join(card_name(c) for c in player_hand))

            while True:
                player_choice = safe_input("Your card: ").strip().upper()

                try:
                    if player_choice in ["J", "Q", "K", "A"]:
                        mapping = {"J": 11, "Q": 12, "K": 13, "A": 14}
                        player_card = mapping[player_choice]
                    else:
                        player_card = int(player_choice)

                    if player_card not in player_hand:
                        raise ValueError

                    break
                except:
                    print("Invalid card. Try again or type 'quit' to exit.")

            player_hand.remove(player_card)

            print(f"\nAI played: {card_name(ai_card)}")
            print(f"You played: {card_name(player_card)}\n")

            if player_card > ai_card:
                print("You win the round!")
                total_won = ai_bet + (buyin * 2)
                print(f"You won {total_won} credits total ({ai_bet} bet + {buyin*2} buy-ins)")
                player_balance += ai_bet + buyin*2
                ai_balance -= ai_bet + buyin
            elif ai_card > player_card:
                print("AI wins the round!")
                total_won = ai_bet + (buyin * 2)
                print(f"You lost {total_won} credits total ({ai_bet} bet + {buyin*2} buy-ins)")
                ai_balance += ai_bet + buyin*2
                player_balance -= ai_bet + buyin
            else:
                print("TIE — both players lose their buy-in.")
                print(f"(Each player loses {buyin} buy-in)")
                player_balance -= buyin
                ai_balance -= buyin

        else:
            # PLAYER BETS FIRST
            print("Your hand:", ", ".join(card_name(c) for c in player_hand))
            
            print("\nSelect a card to play:")
            print(", ".join(card_name(c) for c in player_hand))

            while True:
                player_choice = safe_input("Your card: ").strip().upper()

                try:
                    if player_choice in ["J", "Q", "K", "A"]:
                        mapping = {"J": 11, "Q": 12, "K": 13, "A": 14}
                        player_card = mapping[player_choice]
                    else:
                        player_card = int(player_choice)

                    if player_card not in player_hand:
                        raise ValueError

                    break
                except:
                    print("Invalid card. Try again or type 'quit' to exit.")

            player_hand.remove(player_card)

            while True:
                try:
                    player_bet = int(safe_input("Your bet: ").strip())
                    if player_bet < 0:
                        print("Bet must be non-negative.")
                        continue
                    break
                except ValueError:
                    print("Invalid bet. Enter a number.")

            print(f"\nYou bet: {player_bet}")
            
            ai_card = ai_brain.select_card(ai_hand, round_num, balance_diff, cards_remaining)
            ai_decision = ai_brain.decide_call_or_fold(ai_card, player_bet, balance_diff, cards_remaining)

            print(f"AI decides to: {ai_decision.upper()}")

            if ai_decision == "fold":
                print("AI folded. You win the round!")
                print(f"You won {buyin} buy-in from AI")
                player_balance += buyin
                ai_balance -= buyin
                ai_folds += 1
                continue

            print(f"\nAI played: {card_name(ai_card)}")
            print(f"You played: {card_name(player_card)}\n")

            if player_card > ai_card:
                print("You win the round!")
                total_won = player_bet + (buyin * 2)
                print(f"You won {total_won} credits total ({player_bet} bet + {buyin*2} buy-ins)")
                player_balance += player_bet + buyin*2
                ai_balance -= player_bet + buyin
            elif ai_card > player_card:
                print("AI wins the round!")
                total_won = player_bet + (buyin * 2)
                print(f"You lost {total_won} credits total ({player_bet} bet + {buyin*2} buy-ins)")
                ai_balance += player_bet + buyin*2
                player_balance -= player_bet + buyin
            else:
                print("TIE — both players lose their buy-in.")
                print(f"(Each player loses {buyin} buy-in)")
                player_balance -= buyin
                ai_balance -= buyin

        if player_balance <= 0 or ai_balance <= 0:
            break

    print("\n=========== GAME OVER ===========")
    print(f"Final Player Balance: {player_balance}")
    print(f"Final AI Balance: {ai_balance}")
    print(f"\nStatistics:")
    print(f"Player folds: {player_folds}")
    print(f"AI folds: {ai_folds}\n")

    if player_balance > ai_balance:
        print("YOU WIN THE MATCH!")
    elif ai_balance > player_balance:
        print("AI WINS THE MATCH!")
    else:
        print("It's a draw!")

# Main Entry Point
def main():
    ai_brain = QLearningAI(epsilon=0.05)  # 5% exploration for variety
    
    if not ai_brain.load_model():
        print("ERROR: No trained model found!")
        print("Please train the AI first using the training script.")
        sys.exit()
    
    print("\n" + "=" * 50)
    print("TRAINED AI LOADED")
    print("=" * 50)
    print(f"Games trained: {ai_brain.games_played}")
    print(f"Training record: {ai_brain.wins}W - {ai_brain.losses}L")
    if ai_brain.games_played > 0:
        win_rate = ai_brain.wins / ai_brain.games_played * 100
        print(f"Training win rate: {win_rate:.1f}%")
    print("=" * 50)
    
    while True:
        play_game(ai_brain)
        
        choice = safe_input("\nPlay again? (y/n): ").lower().strip()
        if choice != 'y':
            break
    
    print("\nThanks for playing!")

if __name__ == "__main__":
    main()