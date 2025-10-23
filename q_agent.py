import pickle
import pandas as pd

def load_q_values(path="trained_Q.pkl"):
    with open(path, "rb") as f:
        Q = pickle.load(f)
    df = pd.DataFrame(Q.items(), columns=['state_action_pair', 'Q_value'])
    return df


def suggest_action_and_q_value(state, df):
    stay_action, hit_action, surrender_action = 0, 1, 2
    state_action_pair_stay = (state, stay_action)
    state_action_pair_hit = (state, hit_action)
    state_action_pair_surrender = (state, surrender_action)

    state_action_pairs = [tuple(pair) for pair in df['state_action_pair']]
    stay_value = hit_value = surrender_value = 0

    if state_action_pair_stay in state_action_pairs and state_action_pair_hit in state_action_pairs:
        stay_value = df[df['state_action_pair'] == state_action_pair_stay]['Q_value'].values[0]
        hit_value = df[df['state_action_pair'] == state_action_pair_hit]['Q_value'].values[0]
        if state_action_pair_surrender in state_action_pairs:
            surrender_value = df[df['state_action_pair'] == state_action_pair_surrender]['Q_value'].values[0]
        else:
            surrender_value = -0.3

        suggested_action = 'stay' if stay_value >= hit_value and stay_value >= surrender_value else \
            ('hit' if hit_value >= stay_value and hit_value >= surrender_value else 'surrender')
    else:
        suggested_action = 'stay'

    return suggested_action, max(stay_value, hit_value, surrender_value)
