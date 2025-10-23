import gym
import random
import pickle
from collections import defaultdict

env = gym.make('Blackjack-v1')

Q = defaultdict(float)
total_return = defaultdict(float)
N = defaultdict(int)

def epsilon_greedy(state, Q, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(range(env.action_space.n), key=lambda x: Q[(state, x)])


def generate_episode(Q, epsilon, surrender_probability, num_timesteps, surrender_action):
    episode = []
    state = env.reset()
    for i in range(num_timesteps):
        action = epsilon_greedy(state, Q, epsilon)
        if action == surrender_action and random.uniform(0, 1) < surrender_probability:
            episode.append((state, action, -0.5))
            break
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state
    return episode


def update_q_value(Q, total_return, N, state, action, rewards):
    G = sum(rewards)
    total_return[(state, action)] += G
    N[(state, action)] += 1
    Q[(state, action)] = total_return[(state, action)] / N[(state, action)]


def train(num_episodes=500000, num_timesteps=50, epsilon=0.3, epsilon_decay=0.9999,
          min_epsilon=0.05, surrender_action=2, surrender_probability=0.1):

    for i in range(num_episodes):
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode = generate_episode(Q, epsilon, surrender_probability, num_timesteps, surrender_action)
        all_state_action_pairs = []
        rewards = []
        for t, (state, action, reward) in enumerate(episode):
            if not (state, action) in all_state_action_pairs[:t]:
                all_state_action_pairs.append((state, action))
                rewards.append(reward)
                update_q_value(Q, total_return, N, state, action, rewards)

    # 학습 완료 후 Q 저장
    with open("trained_Q.pkl", "wb") as f:
        pickle.dump(dict(Q), f)
    print("✅ Q-learning 학습 완료 및 trained_Q.pkl 저장됨")


if __name__ == "__main__":
    train()
