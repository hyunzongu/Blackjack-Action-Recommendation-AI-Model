#초기에는 탐험을 많이하지만 시간이 지남에 따라 탐험이 감소하도록 앱실론값을 감소시킴
#학습 횟수를 더 늘려서 Q 함수를 더 정확하게 업데이트함


import gym
import numpy as np
import pandas as pd
import random
from collections import defaultdict

env = gym.make('Blackjack-v1')

Q = defaultdict(float) #상태와 액션 쌍의 가치를 나타내는 Q함수를 초기화
total_return = defaultdict(float) #얻은 총 리턴을 저장하는 딕셔너리 초기화
N = defaultdict(int) #상태와 액션 쌍의 방문된 횟수를 저장하는 N함수를 초기화

#강화 학습에서 탐험과 활용 사이의 균형을 유지하기 위해 사용하는 전략 중 하나
def epsilon_greedy(state, Q, epsilon):
    #0이랑 1사이에서 무작위로 선택한 값이 입실론 확률보다 작으면 무작위로 액션(현재 0.3)
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        # Q는 (state, action) : value 로 이루어진 Dictionary 이기 때문에
        # key를 (state, action) 으로 주어
        # 현 state에서 가장 높은 value를 가진 key를 찾는다.
        #무작위값이 입실론확률보다 크면 Q를 기반으로 가치가 가장 높은 액션 선택
        return max(range(env.action_space.n), key=lambda x: Q[(state, x)])


#블랙잭 게임 진행
def generate_episode(Q, epsilon, surrender_probability):
    episode = []
    state = env.reset()

    # time step 만큼 step을 진행시킴
    for i in range(num_timesteps):
        # epsilon greedy로 action 추출
        action = epsilon_greedy(state, Q, epsilon)

        #surrender 확률을 설정하여 surrender 결정
        #0과 1사이 랜덤한 값을 생성하여 이 값이 surrender확률(0.1)보다 작으면 서렌더
        if action == surrender_action and random.uniform(0, 1) < surrender_probability:
            episode.append((state, action, -0.5))  # 보상을 -0.5로 설정
            break  # surrender 시에는 에피소드 종료
        else:
     #new_state: 새로운 게임상태, reward: 액션에 대한 보상, done: 게임종료 여부. _:추가정보
            next_state, reward, done, _ = env.step(action)

        episode.append((state, action, reward))
        if done:
            break
        state = next_state

    return episode

def update_q_value(Q, total_return, N, state, action, rewards):
    #Q: Q 함수를 나타내는 딕셔너리 (state, action) 쌍을 키로 사용
    #해당 상태에서 해당 액션을 선택했을 때의 가치를 저장
    #total_return: 각 상태-액션 쌍에 대한 총 누적 보상을 저장하는 딕셔너리
    #N: 각 상태-액션 쌍이 선택된 횟수를 저장하는 딕셔너리
    #state: 현재 상태
    #action: 현재 액션
    #rewards: 현재 스텝에서부터 에피소드가 끝날 때까지 받은 보상들의 리스트
    #G: 현재 스텝에서부터 에피소드가 끝날 때까지 받은 보상들의 총합
    G = sum(rewards)
    #상태, 액션 쌍의 총 누적 보상에 현재 에피소드에서 받은 총 보상을 더함
    total_return[(state, action)] += G
    #상태-액션 쌍이 선택된 횟수를 1 증가시킴
    N[(state, action)] += 1
    #Q함수를 상태,액션 쌍의 가치를 총 누적 보상과 선택된 횟수를 나누어 계산한 평균값으로 조정
    Q[(state, action)] = total_return[(state, action)] / N[(state, action)]

num_timesteps = 50 #각 에피소드에서 수행할 수 있는 최대 스탭
num_episodes = 5000000 #학습 횟수
epsilon = 0.3 #탐험을 위한 확률을 나타냄
epsilon_decay = 0.9999 #학습이 진행될수록 epsilon값을 감소시킴, 더 가치가 높은 액션선택
min_epsilon = 0.05 #epsilon의 최솟값
surrender_action = 2  # surrender에 해당하는 액션 코드
surrender_probability = 0.1  # surrender 확률

for i in range(num_episodes):
    #에피소드마다 epsilon 값을 업데이트, 시간이 지날 수록 더 낮은 탐험
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    #에피소드 생성
    episode = generate_episode(Q, epsilon, surrender_probability)
    # 에피소드에서 등장한 모든 상태,액션 쌍을 저장리스트
    all_state_action_pairs = []
    #받은 보상 저장리스트
    rewards = []

    #현재 에피소드의 각 스텝에 대해 반복
    for t, (state, action, reward) in enumerate(episode):
        #같은 상태에서 같은 액션을 중복 선택 방지를 위함
        if not (state, action) in all_state_action_pairs[:t]:
            #현재 상태, 액션 쌍 추가
            all_state_action_pairs.append((state, action))
            #현재 받은 보상 추가
            rewards.append(reward)
            #Q함수 업데이트하는 함수 호출
            update_q_value(Q, total_return, N, state, action, rewards)

# Q 함수에서 추정된 상태, 액션 쌍의 가치를 DataFrame으로 변환
df = pd.DataFrame(Q.items(), columns=['state_action_pair', "Q_value"])

# 상태, 액션 쌍에 대한 Q 함수를 기반으로한 제안하는 함수
def suggest_action_and_q_value(state, df):
    # 각 액션에 대한 코드 정의
    stay_action = 0
    hit_action = 1
    surrender_action = 2

    # 주어진 상태와 액션에 대한 상태, 액션 쌍 정의
    state_action_pair_stay = (state, stay_action)
    state_action_pair_hit = (state, hit_action)
    state_action_pair_surrender = (state, surrender_action)

    # DataFrame에 있는 상태,액션 쌍을 리스트로 변환
    state_action_pairs = [tuple(pair) for pair in df['state_action_pair']]

    stay_value = 0
    hit_value = 0
    surrender_value = 0

   # 'stay', 'hit', 'surrender' 액션이 Q 함수에 있는지 확인
    if state_action_pair_stay in state_action_pairs and state_action_pair_hit in state_action_pairs:
        # 'stay', 'hit', 'surrender' 각각에 대한 Q 값 추출
        stay_value = df[df['state_action_pair'] == state_action_pair_stay]['Q_value'].values[0]
        hit_value = df[df['state_action_pair'] == state_action_pair_hit]['Q_value'].values[0]

        # 'surrender'에 대한 Q 값이 있다면 추출, 없으면 -0.3으로 설정
        # 'hit'와 'stay'에 대한 벨류가 둘 다 좋지 않을 때 서렌더를 하기위해 -0.3으로 설정
        if state_action_pair_surrender in state_action_pairs:
            surrender_value = df[df['state_action_pair'] == state_action_pair_surrender]['Q_value'].values[0]
        else:
            surrender_value = -0.3

        # 'stay', 'hit', 'surrender' 중에서 최댓값을 가진 액션 선택
        # 'stay', 'hit' 가  'surrender' 보다 벨류값이 작다면 surrender 선택
        suggested_action = 'stay' if stay_value >= hit_value and stay_value >= surrender_value else \
            ('hit' if hit_value >= stay_value and hit_value >= surrender_value else 'surrender')
    else:
        # 'stay', 'hit'만 존재하면 'stay' 선택
        suggested_action = 'stay'

    # 추천된 액션과 해당 액션의 Q 값 중 최댓값 반환
    return suggested_action, max(stay_value, hit_value, surrender_value)

# 사용법 : (나의 카드 두 장의 합, 딜러 카드 한 장, Ace를 11로 썼는가의 여부(0:거짓, 1:참)
current_state =  (14,2,0)

suggested_action, q_value = suggest_action_and_q_value(current_state, df)
print(f"제안된 행동: {suggested_action}, 최대 Q 값: {q_value}")
