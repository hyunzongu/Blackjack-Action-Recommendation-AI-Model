# Blackjack-Action-Recommendation-AI-Model

### 개요
블랙잭 게임 중, 나의 카드 두 장의 합, 딜러 카드 한 장, A를 11로 썼는지 여부로 본인이 해야 할 행동을 추천해주는 모델(ex | stay, hit , surrender) 구현

### 동작 방식
**Q-learning**과 **Epsilon-Greedy** 정책을 이용하여 블랙잭 게임의 최적 전략을 학습하며, 시간이 지남에 따라 탐험을 줄이고 더 많이 활용하여 최적의 전략을 찾아감

**Q-learning** : Q 러닝은 주어진 상태에서 주어진 행동을 수행하는 것이 가져다 줄 효용의 기대값을 예측하는 함수인 Q 함수를 학습함으로써 최적의 정책을 학습
**Q 함수** :  상태-행동 가치 함수라고도 불리며 상태와 행동을 입력했을 때 이에 대한 가치를 출력으로 주는 함수
**Epsilon-greedy** :강화 학습에서 탐험과 활용 사이의 균형을 유지하기 위해 사용하는 전략

### 동작 결과
실제 카드로 나의 카드 두 장의 합, 딜러 카드 한 장 등 입력값을 입력해보며 블랙잭 게임을 진행한 결과
주어진 금액 500에서 9승 2무 9패(surrender 4회)로 545, +45의 결과 도출


## 전체 동작 흐름
### 한 번만 실행 (학습 단계)
python train_model.py


여기서 실제 Q-learning 강화학습이 수행됩니다.

수백만 번의 게임 시뮬레이션을 돌려서
(state, action) 조합별 Q 값을 계산합니다.

학습이 끝나면 trained_Q.pkl 파일로 저장됩니다.

📂 project/
 ├── train_model.py
 ├── q_agent.py
 ├── main.py
 └── trained_Q.pkl  ← 여기 저장됨

### 그 이후부터는
python main.py


학습 결과(trained_Q.pkl)를 불러와서 바로 사용합니다.

즉, AI는 이미 학습된 상태로 동작하기 때문에
다시 훈련할 필요가 없습니다.

단순히 current_state만 입력하면 즉시 추천 행동이 나옵니다.

current_state = (14, 2, 0)
제안된 행동: stay, 최대 Q 값: 0.45

### 필요한 경우에만 재학습

만약 학습 파라미터를 바꾸거나
게임 환경을 수정했을 때만 다시 학습시키면 됩니다.

예를 들어 

num_episodes를 5,000,000 → 10,000,000으로 늘리거나

surrender_probability를 바꾸거나

환경을 다른 Gym 환경으로 교체할 때

그때만 다시 train_model.py 실행해서
새로운 trained_Q.pkl을 만들어주면 됩니다.
