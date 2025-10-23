# 🃏 Blackjack Action Recommendation Model

## 📘 개요
블랙잭 게임 중 **나의 카드 두 장의 합**, **딜러의 카드 한 장**, **A를 11로 사용 여부**를 기반으로  
가장 유리한 행동(`stay`, `hit`, `surrender`)을 추천해주는 모델입니다.

---

## ⚙️ 동작 원리
본 모델은 **Q-learning** 기반의 강화학습 알고리즘을 사용하여,  
시간이 지날수록 **탐험(Exploration)**은 줄이고 **활용(Exploitation)**을 늘리며  
점차 최적의 전략을 학습합니다.

- **Q-learning**  
  상태(state)-행동(action) 쌍의 가치를 예측하는 **Q 함수**를 학습하여 최적의 정책을 도출  
- **Q 함수 (State-Action Value Function)**  
  특정 상태에서 특정 행동을 취했을 때의 기대 가치를 나타내는 함수  
- **Epsilon-Greedy 정책**  
  일정 확률(ε)로 무작위 탐험을 수행하며, 학습이 진행될수록 ε 값을 점진적으로 감소시켜  
  더 많은 활용을 수행하도록 유도  

---

## 🎯 동작 결과 테스트
실제 카드 입력값으로 블랙잭 게임을 진행한 결과:

- 초기 금액 **500 → 최종 545**  
- **9승 2무 9패 (서렌더 4회)**  
- **총 +45 이익 달성**

---

## 🧩 전체 구조

📂 project/

├── train_model.py # 최초 Q-learning 학습 수행 및 Q값 저장

├── q_agent.py # 학습된 Q값 로드 및 행동 추천 함수

├── main.py # 메인 실행

└── trained_Q.pkl # 학습된 Q값 저장 파일 - train_model.py 실행 시 생성

---

## 🚀 실행 방법

### 1️⃣ 학습 단계 (최초 1회만 실행)
```bash
python train_model.py
수백만 번의 블랙잭 시뮬레이션을 통해
(state, action) 조합별 Q 값을 학습합니다.

학습 완료 후 결과가 trained_Q.pkl로 저장됩니다.
```

### 2️⃣ 학습된 모델 실행
```bash
python main.py
학습 결과(trained_Q.pkl)를 불러와 바로 동작합니다.

이미 학습된 모델이므로 재학습이 필요 없습니다.

예시:

current_state = (14, 2, 0)
# (내 카드 합, 딜러 카드, A를 11로 사용했는가)
출력:

제안된 행동: stay
최대 Q 값: 0.45
```

### 3️⃣ 재학습이 필요한 경우
```bash
다음과 같이 환경이나 파라미터를 변경한 경우에만 train_model.py를 다시 실행합니다.

num_episodes (학습 횟수) 변경
→ 예: 5,000,000 → 10,000,000

surrender_probability (서렌더 확률) 변경

Gym 환경(Blackjack-v1 → 다른 환경) 교체

새로운 조건에서 재학습 후,
업데이트된 trained_Q.pkl을 생성하면 됩니다.
```

### 🧠 핵심 포인트
```bash
Epsilon 감소 스케줄을 통해 초기에는 다양한 탐험을 수행하고,
학습이 진행될수록 안정적으로 최적 행동을 선택하도록 유도

상태-행동 가치 기반 의사결정(Q-Value)
→ 경험적 데이터에 따라 stay, hit, surrender 중 최선의 선택 추천

모듈 분리 설계

학습(train_model.py)

모델 로드(q_agent.py)

실행(main.py)
```

### 🏁 예시 시각화 (간략 개념도)
```bash
┌──────────────┐
│ Environment  │ ← Gym(Blackjack-v1)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Q-Learning  │  ← 학습 및 Q값 갱신
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Q-Table    │  ← trained_Q.pkl 저장
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Action Guide │  ← stay / hit / surrender 추천
└──────────────┘
```
💬 “단순한 규칙 기반이 아닌, 경험적 학습을 통해 스스로 전략을 찾아가는 블랙잭 추천 모델입니다.”
