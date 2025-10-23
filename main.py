from q_agent import load_q_values, suggest_action_and_q_value

# 학습된 Q값 로드
df = load_q_values("trained_Q.pkl")

# 예시 상태 입력
current_state = (14, 2, 0)
suggested_action, q_value = suggest_action_and_q_value(current_state, df)
print(f"제안된 행동: {suggested_action}, 최대 Q 값: {q_value}")
