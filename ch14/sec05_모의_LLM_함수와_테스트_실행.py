"""
으뜸 딥러닝 — 14장 05절
모의 LLM 함수와 테스트 실행
"""

def mock_llm(prompt):
    """Rule-based mock that simulates ReAct reasoning."""
    if "Observation" not in prompt:
        return (
            "Thought: I need to find the first author of "
            "the Transformer paper.\n"
            "Action: search[transformer]"
        )
    elif "Ashish Vaswani" in prompt and "India" not in prompt:
        return (
            "Thought: The first author is Ashish Vaswani. "
            "I need to find his birth country.\n"
            "Action: search[Ashish Vaswani]"
        )
    elif "India" in prompt and "New Delhi" not in prompt:
        return (
            "Thought: He was born in India. "
            "I need to find the capital of India.\n"
            "Action: search[India]"
        )
    else:
        return (
            "Thought: India's capital is New Delhi. "
            "I have all the information.\n"
            "Answer: New Delhi"
        )

# Run the agent
question = ("What is the capital of the birth country "
            "of the first author of the Transformer paper?")
answer, trace = react_agent(question, mock_llm)

print(f"Final answer: {answer}\n")
for t in trace:
    print(t)
# --- Step 1 ---
# Thought: I need to find the first author ...
# Action: search[transformer]
# Observation: Transformer was proposed by Vaswani ...
# --- Step 2 ---
# Thought: The first author is Ashish Vaswani ...
# Action: search[Ashish Vaswani]
# Observation: Ashish Vaswani is ... born in India.
# --- Step 3 ---
# Thought: He was born in India ...
# Action: search[India]
# Observation: India is ... capital is New Delhi ...
# --- Step 4 ---
# Thought: India's capital is New Delhi ...
# Answer: New Delhi
