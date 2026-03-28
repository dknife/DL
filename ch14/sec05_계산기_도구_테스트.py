"""
으뜸 딥러닝 — 14장 05절
계산기 도구 테스트
"""

def mock_llm_calc(prompt):
    """Mock LLM for a calculation task."""
    if "Observation" not in prompt:
        return (
            "Thought: I need to compute 371 * 49. "
            "Let me use the calculator.\n"
            "Action: calculator[371 * 49]"
        )
    else:
        # Extract the observation value
        obs = re.search(r"Observation: (\d+)", prompt)
        val = obs.group(1) if obs else "unknown"
        return (
            f"Thought: The calculator returned {val}.\n"
            f"Answer: {val}"
        )

answer, trace = react_agent(
    "What is 371 * 49?", mock_llm_calc)
print(f"Final answer: {answer}")
# Final answer: 18179

# Verify
print(f"Correct: {371 * 49}")
# Correct: 18179
