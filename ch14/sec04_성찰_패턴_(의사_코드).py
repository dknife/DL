"""
으뜸 딥러닝 — 14장 04절
성찰 패턴 (의사 코드)
"""

def solve_with_reflection(task, max_attempts=3):
    result = None
    feedback = ""

    for attempt in range(max_attempts):
        # Generate or revise solution
        result = react_loop(task, prior_feedback=feedback)

        # Self-evaluate
        evaluation = llm.chat(f"""
        Task: {task}
        Solution: {result}
        Evaluate: Is this correct and complete?
        If not, explain what needs to be fixed.
        """)

        if "correct" in evaluation.lower():
            return result

        # Use evaluation as feedback for next attempt
        feedback = evaluation

    return result  # return best effort
