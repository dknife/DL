"""
으뜸 딥러닝 — 14장 04절
사전 계획-실행 패턴 (의사 코드)
"""

def plan_then_execute(task):
    # Step 1: LLM generates a plan
    plan = llm.chat(f"""
    Break down the following task into steps:
    Task: {task}
    Output a numbered list of steps.
    """)

    # Step 2: Execute each step sequentially
    results = []
    for step in parse_steps(plan):
        result = react_loop(step, context=results)
        results.append(result)

    # Step 3: Synthesize final answer
    return llm.chat(f"Summarize results: {results}")
