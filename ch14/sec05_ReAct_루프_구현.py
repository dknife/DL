"""
으뜸 딥러닝 — 14장 05절
ReAct 루프 구현
"""

def parse_action(text):
    """Extract tool name and argument from Action line."""
    match = re.search(
        r"Action:\s*(\w+)\[(.+?)\]", text)
    if match:
        return match.group(1), match.group(2)
    return None, None

def react_agent(question, llm_fn, max_steps=5):
    """Run the ReAct loop for a given question."""
    prompt = (
        "Answer the question using Thought/Action/Observation steps.\n"
        "Available tools: calculator[expr], search[query], "
        "string_length[text]\n"
        "When you have the final answer, write: "
        "Answer: <your answer>\n\n"
        f"Question: {question}\n"
    )
    trace = []  # record each step

    for step in range(1, max_steps + 1):
        # LLM generates Thought and optionally Action
        response = llm_fn(prompt)
        prompt += response + "\n"
        trace.append(f"--- Step {step} ---\n{response}")

        # Check if final answer is given
        if "Answer:" in response:
            answer_match = re.search(
                r"Answer:\s*(.+)", response)
            answer = (answer_match.group(1).strip()
                      if answer_match else response)
            return answer, trace

        # Parse and execute action
        tool_name, tool_arg = parse_action(response)
        if tool_name and tool_name in TOOLS:
            observation = TOOLS[tool_name](tool_arg)
            obs_text = f"Observation: {observation}"
            prompt += obs_text + "\n"
            trace.append(obs_text)
        else:
            obs_text = "Observation: Invalid action. " \
                       "Use tool_name[argument] format."
            prompt += obs_text + "\n"
            trace.append(obs_text)

    return "Max steps reached.", trace
