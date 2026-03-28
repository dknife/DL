"""
으뜸 딥러닝 — 14장 05절
ReAct 에이전트의 도구 정의
"""

import re

# Tool 1: Calculator — evaluate math expressions
def calculator(expression: str) -> str:
    try:
        result = eval(expression,
                      {"__builtins__": {}},  # safe eval
                      {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Tool 2: Search — look up a knowledge base
KNOWLEDGE = {
    "transformer": "Transformer was proposed by Vaswani "
        "et al. in 2017 in 'Attention Is All You Need'. "
        "The first author is Ashish Vaswani.",
    "ashish vaswani": "Ashish Vaswani is a computer "
        "scientist born in India.",
    "india": "India is a country in South Asia. "
        "Its capital is New Delhi. Population: 1.4B.",
    "python": "Python is a programming language created "
        "by Guido van Rossum in 1991.",
}

def search(query: str) -> str:
    query_lower = query.lower()
    for key, value in KNOWLEDGE.items():
        if key in query_lower:
            return value
    return "No relevant information found."

# Tool 3: String length
def string_length(text: str) -> str:
    return str(len(text))

# Registry mapping tool names to functions
TOOLS = {
    "calculator": calculator,
    "search": search,
    "string_length": string_length,
}
