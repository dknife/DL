"""
으뜸 딥러닝 — 14장 03절
함수 호출 예시 (의사 코드)
"""

# Step 1: Define available tools as JSON schema
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string",
                     "description": "City name"}
        },
        "required": ["city"]
    }
}]

# Step 2: LLM selects tool and generates arguments
response = llm.chat(
    messages=[{"role": "user",
               "content": "What is the weather in Seoul?"}],
    tools=tools
)
# LLM output: {"name": "get_weather", "args": {"city": "Seoul"}}

# Step 3: Execute tool and feed result back
weather = get_weather(city="Seoul")  # {"temp": 18, ...}
final = llm.chat(
    messages=[..., {"role": "tool", "content": str(weather)}]
)
# Final output: "Seoul is currently 18 degrees Celsius."
