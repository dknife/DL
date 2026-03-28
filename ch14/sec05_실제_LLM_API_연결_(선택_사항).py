"""
으뜸 딥러닝 — 14장 05절
실제 LLM API 연결 (선택 사항)
"""

# pip install openai
# import openai

# def real_llm(prompt):
#     """Call OpenAI API as the LLM backbone."""
#     response = openai.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0,
#         max_tokens=256,
#     )
#     return response.choices[0].message.content

# answer, trace = react_agent(
#     "What is the population of France squared?",
#     real_llm
# )
