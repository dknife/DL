"""
으뜸 딥러닝 — 14장 03절
ReAct 추적 예시 (의사 코드)
"""

# Question: What is the capital of the birth country
#           of the first author of the Transformer paper?

# --- Step 1 ---
# Thought: I need to find who the first author of the
#          Transformer paper ("Attention Is All You Need")
#          is. Let me search.
# Action:  search("first author Attention Is All You Need")
# Observation: The first author is Ashish Vaswani.

# --- Step 2 ---
# Thought: Now I need to find where Ashish Vaswani was
#          born. Let me search.
# Action:  search("Ashish Vaswani birthplace")
# Observation: Ashish Vaswani was born in India.

# --- Step 3 ---
# Thought: The birth country is India. The capital of
#          India is New Delhi. I have enough information.
# Answer:  New Delhi.
