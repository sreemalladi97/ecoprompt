# test_step3.py
import sys
sys.path.insert(0, ".")

from core.cache import cache_lookup, cache_store, validate_answer

print("Testing Step 3 — Answer Validator\n")

# --- Test the validator directly first ---

# Test 1: Good match — answer clearly answers the question
q1 = "What is GitHub?"
a1 = "GitHub is a platform for hosting and collaborating on code using Git."
result1 = validate_answer(q1, a1)
print(f"Test 1 - Good match:")
print(f"  Question: {q1}")
print(f"  Answer:   {a1}")
print(f"  Valid: {'✅ ACCEPTED' if result1 else '❌ REJECTED'}\n")

# Test 2: Bad match — the exact scenario from your PDF
q2 = "Who is the data science student?"
a2 = "That's great to hear you're a data science student!"
result2 = validate_answer(q2, a2)
print(f"Test 2 - Bad match (PDF example):")
print(f"  Question: {q2}")
print(f"  Answer:   {a2}")
print(f"  Valid: {'✅ ACCEPTED' if result2 else '❌ REJECTED (correct!)'}\n")

# Test 3: Unrelated answer
q3 = "How do I reset my password?"
a3 = "The weather in Texas is hot and humid in the summer."
result3 = validate_answer(q3, a3)
print(f"Test 3 - Completely unrelated answer:")
print(f"  Question: {q3}")
print(f"  Answer:   {a3}")
print(f"  Valid: {'✅ ACCEPTED' if result3 else '❌ REJECTED (correct!)'}\n")

# --- Test the full pipeline ---
print("--- Full Pipeline Test ---\n")

# Store a Q&A pair
cache_store(
    [{"role": "user", "content": "What is GitHub?"}],
    {"content": "GitHub is a platform for hosting and collaborating on code."}
)

# Similar question — should HIT and pass validation
result4 = cache_lookup([{"role": "user", "content": "What is GitHub used for?"}])
print(f"Pipeline Test - Similar question:")
print(f"  Result: {'HIT + VALIDATED ✅' if result4 else 'MISS or REJECTED ❌'}\n")