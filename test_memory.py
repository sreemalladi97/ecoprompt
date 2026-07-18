# test_memory.py — Full Memory Layer Demo
import sys
sys.path.insert(0, ".")
from core.memory import (
    memory_clear, memory_store, memory_get,
    memory_get_all, memory_delete,
    memory_inject, handle_memory_command
)

print("=== EcoPrompt Memory Layer Demo ===\n")

# Clean slate
memory_clear()

# 1. Store facts manually (how the system stores operational context)
print("1. Storing project facts...")
memory_store("language", "JavaScript")
memory_store("framework", "React")
memory_store("database", "MongoDB")
memory_store("cloud", "Azure")
print(f"   Stored: {memory_get_all()}\n")

# 2. User sends a remember command
print("2. User types a remember command...")
messages = [{"role": "user", "content": "remember: project = Claims API"}]
was_command, confirmation = handle_memory_command(messages)
print(f"   Command detected: {'✅' if was_command else '❌'}")
print(f"   Response: {confirmation}\n")

# 3. Inject memory into a real user message
print("3. User asks a question — memory injected automatically...")
user_messages = [{"role": "user", "content": "Write me a database connection function"}]
enriched = memory_inject(user_messages)
print(f"   Original message count: {len(user_messages)}")
print(f"   Enriched message count: {len(enriched)}")
print(f"   Context injected: {enriched[0]['content']}")
print(f"   User message: {enriched[1]['content']}\n")

# 4. Delete a fact
print("4. Removing a fact...")
memory_delete("cloud")
print(f"   Remaining facts: {memory_get_all()}\n")

# 5. Persistence check
print("5. Reloading from disk...")
reloaded = memory_get_all()
print(f"   Facts survived: {'✅' if len(reloaded) > 0 else '❌'}")
print(f"   {reloaded}")