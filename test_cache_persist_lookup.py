# test_step5_lookup.py
import sys
sys.path.insert(0, ".")
from core.cache import cache_lookup

result = cache_lookup([{"role": "user", "content": "What is persistent storage?"}])
print(f"Result: {'HIT ✅ — data survived restart!' if result else 'MISS ❌ — not persisted'}")
print(result)