# test_step5_store.py
import sys
sys.path.insert(0, ".")
from core.cache import cache_store

cache_store(
    [{"role": "user", "content": "What is persistent storage?"}],
    {"content": "Persistent storage saves data to disk so it survives restarts."}
)
print("Stored. Now run test_step5_lookup.py")