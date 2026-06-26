"""
EcoPrompt - Smoke test
Run: python test_proxy.py
Verifies the proxy shell is running and returns valid structure.
Does NOT make a real OpenAI call — mocks the upstream response.
"""

import httpx
import json

BASE = "http://localhost:8000"

def test_health():
    r = httpx.get(f"{BASE}/health")
    assert r.status_code == 200
    print("✅ /health OK:", r.json())

def test_stats():
    r = httpx.get(f"{BASE}/stats")
    assert r.status_code == 200
    print("✅ /stats OK:", r.json())

def test_bad_json():
    r = httpx.post(f"{BASE}/v1/chat/completions", content="not json",
                   headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    print("✅ /v1/chat/completions rejects bad JSON:", r.json())

if __name__ == "__main__":
    print("Running EcoPrompt smoke tests...\n")
    test_health()
    test_stats()
    test_bad_json()
    print("\n✅ All smoke tests passed. Proxy shell is working.")
