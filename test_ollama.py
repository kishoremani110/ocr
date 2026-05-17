#!/usr/bin/env python3
"""Quick sanity-check: can Ollama respond at all?  Run this first."""
import sys, time
try:
    import ollama
except ImportError:
    print("pip install ollama"); sys.exit(1)

HOST  = "http://localhost:11434"
MODEL = "glm-ocr"

client = ollama.Client(host=HOST)

# 1. List models — fast, no inference needed
print("Checking Ollama is reachable ...")
try:
    models = client.list()
    names  = [m.model for m in models.models]
    print(f"  Models available: {names}")
    if not any(MODEL in n for n in names):
        print(f"\n  WARNING: '{MODEL}' not found. Run:  ollama pull {MODEL}")
except Exception as exc:
    print(f"  ERROR: cannot reach Ollama at {HOST}\n  {exc}")
    sys.exit(1)

# 2. Text-only ping — no image, should be near-instant
print(f"\nSending text-only ping to {MODEL} ...")
t0 = time.time()
try:
    resp = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with one word: ready"}],
        stream=False,
    )
    print(f"  Response ({time.time()-t0:.1f}s): {resp['message']['content']!r}")
except Exception as exc:
    print(f"  ERROR: {exc}")
