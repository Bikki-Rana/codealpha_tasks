"""
measure_latency.py
-------------------
Simple local latency measurement - run this after installing
requirements to see real (not invented) numbers for YOUR machine.

Usage:
    cd backend
    python measure_latency.py
"""

import statistics
from translator import get_translator

SAMPLES = [
    ("English", "Hindi", "Where are you going?"),
    ("English", "Hindi", "Artificial intelligence is changing how we work and communicate."),
    ("Hindi", "English", "आप कहाँ जा रहे हैं?"),
    ("English", "Bengali", "This is a longer sentence used to measure translation latency on a normal CPU."),
]

if __name__ == "__main__":
    translator = get_translator()
    print("Loading model (one-time cost, not counted in per-request latency)...")
    translator.load()

    latencies = []
    for src, tgt, text in SAMPLES:
        translated, chunks, latency_ms = translator.translate(text, src, tgt)
        latencies.append(latency_ms)
        print(f"[{src} -> {tgt}] {latency_ms:.1f} ms ({chunks} chunk(s))")
        print(f"   in:  {text}")
        print(f"   out: {translated}")

    print("\n--- Summary (this machine, CPU) ---")
    print(f"mean:   {statistics.mean(latencies):.1f} ms")
    print(f"median: {statistics.median(latencies):.1f} ms")
    print(f"min:    {min(latencies):.1f} ms")
    print(f"max:    {max(latencies):.1f} ms")
    print("\nNote: these numbers are specific to this machine's CPU and will")
    print("differ elsewhere. They are measured here, never invented.")
