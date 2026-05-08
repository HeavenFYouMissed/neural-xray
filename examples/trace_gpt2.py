"""
Trace how a concept flows through every layer of GPT-2.

Run:
    pip install neural-xray
    python examples/trace_gpt2.py
"""

from neural_xray.loader import ModelLoader
from neural_xray.tracer import ConceptFlowTracer

# Load GPT-2 — runs on CPU with float32
loader = ModelLoader("gpt2", force_quantization="float32")
loader.load()

tracer = ConceptFlowTracer(loader)

# Trace a concept through all layers
print("\nTracing 'fire' through GPT-2...")
trace = tracer.trace("fire")
print(f"  Trajectory shifts: {len(trace.trajectory)}")
print(f"  Concept chain: {' → '.join(trace.top_concepts[:8])}")
print(f"\n  Layer breakdown:")
for step in trace.layer_steps[:10]:
    print(f"    {step}")

# Trace multiple concepts
print("\n\nTracing 'gravity', 'love', 'democracy'...")
for concept in ["gravity", "love", "democracy"]:
    trace = tracer.trace(concept)
    print(f"  {concept:<15} {len(trace.trajectory)} shifts  →  {' → '.join(trace.top_concepts[:5])}")
