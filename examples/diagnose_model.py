"""
Run a full 7-layer health diagnostic on GPT-2.

Run:
    pip install neural-xray
    python examples/diagnose_model.py
"""

from neural_xray.loader import ModelLoader
from neural_xray.diagnostics import ModelDiagnostics

# Load GPT-2 — runs on CPU with float32
loader = ModelLoader("gpt2", force_quantization="float32")
loader.load()

# Run 7-layer health scan
diag = ModelDiagnostics(loader)
report = diag.run_all()

print(f"\n{'='*60}")
print(f"Diagnostic Report — gpt2")
print(f"{'='*60}")
for check in report.checks:
    icon = "✓" if check.severity == "ok" else ("⚠" if check.severity == "warn" else "✗")
    print(f"  {icon} {check.name:<35} score={check.score:.3f}  [{check.severity}]")
    if check.details:
        print(f"      {check.details}")

print(f"\nOverall health: {report.overall_score:.3f}")
print("\nDone. Run on your own model:")
print("  loader = ModelLoader('your-org/your-model', force_quantization='float32')")
