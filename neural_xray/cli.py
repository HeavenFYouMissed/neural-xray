"""
neural-xray — CLI entry point
X-ray vision into any LLM. Works on CPU, no GPU required.

Commands:
    diagnose    Run 7-layer health scan on a model
    trace       Trace how a concept flows through every layer
    extract     Extract concept vectors from a model
    visualize   Generate interactive HTML visualization
    map         Print layer architecture map
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_diagnose(args):
    from neural_xray.loader import ModelLoader
    from neural_xray.diagnostics import ModelDiagnostics

    print(f"[neural-xray] Loading {args.model} ...")
    loader = ModelLoader(args.model, force_quantization=args.quantization)
    loader.load()

    diag = ModelDiagnostics(loader)
    report = diag.run_all()

    print(f"\n{'='*60}")
    print(f"Diagnostic Report — {args.model}")
    print(f"{'='*60}")
    for check in report.checks:
        icon = "✓" if check.severity == "ok" else ("⚠" if check.severity == "warn" else "✗")
        print(f"  {icon} {check.name:<35} score={check.score:.3f}  ({check.severity})")
        if check.details:
            print(f"      {check.details}")
    print(f"\nOverall health: {report.overall_score:.3f}")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(report.__dict__ if hasattr(report, '__dict__') else str(report), indent=2))
        print(f"Report saved to {out}")


def cmd_trace(args):
    from neural_xray.loader import ModelLoader
    from neural_xray.tracer import ConceptFlowTracer

    print(f"[neural-xray] Loading {args.model} ...")
    loader = ModelLoader(args.model, force_quantization=args.quantization)
    loader.load()

    tracer = ConceptFlowTracer(loader)

    concepts = args.concepts if args.concepts else [args.concept]
    for concept in concepts:
        print(f"\n[neural-xray] Tracing '{concept}' through {loader.num_layers} layers...")
        trace = tracer.trace(concept)
        print(f"\nTrace — {concept}  ({len(trace.trajectory)} trajectory shifts)")
        print(f"  Chain: {' → '.join(trace.top_concepts[:10])}")
        print(f"\n  Layer breakdown:")
        for step in trace.layer_steps[:20]:
            print(f"    {step}")

    if args.output:
        print(f"[neural-xray] Output saved to {args.output}")


def cmd_extract(args):
    from neural_xray.loader import ModelLoader
    from neural_xray.extractor import ConceptExtractor
    import torch

    print(f"[neural-xray] Loading {args.model} ...")
    loader = ModelLoader(args.model, force_quantization=args.quantization)
    loader.load()

    extractor = ConceptExtractor(loader)
    concepts = args.concepts

    print(f"[neural-xray] Extracting {len(concepts)} concept vectors...")
    vectors = extractor.extract(concepts)

    print(f"\nExtracted {len(vectors)} vectors, dim={loader.hidden_size}")
    for name, vec in list(vectors.items())[:5]:
        norm = vec.norm().item() if hasattr(vec, 'norm') else 0
        print(f"  {name:<20} norm={norm:.4f}")
    if len(vectors) > 5:
        print(f"  ... and {len(vectors)-5} more")

    if args.output:
        out = Path(args.output)
        # Save as JSON-serializable dict
        save_data = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in vectors.items()}
        out.write_text(json.dumps(save_data))
        print(f"Vectors saved to {out}")


def cmd_visualize(args):
    from neural_xray.loader import ModelLoader
    from neural_xray.extractor import ConceptExtractor
    from neural_xray.tracer import ConceptFlowTracer
    from neural_xray.visualizer import NeuralVisualizer

    print(f"[neural-xray] Loading {args.model} ...")
    loader = ModelLoader(args.model, force_quantization=args.quantization)
    loader.load()

    output_path = args.output or "neural_xray_visualization.html"

    extractor = ConceptExtractor(loader)
    concepts = args.concepts or ["fire", "water", "gravity", "love", "time"]

    print(f"[neural-xray] Extracting {len(concepts)} concepts...")
    vectors = extractor.extract(concepts)

    print(f"[neural-xray] Building visualization...")
    viz = NeuralVisualizer(loader)
    html = viz.build(vectors)

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"[neural-xray] Visualization saved to: {output_path}")
    print(f"  Open in any browser to explore.")


def cmd_map(args):
    from neural_xray.loader import ModelLoader
    from neural_xray.mapper import ArchitectureMapper

    print(f"[neural-xray] Loading {args.model} ...")
    loader = ModelLoader(args.model, force_quantization=args.quantization)
    loader.load()

    mapper = ArchitectureMapper(loader)
    arch = mapper.map()

    print(f"\nArchitecture — {args.model}")
    print(f"  Hidden size:    {arch.hidden_size}")
    print(f"  Num layers:     {arch.num_layers}")
    print(f"  Attention heads:{arch.num_heads}")
    print(f"  MLP dim:        {arch.mlp_dim}")
    print(f"  Vocab size:     {arch.vocab_size}")
    print(f"  Total params:   {arch.total_params:,}")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(arch.__dict__ if hasattr(arch, '__dict__') else str(arch), indent=2))
        print(f"Architecture saved to {out}")


def main():
    parser = argparse.ArgumentParser(
        prog="neural-xray",
        description="X-ray vision into any LLM. Works on CPU, no GPU required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neural-xray diagnose --model gpt2
  neural-xray trace --model gpt2 --concept fire
  neural-xray trace --model gpt2 --concepts fire water gravity
  neural-xray extract --model gpt2 --concepts fire water --output vectors.json
  neural-xray visualize --model gpt2 --output viz.html
  neural-xray map --model gpt2

Tips:
  Use --quantization float32 for CPU (default on machines without GPU).
  Use --quantization float16 or 8bit on GPU for larger models.
  Any HuggingFace model ID works (e.g., distilbert/distilgpt2, meta-llama/Llama-2-7b-hf).
        """
    )

    # Global flags
    parser.add_argument("--version", action="version", version="neural-xray 0.1.0")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared args factory
    def add_common(sub):
        sub.add_argument("--model", required=True, help="HuggingFace model ID or local path")
        sub.add_argument("--quantization", default=None,
                         choices=["float32", "float16", "8bit", "4bit"],
                         help="Force a specific quantization (default: auto-detect)")
        sub.add_argument("--output", default=None, help="Save results to this file")

    # diagnose
    p_diag = subparsers.add_parser("diagnose", help="Run 7-layer health scan")
    add_common(p_diag)

    # trace
    p_trace = subparsers.add_parser("trace", help="Trace how a concept flows through layers")
    add_common(p_trace)
    p_trace.add_argument("--concept", default=None, help="Single concept to trace")
    p_trace.add_argument("--concepts", nargs="+", default=None, help="Multiple concepts to trace")

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract concept vectors")
    add_common(p_extract)
    p_extract.add_argument("--concepts", nargs="+", required=True, help="Concepts to extract")

    # visualize
    p_viz = subparsers.add_parser("visualize", help="Generate interactive HTML visualization")
    add_common(p_viz)
    p_viz.add_argument("--concepts", nargs="+", default=None, help="Concepts to visualize")

    # map
    p_map = subparsers.add_parser("map", help="Print model architecture map")
    add_common(p_map)

    args = parser.parse_args()

    # Validate trace has at least one concept
    if args.command == "trace" and not args.concept and not args.concepts:
        parser.error("trace requires --concept WORD or --concepts WORD [WORD ...]")

    dispatch = {
        "diagnose": cmd_diagnose,
        "trace": cmd_trace,
        "extract": cmd_extract,
        "visualize": cmd_visualize,
        "map": cmd_map,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
