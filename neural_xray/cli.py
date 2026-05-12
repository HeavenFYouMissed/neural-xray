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


def _check_for_updates():
    """Non-blocking update check — compares installed version to latest GitHub release."""
    try:
        import urllib.request, json as _json
        from neural_xray import __version__ as installed
        req = urllib.request.Request(
            "https://api.github.com/repos/HeavenFYouMissed/neural-xray/releases/latest",
            headers={"User-Agent": "neural-xray-update-check"},
        )
        with urllib.request.urlopen(req, timeout=4) as r:
            data = _json.loads(r.read())
        latest = data.get("tag_name", "").lstrip("v")
        if latest and latest != installed:
            print(f"\n[neural-xray] Update available: v{installed} → v{latest}")
            print(f"[neural-xray] Run:  pip install --upgrade \"git+https://github.com/HeavenFYouMissed/neural-xray.git\"\n")
    except Exception:
        pass  # never block startup — network down, rate limit, etc. all silently ignored


def cmd_serve(args):
    """Launch the full GUI dashboard (FastAPI + bundled React app)."""
    import webbrowser
    import threading
    import time

    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
    except ImportError:
        print("[neural-xray] Server deps missing. Install with:")
        print("    pip install 'neural-xray[server]'")
        sys.exit(1)

    from neural_xray.server.server import app

    url = f"http://{args.host}:{args.port}"
    print(f"[neural-xray] Launching dashboard at {url}")
    print(f"[neural-xray] Press Ctrl+C to stop.")

    threading.Thread(target=_check_for_updates, daemon=True).start()

    if not args.no_browser:
        def _open():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_surgery(args):
    """Knowledge transplant: copy concept(s) from source model into target model."""
    from neural_xray.loader import ModelLoader
    from neural_xray.extractor import ConceptExtractor
    from neural_xray.transplanter import KnowledgeTransplanter

    print(f"[neural-xray] Loading source: {args.source}")
    src = ModelLoader(args.source, force_quantization=args.quantization); src.load()
    print(f"[neural-xray] Loading target: {args.target}")
    tgt = ModelLoader(args.target, force_quantization=args.quantization); tgt.load()

    print(f"[neural-xray] Extracting {len(args.concepts)} concept(s) from source...")
    src_vecs = ConceptExtractor(src).extract(args.concepts)

    print(f"[neural-xray] Transplanting into target...")
    tx = KnowledgeTransplanter(src, tgt)
    report = tx.transplant(src_vecs)
    print(f"\nTransplant complete.")
    print(f"  concepts: {len(args.concepts)}")
    print(f"  layers touched: {getattr(report, 'layers_modified', 'n/a')}")
    if args.output:
        Path(args.output).write_text(json.dumps(getattr(report, '__dict__', {}), indent=2, default=str))
        print(f"  report: {args.output}")


def cmd_abliterate(args):
    """Remove a concept direction from a model (abliteration)."""
    from neural_xray.loader import ModelLoader
    from neural_xray.extractor import ConceptExtractor
    import torch

    print(f"[neural-xray] Loading {args.model}")
    loader = ModelLoader(args.model, force_quantization=args.quantization); loader.load()

    print(f"[neural-xray] Extracting direction for: {args.concept}")
    vecs = ConceptExtractor(loader).extract([args.concept])
    direction = list(vecs.values())[0]
    direction = direction / (direction.norm() + 1e-8)

    print(f"[neural-xray] Subtracting direction from output projection weights...")
    model = loader.model
    count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim == 2 and param.shape[-1] == loader.hidden_size:
                proj = param @ direction.to(param.device).to(param.dtype)
                param.data -= torch.outer(proj, direction.to(param.device).to(param.dtype))
                count += 1
    print(f"  modified {count} weight matrices.")
    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out))
        loader.tokenizer.save_pretrained(str(out))
        print(f"  model saved to {out}")



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

    # serve — full GUI dashboard
    p_serve = subparsers.add_parser("serve", help="Launch the full GUI dashboard in your browser")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser")

    # surgery — knowledge transplant
    p_surg = subparsers.add_parser("surgery", help="Transplant concepts from source model into target model")
    p_surg.add_argument("--source", required=True, help="Source HuggingFace model ID")
    p_surg.add_argument("--target", required=True, help="Target HuggingFace model ID")
    p_surg.add_argument("--concepts", nargs="+", required=True, help="Concepts to transplant")
    p_surg.add_argument("--quantization", default=None, choices=["float32","float16","8bit","4bit"])
    p_surg.add_argument("--output", default=None, help="Save transplant report JSON")

    # abliterate — remove concept direction
    p_abl = subparsers.add_parser("abliterate", help="Remove a concept direction from a model")
    add_common(p_abl)
    p_abl.add_argument("--concept", required=True, help="Concept to remove")
    p_abl.add_argument("--save", default=None, help="Directory to save the modified model")

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
        "serve": cmd_serve,
        "surgery": cmd_surgery,
        "abliterate": cmd_abliterate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
