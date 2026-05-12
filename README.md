<div align="center">

<img width="1184" height="532" alt="image (3)" src="https://github.com/user-attachments/assets/563767cd-f789-47e1-8333-910c29cc2119" />

#  Neural-Xray

### **X-ray vision into any LLM — see where concepts live, how they flow, what can be cut.**

[![PyPI](https://img.shields.io/badge/install-pip-blue.svg)](https://github.com/HeavenFYouMissed/neural-xray)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CPU · GPU · Apple Silicon](https://img.shields.io/badge/runs%20on-CPU%20%7C%20CUDA%20%7C%20MPS-orange.svg)]()
[![Models](https://img.shields.io/badge/models-any%20HF%20transformer-purple.svg)]()

**Trace concepts • Diagnose health • Transplant knowledge • Abliterate directions • Visualize attention**

</div>

---

<div align="center">

###  One command. Full dashboard.

```bash
pip install git+https://github.com/HeavenFYouMissed/neural-xray.git
neural-xray serve
```

> Browser opens. Load any HuggingFace model. Start dissecting.

</div>

---

##  What you get

<table>
<tr>
<td width="50%" valign="top">

###  The Lab
- **Trace** — watch a concept flow through every layer
- **Diagnose** — 7-layer health scan in one click
- **Layer Map** — see which concepts live where
- **Attention** — visualize attention head patterns
- **SAE** — sparse autoencoder feature decomposition

</td>
<td width="50%" valign="top">

###  The OR
- **Surgery** — transplant knowledge between models
- **Abliterate** — remove a concept direction permanently
- **Patching** — causal tracing & activation patching
- **Stepthrough** — step layer-by-layer like a debugger
- **Compare** — diff two models side-by-side

</td>
</tr>
</table>

---

##  The Dashboard

> Screenshots dropping soon — placeholders below.
<img width="1361" height="1016" alt="Screenshot 2026-04-06 064623" src="https://github.com/user-attachments/assets/555436fe-eebc-46df-8507-01ecb16423c6" />
<div align="center">

### Trace · Layer Map · Diagnostics
<table>
<tr>
<td align="center" width="33%"><b>Concept Trace</b></td>
<td align="center" width="33%"><b>Layer Map</b></td>
<td align="center" width="33%"><b>Diagnostics</b></td>
</tr>
<tr>
<td><img width="1438" height="823" alt="image" src="https://github.com/user-attachments/assets/904c0e1c-d329-459a-b8cd-ca9df22a69da" />
<td><img width="1913" height="1084" alt="image" src="https://github.com/user-attachments/assets/2861cadc-c3cd-460b-aeec-084cb0ce11e5" />
<td><img width="1437" height="812" alt="image" src="https://github.com/user-attachments/assets/563fcd48-18d6-4cce-86cf-46714cbb990b" />

</table>







### Surgery · Abliterate · Attention
<table>
<tr>
<td align="center" width="33%"><b>Knowledge Transplant</b></td>
<td align="center" width="33%"><b>Concept Removal</b></td>
<td align="center" width="33%"><b>Attention Heads</b></td>
</tr>
<tr>
<td><img src="https://placehold.co/600x400/0d1117/f78166?text=Surgery" alt="Surgery"></td>
<td><img src="https://placehold.co/600x400/0d1117/f78166?text=Abliterate" alt="Abliterate"></td>
<td><img src="https://placehold.co/600x400/0d1117/f78166?text=Attention" alt="Attention"></td>
</tr>
</table>

### Graph · SAE · Compare
<table>
<tr>
<td align="center" width="33%"><b>Activation Graph</b></td>
<td align="center" width="33%"><b>SAE Features</b></td>
<td align="center" width="33%"><b>Model Compare</b></td>
</tr>
<tr>
<td><img src="https://placehold.co/600x400/0d1117/a371f7?text=Graph+View" alt="Graph"></td>
<td><img src="https://placehold.co/600x400/0d1117/a371f7?text=SAE" alt="SAE"></td>
<td><img src="https://placehold.co/600x400/0d1117/a371f7?text=Compare" alt="Compare"></td>
</tr>
</table>

</div>

---

##  Quick start

### Option A — The full GUI

```bash
pip install git+https://github.com/HeavenFYouMissed/neural-xray.git
neural-xray serve
```

Browser opens at `http://127.0.0.1:8000`. Load a model. Click around.

### Option B — The CLI

```bash
neural-xray diagnose  --model gpt2
neural-xray trace     --model gpt2 --concept fire
neural-xray map       --model gpt2
neural-xray extract   --model gpt2 --concepts fire water gravity --output vecs.json
neural-xray visualize --model gpt2 --output viz.html
neural-xray surgery   --source gpt2 --target distilgpt2 --concepts fire water
neural-xray abliterate --model gpt2 --concept hate --save ./gpt2-cleaned
```

### Option C — The Python API

```python
from neural_xray import ModelLoader, ConceptFlowTracer, ModelDiagnostics

loader = ModelLoader("gpt2")
loader.load()

# Trace any concept through every layer
trace = ConceptFlowTracer(loader).trace("gravity")
print(trace.top_concepts)

# 7-layer health scan
report = ModelDiagnostics(loader).run_all()
print(f"Overall: {report.overall_score:.3f}")
```

<div align="center">

##  Hardware

| Platform        | Status | Notes                                                       |
| --------------- | :----: | ----------------------------------------------------------- |
| Linux + CUDA    |   ✅   | Full speed, 4-bit / 8-bit quantization via `bitsandbytes`   |
| Windows + CUDA  |   ✅   | Same as Linux                                               |
| Apple Silicon   |   ✅   | MPS-accelerated (M1/M2/M3/M4), float16                      |
| Intel Mac / CPU |   ✅   | Float32 fallback, slow on big models but works              |
| No GPU at all   |   ✅   | CPU mode — fine for GPT-2 / DistilGPT-2 class              
<No model size limit.** Load 70B, 405B, whatever your hardware can handle.

<div align="center">

##  What's inside
<div align="center">
```
neural_xray/
  loader.py         model loading + auto quantization (CUDA / MPS / CPU)
  mapper.py         architecture discovery
  extractor.py      pull concept vectors via activation hooks
  tracer.py         layer-by-layer concept flow
  diagnostics.py    7-layer health checks
  cluster.py        semantic clustering of concepts
  cartography.py    LoRA cartography
  blueprint.py      structured JSON of model knowledge
  projector.py      cross-model dimension projection
  transplanter.py   write concept vectors into target weights
  sae.py            sparse autoencoders
  visualizer.py     interactive HTML
  evidence.py       reproducibility bundles
  cli.py            command-line interface
  server/           FastAPI + bundled React dashboard
```
</div>

---

## 🤝 Contributing

Issues, PRs, screenshots, demos — all welcome.

##  License

MIT. Use it however you want.

---

<div align="center">

**Built for the next generation of mechanistic interpretability.** • 

[File an issue](https://github.com/HeavenFYouMissed/neural-xray/issues) • 

<div align="center">

  <img width="180" height="180" alt="qr-code" src="https://github.com/user-attachments/assets/85ea0abb-19eb-4a86-9398-42e5b5a7adff" /> 
</div>
---

  <div align="center">
    
<a href="https://www.buymeacoffee.com/HeavenFYouMissed" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" style="height: 60px !important;width: 217px !important;" ></a>

---

<div align="center">
  
 Star if this is useful • 
</div>
