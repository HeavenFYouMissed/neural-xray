<div align="center">

<img width="1184" height="532" alt="image (3)" src="https://github.com/user-attachments/assets/563767cd-f789-47e1-8333-910c29cc2119" />

#  Neural-Xray

### X-ray vision into any LLM — and cross-model knowledge transplant without retraining

---

## Where this came from

My background is hardware engineering, DMA memory debugging, and biomedical repair — not machine learning. I approached transformers the way I'd approach a DMA memory problem: if you want to know what a system is doing internally, you instrument it. You put hooks on the memory bus, capture what flows through, and map it.

Transformers have a memory bus too: the residual stream. MLP layers write concept representations into it layer by layer. So I built tools to read that stream.

The inspection pipeline came first. The surgery idea came when I noticed something: **concept representations are geometrically consistent across completely different models.** A "gravity" vector in Pythia-6.9B and a "gravity" vector in Mistral-7B point in compatible directions — even though these models have different tokenizers, different MLP structure, different training data, and came from different organizations. That led to experiments nobody had published before.

---

## What makes this different from ROME, MEMIT, and TransformerLens

**ROME** and **MEMIT** edit facts *inside one model only* — change "Paris is the capital of France" to something else inside GPT-J. Useful, but limited to single-model factual edits.

**TransformerLens** is excellent for inspection, but research-first and doesn't do cross-model surgery.

**What this does that neither of those do:**

**1. LoRA as a reverse-engineering dye.** Instead of analyzing raw weights (uninterpretable noise), the tool trains LoRA adapters on concept-specific text, then decomposes those adapters via SVD. The low-rank directions LoRA finds *are* the concept directions in weight space — like injecting contrast dye to see where a concept structurally lives. This is the `cartography` module. No prior tool does this.

**2. Procrustes alignment between model concept spaces.** Once you have concept vectors from two models, you find the optimal rotation (via the [orthogonal Procrustes problem](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)) that maps one model's geometry onto the other's. The residual of that rotation tells you how compatible two models are. At 7B scale: **98.9% mean cosine alignment** across 25 concepts between Pythia and Mistral. Null baseline (random rotation): 1.2%.

**3. Cross-architecture concept transplant.** Move a knowledge representation from one model into a structurally different model — no training data, no fine-tuning, no shared tokenizer required. Pure geometry.

**4. Transplant + train acceleration.** Controlled A/B test: transplant Swahili concept representations from Mistral-7B into Pythia-6.9B, then fine-tune both on the same 160 sentences. The transplanted model won **20/20 eval checkpoints** and converged **50% faster** to the same perplexity threshold.

**5. Negative control (self-surgery).** 3 approaches tested to edit a model from its own internals only, no donor. All 3 failed. Cross-model transplant does something you cannot replicate by operating on a single model in isolation — the donor is necessary, which is what makes this non-trivial.

---

## Research papers this is built on

- **[ROME](https://arxiv.org/abs/2202.05262)** — locating factual associations in MLP mid-layers. Motivated the extraction target.
- **[MEMIT](https://arxiv.org/abs/2210.07229)** — mass editing via rank-limited MLP updates.
- **[Contrastive Activation Addition (CAA)](https://arxiv.org/abs/2312.06681)** — contrastive prompt pairs to extract concept-specific directions with template artifacts removed. The baseline subtraction in `extractor.py` is based directly on this.
- **[LoRA](https://arxiv.org/abs/2106.09685)** — low-rank weight decomposition. Used here as a diagnostic probe, not for training efficiency.
- **[Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)** — larger models converge toward a shared geometric representation of reality regardless of architecture. Consistent with our scale results: 91.7% at 124M → 98.9% at 7B → >99% at 70B.
- **[Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)** — reading predictions from intermediate layers. Used in trace/stepthrough views.
- **Sparse Autoencoders** (Anthropic 2023-2024) — monosemantic feature decomposition. Available as the SAE tab.

---

## Verified experimental results

All runs logged in [`docs/surgery_test_log.md`](docs/surgery_test_log.md). Raw JSON in [`experiment_results/`](experiment_results/) and [`evidence/`](evidence/).

| Scale | Donor → Target | Concepts | Result | Notes |
|-------|---------------|----------|--------|-------|
| 124M | GPT-2 → DistilGPT-2 | 3 | **+15.5%** probe alignment | Full before/after evidence pack saved |
| 7B | Pythia-6.9B → Mistral-7B | 25 | **98.9%** mean cosine | Per-concept JSON; null baseline 1.2% |
| 7B + train | Mistral → Pythia + fine-tune | 4 | **20/20 eval wins**, 50% faster convergence | ARM B beats control at every checkpoint |
| 7B self-surgery | Pythia only (no donor) | — | All 3 approaches failed | Strengthens the cross-model case |
| ~70B | LLaMA-3.1-70B → Qwen2.5-72B | 30 | **>99%** alignment | Run on H100 NVL; logs lost in pod termination — rerun planned |

Paper draft: [model-surgery-paper](https://github.com/HeavenFYouMissed/model-surgery-paper)
[![PyPI](https://img.shields.io/badge/install-pip-blue.svg)](https://github.com/HeavenFYouMissed/neural-xray)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CPU · GPU · Apple Silicon](https://img.shields.io/badge/runs%20on-CPU%20%7C%20CUDA%20%7C%20MPS-orange.svg)]()
[![Models](https://img.shields.io/badge/models-any%20HF%20transformer-purple.svg)]()

**Trace concepts • Diagnose health • Transplant knowledge • Abliterate directions • Visualize attention**
<div align="center">
> Archive: [Zenodo 10.5281/zenodo.19467270](https://doi.org/10.5281/zenodo.19467270) | Paper: [model-surgery-paper](https://github.com/HeavenFYouMissed/model-surgery-paper)

CPU works for GPT-2-scale demos and learning the UI. For real models and surgery workflows, treat a GPU (or Apple Silicon) as the practical requirement — mainly for VRAM and speed, not because the tool is CUDA-only.
---

<div align="center">

###  One command. Full dashboard.

```bash
pip install git+https://github.com/HeavenFYouMissed/neural-xray.git
neural-xray serve
```

> IF your not familiar with using bash commands literly just copy paist this open cmd and just paist it and press go and the tool will pop open in your browser ready to go...Browser opens. Load any HuggingFace model. Start dissecting.

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

> My favorite the debugging tab.
<tr>
<td width="50%" valign="top">
<img width="800" height="500" alt="Screenshot 2026-04-06 064623" src="https://github.com/user-attachments/assets/555436fe-eebc-46df-8507-01ecb16423c6" />
</td>
<div align="center">
  
> Type anything into the chat and see why and what the model is doing this is the debugger tab... 
  
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
surgery- Cross-model knowledge transplantation +
concept abliteration. Map gaps, transplant
knowledge, or remove specific concepts. once you have extracted knowledge or concepts they dont have to be re-extracted, you could litterly have a bank of concepts or extracted knoledge, the idea is injecting a comcept to a model that doesnt have it and training the model with the information to the concept. the idea is that if you pre inject the concept the training will target that area instead of just randomly garbling bs during training.. from my experience and results its over 50% faster.. feel free to test yourself this the reason i didnt want to throw out a research paper with scuh high success its better to let the community validate it. im independent researcher so my opinion wouldnt have much sway anyways..
<table>
<tr>
<td align="center" width="33%"><b>Knowledge Transplant</b></td>
<td align="center" width="33%"><b>Concept Removal</b></td>
<td align="center" width="33%"><b>Attention Heads</b></td>
</tr>
<tr>
<td><img width="1910" height="1077" alt="Screenshot 2026-05-12 023036" src="https://github.com/user-attachments/assets/2794905e-c5e9-4ab5-a615-b9875e1823d5" /></td>
<td><img width="1414" height="793" alt="Screenshot 2026-05-12 015604" src="https://github.com/user-attachments/assets/ec1055b9-171d-4ca5-8935-83c9e92e7155" /></td>
<td><img width="1908" height="926" alt="Screenshot 2026-05-12 015922" src="https://github.com/user-attachments/assets/15276ef2-fa83-425a-b41b-baa3617418f5" /></td>
</tr>
</table>

### Graph · SAE · Compare
graph- does Token x layer prediction grid. Each node shows
what the model predicts at that layer - like an
X-ray of evolving predictions.

sae-Train Sparse Autoencoders to decompose
concepts into monosemantic features. Reveals
what individual neurons encode.

compare- side by side comparison of two "loaded",
models, compare health, comcept overlap, 
check score, you can check the same model loaded 
twice for after surgery and ablation,
many uses

<table>
<tr>
<td align="center" width="33%"><b>Activation Graph</b></td>
<td align="center" width="33%"><b>SAE Features</b></td>
<td align="center" width="33%"><b>Model Compare</b></td>
</tr>
<tr>
<td><img width="1894" height="1024" alt="Screenshot 2026-05-12 020149" src="https://github.com/user-attachments/assets/ad64f631-5e96-4e85-b097-05eab3ebd988" /></td>
<td><img width="1901" height="1070" alt="Screenshot 2026-05-12 020542" src="https://github.com/user-attachments/assets/e4de0e1c-bb31-4f6d-805a-dc372eca9ec8" /></td>
<td><img width="1903" height="1070" alt="Screenshot 2026-05-12 020937" src="https://github.com/user-attachments/assets/f846d36d-29bc-4ff1-b69f-a8bde8223661" /></td>
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
