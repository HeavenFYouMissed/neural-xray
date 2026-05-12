<div align="center">

<img width="1184" height="532" alt="image (3)" src="https://github.com/user-attachments/assets/563767cd-f789-47e1-8333-910c29cc2119" />

#  Neural-Xray

### **X-ray vision into any LLM — See AI model internals like never before. Transfer concepts and knowledge to concentrate training a new method with crazy results. I was planning to gatekeep and sell but decided against it, if your finding this i hope that you like it, and if you have any suggestions or insight let me know i will be keeping up on this! I am not much of a machine learning person but enjoy messing with ai, and im from a engineering background and dma research/debugging and biomedical repair, i kinda stumbled onto this and thought it would make a difference.. i dont understand why you can see ai output but you cant see why a ai works or its internals so i took a look at it like a dma problem or memory debugging issue, and this is what i came up with and it works!!! have fun...

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
