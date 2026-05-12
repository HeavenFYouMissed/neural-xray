# Research results (what the surgery work actually tested)

Neural-Xray is the **inspection + surgery UI**. The experiments below are the research behind it. Scripts, logs, JSON, and the paper draft live in the main **Model Surgery** research tree (same author); this page is the short public summary.

**Paper draft:** [Model Surgery: Transplanting Knowledge Between Neural Networks Without Retraining](https://github.com/HeavenFYouMissed/model-surgery-paper) (also `paper/model_surgery_v2.tex` in the research repo)  
**Permanent copy:** [Zenodo 10.5281/zenodo.19467270](https://doi.org/10.5281/zenodo.19467270)  
**Session log:** `docs/surgery_test_log.md` in the research repo

---

## Paste into README (short)

```markdown
## Research behind this repo

Open-sourced **Neural-Xray** (see inside the model) plus **Model Surgery** (copy knowledge between models without full retraining).

| Scale | What we ran | Headline (from our logs / paper) |
| ----- | ----------- | -------------------------------- |
| Laptop | GPT-2 → smaller models | Measured before/after on topic transfer; evidence in `evidence/2026-04-06_transplant_validation/` |
| ~7B (RunPod) | Pythia-6.9B ↔ Mistral-7B | 25 topics, **~98.9%** alignment; saved JSON in `experiment_results/7b_experiment_*.json` |
| ~7B + training | Swahili lessons after a small copy | Copy-first model stayed ahead in that A/B run (see `docs/surgery_test_log.md`, `transplant_train.log`) |
| ~70B (H100) | LLaMA-3.1-70B → Qwen2.5-72B | Paper §70B: **30** topics, **>99%** alignment, **0** interference aborts — reproduce with `experiments/map_70b_concepts.py` + `experiments/run_70b_transplant.py` |

Full write-up: [docs/research-results.md](docs/research-results.md). Re-run the numbers on your hardware; don’t trust a README table alone.
```

---

## Paste into Reddit (plain English)

**Title idea:** Open-sourced an LLM “X-ray” tool + the model-surgery experiments behind it (124M → 7B → 70B)

**Body:**

I spent a stupid amount of time and cloud GPU money testing whether you can **copy knowledge from one open model into another** without training the donor from scratch—and built a **local dashboard** to see inside the model while you do it.

**The tool (try this first):** [Neural-Xray](https://github.com/HeavenFYouMissed/neural-xray) — `pip install` + `neural-xray serve`, load a Hugging Face model, watch layers react to your text. CPU is fine for small models; big models need real GPU/RAM.

**What we actually ran (reproducible, not a pitch deck):**

- **On a laptop:** GPT-2 → smaller model, saved before/after probes and generations.
- **On RunPod (~7B):** Two different 7B models, 25 topics—alignment **~98.9%** in our saved JSON; scripts and logs in the research repo.
- **Copy + teach (7B):** Copy a small slice of Swahili-related weights from the stronger multilingual model, then train both on the same lessons—the copy-first run stayed ahead in that test (details in the surgery log).
- **Large (~70B on H100):** Described in our paper draft: LLaMA-3.1-70B → Qwen2.5-72B, 30 topics, **>99%** alignment in the write-up—driver scripts are in the repo; you need serious hardware to rerun.

Paper draft + Zenodo: linked from `docs/research-results.md` in the Neural-Xray repo.

I’m an engineer, not a lab. I want people to **rerun** and tell me what breaks. Screenshots in the README; video still on my list.

---

## Results table (for posts / README)

| Scale | Models (donor → target) | What we measured | Headline | Where to verify |
| ----- | ------------------------- | ---------------- | -------- | --------------- |
| 124M | GPT-2 → DistilGPT-2 | Topic transfer + probes | **~91.7%** mean alignment (paper); laptop evidence pack | `evidence/2026-04-06_transplant_validation/`, paper §small |
| 7B | Pythia-6.9B → Mistral-7B | 25 topics, full pipeline | **~98.9%** mean cosine; 25/25 transplanted; null baseline **~1.2%** | `experiment_results/7b_experiment_20260406_154002.json`, `docs/surgery_test_log.md` |
| 7B | Mistral → Pythia (asymmetric) | Niche topics where donor is stronger | Full JSON run log | `experiment_results/asymmetric_experiment_20260406_171810.json` |
| 7B | Self-surgery vs donor | Swahili without donor | Donor transplant + train beat self-edits | `experiment_results/self_surgery_v3.json`, surgery log |
| 7B | Mistral → Pythia + Swahili training | A/B same lessons | Copy-first ahead at every checkpoint in that run | `docs/surgery_test_log.md`, `experiment_results/transplant_train.log` |
| 7B | Language copy (Finnish / Swahili / Welsh / Hungarian) | Perplexity + prompts | Small copy + teach matters; copy-only often hurts | `experiment_results/scale_language_experiment.log`, `run_scale_language_experiment.py` |
| 70B | LLaMA-3.1-70B → Qwen2.5-72B | 30 topics, alignment | **>99%** mean alignment, 30/30, 0 aborts (paper) | `paper/model_surgery_v2.tex` §70B, `experiments/run_70b_transplant.py` |

---

## Reproduce (high level)

**Inspect only (Neural-Xray):**

```bash
pip install git+https://github.com/HeavenFYouMissed/neural-xray.git
neural-xray serve
# or: neural-xray diagnose --model gpt2
```

**7B transplant (research repo, CUDA, RunPod-class GPU):**

```bash
pip install transformers torch accelerate sentencepiece protobuf
python run_7b_experiment.py
```

**70B (research repo, multi-GPU / H100, maps from pod workflow):**

```bash
python experiments/map_70b_concepts.py    # donor map on pod A
python experiments/run_70b_transplant.py  # collector pod B
```

**Language + scale sweep:**

```bash
python run_scale_language_experiment.py
python run_transplant_train.py
```

Compare your numbers to the JSON and logs under `experiment_results/` and `docs/surgery_test_log.md`.

---

## Honest limits

- Not every JSON from every RunPod session is in git (`*.log` is gitignored; some `*.json` paths were only on `/workspace/`).
- 70B headline numbers are in the **paper** and **scripts**; check in your own `70b_experiment_*.json` after a rerun.
- Alignment scores are not the same as “the model is now an expert.” Training and eval still matter—especially for languages.
