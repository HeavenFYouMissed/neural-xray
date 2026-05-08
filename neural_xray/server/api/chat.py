"""
Model Surgery MRI — Chat endpoint with per-token introspection.

Generates text from a loaded model, streaming tokens back to the client
along with per-token internals (logit lens snapshots, entropy, confidence).
"""

import asyncio
import json
import logging
import time
import traceback
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..state import state

logger = logging.getLogger("model-surgery")
router = APIRouter()


class ChatRequest(BaseModel):
    model_id: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    introspect: bool = True  # capture per-token layer data
    introspect_top_k: int = 5  # top-k predictions per layer to return


def _logit_lens_snapshot(model, hidden_states, tokenizer, top_k: int = 5):
    """
    Run logit lens on the last token position across all captured hidden states.
    Returns per-layer top predictions.
    """
    layers = []
    lm_head = model.lm_head if hasattr(model, 'lm_head') else None
    # Some models use model.get_output_embeddings()
    if lm_head is None:
        lm_head_fn = model.get_output_embeddings()
        if lm_head_fn is None:
            return layers
    else:
        lm_head_fn = lm_head

    norm = None
    # Try to find the final layer norm
    for name in ['transformer.ln_f', 'model.norm', 'model.final_layernorm',
                 'gpt_neox.final_layer_norm', 'transformer.norm']:
        parts = name.split('.')
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            norm = obj
            break
        except AttributeError:
            continue

    for i, h in enumerate(hidden_states):
        # h shape: [batch, seq_len, hidden_size] — take last token
        last_h = h[:, -1:, :]  # [1, 1, hidden]

        # Apply final norm if available (more accurate logit lens)
        if norm is not None:
            try:
                normed = norm(last_h)
            except Exception:
                normed = last_h
        else:
            normed = last_h

        with torch.no_grad():
            logits = lm_head_fn(normed)  # [1, 1, vocab]
            probs = torch.softmax(logits[0, 0].float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            topk_probs, topk_ids = probs.topk(top_k)

        tokens = []
        for prob, tid in zip(topk_probs.tolist(), topk_ids.tolist()):
            try:
                tok_str = tokenizer.decode([tid])
            except Exception:
                tok_str = f"<{tid}>"
            tokens.append({"token": tok_str, "prob": round(prob, 4), "id": tid})

        layers.append({
            "layer": i,
            "entropy": round(entropy, 4),
            "top_tokens": tokens,
        })

    return layers


@router.post("/generate")
async def generate(req: ChatRequest):
    """
    Non-streaming generation with optional introspection data.
    Returns the full response + per-token debug info.
    """
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    loader = sess.loader
    model = loader.model
    tokenizer = loader.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer.encode(req.prompt, return_tensors="pt").to(model.device)
    prompt_len = input_ids.shape[1]

    generated_tokens = []
    token_debug = []
    current_ids = input_ids

    logger.info(f"Chat generate: '{req.prompt[:60]}...' model={req.model_id}, max_tokens={req.max_tokens}")
    t0 = time.time()

    with torch.no_grad():
        for step in range(req.max_tokens):
            outputs = model(
                current_ids,
                output_hidden_states=req.introspect,
            )
            logits = outputs.logits[:, -1, :]  # [1, vocab]

            # Apply temperature
            if req.temperature > 0:
                logits = logits / req.temperature

            # Apply top-k filtering
            if req.top_k > 0:
                topk_vals, _ = logits.topk(req.top_k)
                logits[logits < topk_vals[:, -1:]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if req.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= req.top_p
                sorted_logits[mask] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = torch.softmax(logits.float(), dim=-1)
            if req.temperature == 0:
                next_id = probs.argmax(dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)

            token_id = next_id.item()

            # Stop on EOS
            if token_id == tokenizer.eos_token_id:
                break

            token_str = tokenizer.decode([token_id])
            confidence = probs[0, token_id].item()
            entropy = -(probs[0] * torch.log(probs[0] + 1e-10)).sum().item()

            # Introspection: logit lens on this step's hidden states
            layer_data = None
            if req.introspect and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                layer_data = _logit_lens_snapshot(
                    model, outputs.hidden_states, tokenizer,
                    top_k=req.introspect_top_k,
                )

            generated_tokens.append(token_str)
            token_debug.append({
                "step": step,
                "token": token_str,
                "token_id": token_id,
                "confidence": round(confidence, 4),
                "entropy": round(entropy, 4),
                "layers": layer_data,
            })

            # Append token for next step
            current_ids = torch.cat([current_ids, next_id], dim=1)

    elapsed = time.time() - t0
    response_text = "".join(generated_tokens)

    logger.info(
        f"Chat complete: {len(generated_tokens)} tokens in {elapsed:.2f}s "
        f"({len(generated_tokens)/elapsed:.1f} tok/s)"
    )

    return {
        "prompt": req.prompt,
        "response": response_text,
        "tokens_generated": len(generated_tokens),
        "generation_time_ms": round(elapsed * 1000),
        "tokens_per_second": round(len(generated_tokens) / max(elapsed, 0.001), 1),
        "token_debug": token_debug if req.introspect else None,
    }


@router.websocket("/stream")
async def chat_stream(websocket: WebSocket):
    """
    WebSocket streaming generation with per-token introspection.

    Client sends JSON: { model_id, prompt, max_tokens, temperature, top_k, top_p, introspect }
    Server streams JSON messages:
      { type: "token", step, token, token_id, confidence, entropy, layers? }
      { type: "done", response, tokens_generated, generation_time_ms, tokens_per_second }
      { type: "error", detail }
    """
    await websocket.accept()
    logger.info("Chat WebSocket connected")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "detail": "Invalid JSON"}))
                continue

            model_id = req.get("model_id")
            prompt = req.get("prompt", "")
            max_tokens = min(req.get("max_tokens", 128), 512)  # cap at 512
            temperature = req.get("temperature", 0.7)
            top_k = req.get("top_k", 50)
            top_p = req.get("top_p", 0.9)
            introspect = req.get("introspect", True)
            introspect_top_k = req.get("introspect_top_k", 5)

            if not model_id or not prompt:
                await websocket.send_text(json.dumps({
                    "type": "error", "detail": "model_id and prompt are required"
                }))
                continue

            try:
                sess = state.get_session(model_id)
            except KeyError:
                await websocket.send_text(json.dumps({
                    "type": "error", "detail": f"Model '{model_id}' not loaded"
                }))
                continue

            model = sess.loader.model
            tokenizer = sess.loader.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            generated_tokens = []
            current_ids = input_ids
            t0 = time.time()

            logger.info(f"Chat stream: '{prompt[:60]}' model={model_id}")

            try:
                with torch.no_grad():
                    for step in range(max_tokens):
                        outputs = model(
                            current_ids,
                            output_hidden_states=introspect,
                        )
                        logits = outputs.logits[:, -1, :]

                        if temperature > 0:
                            logits = logits / temperature
                        if top_k > 0:
                            topk_vals, _ = logits.topk(min(top_k, logits.size(-1)))
                            logits[logits < topk_vals[:, -1:]] = float('-inf')
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            mask = cum - torch.softmax(sorted_logits, dim=-1) >= top_p
                            sorted_logits[mask] = float('-inf')
                            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

                        probs = torch.softmax(logits.float(), dim=-1)
                        if temperature == 0:
                            next_id = probs.argmax(dim=-1, keepdim=True)
                        else:
                            next_id = torch.multinomial(probs, num_samples=1)

                        token_id = next_id.item()
                        if token_id == tokenizer.eos_token_id:
                            break

                        token_str = tokenizer.decode([token_id])
                        confidence = probs[0, token_id].item()
                        entropy = -(probs[0] * torch.log(probs[0] + 1e-10)).sum().item()

                        layer_data = None
                        if introspect and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                            layer_data = _logit_lens_snapshot(
                                model, outputs.hidden_states, tokenizer,
                                top_k=introspect_top_k,
                            )

                        msg = {
                            "type": "token",
                            "step": step,
                            "token": token_str,
                            "token_id": token_id,
                            "confidence": round(confidence, 4),
                            "entropy": round(entropy, 4),
                            "layers": layer_data,
                        }
                        await websocket.send_text(json.dumps(msg))
                        generated_tokens.append(token_str)
                        current_ids = torch.cat([current_ids, next_id], dim=1)

                        # Yield control so websocket stays responsive
                        await asyncio.sleep(0)

                elapsed = time.time() - t0
                await websocket.send_text(json.dumps({
                    "type": "done",
                    "response": "".join(generated_tokens),
                    "tokens_generated": len(generated_tokens),
                    "generation_time_ms": round(elapsed * 1000),
                    "tokens_per_second": round(len(generated_tokens) / max(elapsed, 0.001), 1),
                }))

            except Exception as e:
                logger.error(f"Chat stream error: {e}\n{traceback.format_exc()}")
                await websocket.send_text(json.dumps({
                    "type": "error", "detail": str(e)
                }))

    except WebSocketDisconnect:
        logger.info("Chat WebSocket disconnected")
