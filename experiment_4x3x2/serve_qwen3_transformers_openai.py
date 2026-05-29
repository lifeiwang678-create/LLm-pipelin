from __future__ import annotations

import argparse
import threading
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small OpenAI-compatible Qwen3 server using transformers.")
    parser.add_argument("--model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--served-model-name", default="qwen3-8b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-model-len", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"Loading tokenizer: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    print(f"Loading model: {args.model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    print(f"Model ready on {args.device}: {args.served_model_name}", flush=True)

    app = FastAPI()
    lock = threading.Lock()

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": args.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> dict[str, Any]:
        payload = await request.json()
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        max_tokens = int(payload.get("max_tokens") or 128)
        temperature = float(payload.get("temperature") or 0.0)
        top_p = payload.get("top_p")
        chat_template_kwargs = payload.get("chat_template_kwargs") or {}
        if not isinstance(chat_template_kwargs, dict):
            raise HTTPException(status_code=400, detail="chat_template_kwargs must be an object")
        chat_template_kwargs.setdefault("enable_thinking", False)

        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **chat_template_kwargs,
            )
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

        if input_ids.shape[-1] > args.max_model_len:
            input_ids = input_ids[:, -args.max_model_len :]
        input_ids = input_ids.to(args.device)
        attention_mask = torch.ones_like(input_ids, device=args.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
            if top_p is not None:
                generate_kwargs["top_p"] = float(top_p)
        else:
            generate_kwargs["do_sample"] = False

        with lock, torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        completion_ids = output_ids[0, input_ids.shape[-1] :]
        content = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        usage = {
            "prompt_tokens": int(input_ids.shape[-1]),
            "completion_tokens": int(completion_ids.shape[-1]),
            "total_tokens": int(input_ids.shape[-1] + completion_ids.shape[-1]),
        }
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": args.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
