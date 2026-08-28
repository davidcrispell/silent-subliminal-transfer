from __future__ import annotations

import gc
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_tokenizer(model_config: dict[str, Any]):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["tokenizer_revision"],
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_config: dict[str, Any],
    *,
    adapter_path: str | Path | None = None,
    trainable_adapter: bool = False,
):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "revision": model_config["revision"],
        "torch_dtype": torch_dtype(model_config["dtype"]),
        "low_cpu_mem_usage": True,
    }
    if model_config.get("attn_implementation"):
        kwargs["attn_implementation"] = model_config["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(model_config["id"], **kwargs)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=trainable_adapter,
        )
    return model


def model_device(model):
    return next(model.parameters()).device


def place_for_inference(model):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The funded model runs require a CUDA GPU")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    model.to(device)
    model.eval()
    return device


def release_model(model) -> None:
    import torch

    if model is not None:
        try:
            model.to("cpu")
        except (RuntimeError, ValueError):
            pass
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def trainable_parameter_summary(model) -> dict[str, int | float]:
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "fraction": trainable / total,
    }
