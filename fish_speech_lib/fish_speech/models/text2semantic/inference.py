# Original work Copyright 2024 Fish Audio Authors (Apache License 2.0)
# Modified by Atm4x in 2025.
# SPDX-License-Identifier: Apache-2.0

# --- In decode_one_token_ar function ---
# Added .clone() before stacking codebooks.
# codebooks = [cb.clone() for cb in codebooks]
# codebooks = torch.stack(codebooks, dim=0)

import os
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import click
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from fish_speech_lib.fish_speech.device import module_device, resolve_device
from fish_speech_lib.fish_speech.conversation import (
    CODEBOOK_PAD_TOKEN_ID,
    Conversation,
    Message,
    TextPart,
    VQPart,
)
from fish_speech_lib.fish_speech.models.text2semantic.llama import BaseModelArgs
from fish_speech_lib.fish_speech.models.text2semantic.sampling import SamplingRng
from fish_speech_lib.fish_speech.models.text2semantic.compile_runtime import compile_on_device
from fish_speech_lib.fish_speech.text import clean_text, split_text
from fish_speech_lib.fish_speech.tokenizer import IM_END_TOKEN, FishTokenizer


from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech_lib.fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)


def multinomial_sample_one_no_sync(
    probs_sort: torch.Tensor, *, noise: torch.Tensor
) -> torch.Tensor:
    """Sample from probabilities using caller-owned request-local noise."""
    return torch.argmax(probs_sort / noise, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=0, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def multinomial_sample_one_no_sync_agent(
    probs_sort: torch.Tensor, *, noise: torch.Tensor
) -> torch.Tensor:
    """Batched sampling using caller-owned request-local noise."""
    return torch.argmax(probs_sort / noise, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs_agent(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    *,
    noise: torch.Tensor,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs, noise=noise)
    return idx_next, probs


def sample_agent(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    *,
    noise: torch.Tensor,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs_agent(
        logits=logits[:, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync_agent(probs, noise=noise)
    return idx_next, probs



def _sampling_parts(
    rng: SamplingRng,
    model: BaseTransformer,
    shapes: list[tuple[int, ...]],
) -> tuple[torch.Tensor, ...]:
    parameter = next(model.parameters())
    return rng.exponential_parts(
        shapes,
        device=parameter.device,
        dtype=parameter.dtype,
    )


def _sampling_noise_ar(
    model: DualARTransformer, x: torch.Tensor, rng: SamplingRng
) -> tuple[torch.Tensor, ...]:
    return _sampling_parts(
        rng,
        model,
        [
            (model.config.vocab_size,),
            *((model.config.codebook_size,) for _ in range(1, model.config.num_codebooks)),
        ],
    )


def _sampling_noise_naive(
    model: NaiveTransformer, x: torch.Tensor, rng: SamplingRng
) -> tuple[torch.Tensor, ...]:
    return _sampling_parts(
        rng,
        model,
        [
            (model.config.vocab_size,),
            *((model.config.codebook_size,) for _ in range(model.config.num_codebooks)),
        ],
    )


def _sampling_noise_ar_agent(
    model: DualARTransformer, x: torch.Tensor, rng: SamplingRng
) -> tuple[torch.Tensor, ...]:
    batch_size = x.shape[0]
    return _sampling_parts(
        rng,
        model,
        [
            (batch_size, model.config.vocab_size),
            *((batch_size, model.config.codebook_size) for _ in range(model.config.num_codebooks)),
        ],
    )


def _sampling_noise_naive_agent(
    model: NaiveTransformer, x: torch.Tensor, rng: SamplingRng
) -> tuple[torch.Tensor, ...]:
    batch_size = x.shape[0]
    return _sampling_parts(
        rng,
        model,
        [
            (model.config.vocab_size,),
            *((batch_size, model.config.codebook_size) for _ in range(model.config.num_codebooks)),
        ],
    )

def _make_decode_callable(compiled_impl, noise_factory):
    def decode_one_token(
        model,
        x,
        input_pos,
        semantic_ids,
        previous_tokens=None,
        *,
        rng: SamplingRng,
        temperature: torch.Tensor = 1.0,
        top_p: torch.Tensor = 1.0,
        repetition_penalty: torch.Tensor = 1.0,
    ):
        return compiled_impl(
            model,
            x,
            input_pos,
            semantic_ids,
            previous_tokens,
            sampling_noise=noise_factory(model, x, rng),
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    decode_one_token._fish_compiled = True
    decode_one_token._fish_compile_device = getattr(
        compiled_impl, "_fish_compile_device", None
    )
    return decode_one_token


def _decode_one_token_ar_agent_impl(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    *,
    sampling_noise: tuple[torch.Tensor, ...],
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    x = model.forward_generate(x, input_pos)
    logits = x.logits
    hidden_states = x.hidden_states

    sampling_kwargs_main = {
        "temperature": 0.1,
        "top_p": 0.1,
        "repetition_penalty": 1.0,
    }

    codebooks = [
        sample_agent(
            logits,
            previous_tokens=None,
            noise=sampling_noise[0],
            **sampling_kwargs_main,
        )[0]
    ]

    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    sampling_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    for codebook_idx in range(model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)
        a = sample_agent(
            logits,
            previous_tokens=(
                previous_tokens[:, codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
            noise=sampling_noise[codebook_idx + 1],
            **sampling_kwargs,
        )[0]
        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)
    semantic_ids_tensor = torch.tensor(semantic_ids, device=codebooks.device)
    codebooks[:, 1:, :] = torch.masked_fill(
        codebooks[:, 1:, :],
        ~torch.isin(codebooks[:, :1, :], semantic_ids_tensor),
        CODEBOOK_PAD_TOKEN_ID,
    )
    return codebooks


def decode_one_token_ar_agent(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    *,
    rng: SamplingRng,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    return _decode_one_token_ar_agent_impl(
        model,
        x,
        input_pos,
        semantic_ids,
        previous_tokens,
        sampling_noise=_sampling_noise_ar_agent(model, x, rng),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


def _decode_one_token_naive_agent_impl(
    model: NaiveTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    *,
    sampling_noise: tuple[torch.Tensor, ...],
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    result = model.forward_generate(x, input_pos)
    sampling_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }

    codebooks = [
        sample(
            result.token_logits,
            previous_tokens=None,
            noise=sampling_noise[0],
            **sampling_kwargs,
        )[0]
    ]

    for i in range(model.config.num_codebooks):
        codebooks.append(
            sample_agent(
                result.codebook_logits[:, :, i],
                previous_tokens=(
                    previous_tokens[:, i + 1] if previous_tokens is not None else None
                ),
                noise=sampling_noise[i + 1],
                **sampling_kwargs,
            )[0]
        )

    codebooks = torch.stack(codebooks, dim=1)
    semantic_ids_tensor = torch.tensor(semantic_ids, device=codebooks.device)
    codebooks[:, 1:, :] = torch.masked_fill(
        codebooks[:, 1:, :],
        ~torch.isin(codebooks[:, :1, :], semantic_ids_tensor),
        CODEBOOK_PAD_TOKEN_ID,
    )
    return codebooks


def decode_one_token_naive_agent(
    model: NaiveTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    *,
    rng: SamplingRng,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    return _decode_one_token_naive_agent_impl(
        model,
        x,
        input_pos,
        semantic_ids,
        previous_tokens,
        sampling_noise=_sampling_noise_naive_agent(model, x, rng),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


def _decode_one_token_ar_impl(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    *,
    sampling_noise: tuple[torch.Tensor, ...],
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    result = model.forward_generate(x, input_pos)
    sampling_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }

    codebooks = [
        sample(
            result.logits,
            previous_tokens=(
                previous_tokens[0] if previous_tokens is not None else None
            ),
            noise=sampling_noise[0],
            **sampling_kwargs,
        )[0]
    ]
    hidden_states = result.hidden_states

    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)
    a = codebooks[0] - model.tokenizer.semantic_begin_id
    a[a < 0] = 0
    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)
        a = sample(
            logits,
            previous_tokens=(
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
            noise=sampling_noise[codebook_idx],
            **sampling_kwargs,
        )[0]
        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = [codebook.clone() for codebook in codebooks]
    return torch.stack(codebooks, dim=0)


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    *,
    rng: SamplingRng,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    return _decode_one_token_ar_impl(
        model,
        x,
        input_pos,
        semantic_ids,
        previous_tokens,
        sampling_noise=_sampling_noise_ar(model, x, rng),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


def _decode_one_token_naive_impl(
    model: NaiveTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    *,
    sampling_noise: tuple[torch.Tensor, ...],
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    result = model.forward_generate(x, input_pos)
    sampling_kwargs_main = {
        "temperature": 0.1,
        "top_p": 0.1,
        "repetition_penalty": 1.0,
    }
    sampling_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }

    codebooks = [
        sample(
            result.token_logits,
            previous_tokens=None,
            noise=sampling_noise[0],
            **sampling_kwargs_main,
        )[0]
    ]

    for i in range(model.config.num_codebooks):
        codebooks.append(
            sample(
                result.codebook_logits[:, :, i],
                previous_tokens=(
                    previous_tokens[i + 1] if previous_tokens is not None else None
                ),
                noise=sampling_noise[i + 1],
                **sampling_kwargs,
            )[0]
        )
    return torch.stack(codebooks, dim=0)


def decode_one_token_naive(
    model: NaiveTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    *,
    rng: SamplingRng,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    return _decode_one_token_naive_impl(
        model,
        x,
        input_pos,
        semantic_ids,
        previous_tokens,
        sampling_noise=_sampling_noise_naive(model, x, rng),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


def decode_n_tokens(
    model: NaiveTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    semantic_ids: list,
    decode_one_token=decode_one_token_naive,
    **sampling_kwargs,
):
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    for i in tqdm(range(num_new_tokens)):
        # Compute the window and clone it
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :win_size].clone()
        else:
            window = previous_tokens[:, i - win_size : i].clone()

        is_cuda = cur_token.device.type == "cuda"
        with sdpa_kernel(SDPBackend.MATH) if is_cuda else nullcontext():
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                semantic_ids=semantic_ids,
                **sampling_kwargs,
            ).clone()

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, i : i + 1] = next_token.view(
            model.config.num_codebooks + 1, -1
        )

        if cur_token[0, 0, -1] == model.tokenizer.get_token_id(IM_END_TOKEN):
            break

    return previous_tokens[:, : i + 1]


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: NaiveTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    decode_one_token=decode_one_token_naive,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    # semantic_id = model.tokenizer.convert_tokens_to_ids("<|semantic|>")
    semantic_ids = [
        model.tokenizer.get_token_id(f"<|semantic:{i}|>") for i in range(1024)
    ]

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype
    model_device = module_device(model)
    if device != model_device:
        raise ValueError(
            f"Prompt is on {device}, but text-semantic model is on {model_device}"
        )

    codebook_dim = 1 + model.config.num_codebooks
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    # Use non-accelerated version for now, to avoid compilation overhead
    prefill_decode = (
        decode_one_token_naive
        if isinstance(model, NaiveTransformer)
        else decode_one_token_ar
    )

    next_token = prefill_decode(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        semantic_ids=semantic_ids,
        **sampling_kwargs,
    )
    seq[:, T : T + 1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    x = decode_n_tokens(
        model,
        next_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        decode_one_token=decode_one_token,
        semantic_ids=semantic_ids,
        **sampling_kwargs,
    )
    # x = torch.cat(generated_tokens, dim=1)
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x

    return seq


def decode_n_tokens_agent(
    model: NaiveTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    semantic_ids: list,
    im_end_id: int = 4,
    decode_one_token=decode_one_token_naive_agent,
    early_stop_threshold: float = 0.6,
    **sampling_kwargs,
):
    batch_size = cur_token.size(0)
    previous_tokens = torch.zeros(
        (batch_size, model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=cur_token.device)
    finished = finished | (cur_token[:, 0, -1] == im_end_id)
    start_time = time.time()

    for i in tqdm(range(num_new_tokens), desc="Decoding: ", total=num_new_tokens):
        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :, :win_size]
        else:
            window = previous_tokens[:, :, i - win_size : i]

        is_cuda = cur_token.device.type == "cuda"
        with sdpa_kernel(SDPBackend.MATH) if is_cuda else nullcontext():
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                semantic_ids=semantic_ids,
                **sampling_kwargs,
            )

        input_pos += 1
        cur_token = next_token.view(batch_size, model.config.num_codebooks + 1, -1)
        previous_tokens[:, :, i : i + 1] = next_token.view(
            batch_size, model.config.num_codebooks + 1, -1
        )

        yield cur_token.cpu()

        finished = finished | (cur_token[:, 0, -1] == im_end_id)
        if finished.all() or (
            0 < early_stop_threshold < 1
            and finished.sum() >= round(batch_size * early_stop_threshold)
        ):
            break

    total_time = time.time() - start_time
    generated_tokens = i + 1
    tokens_per_second = (generated_tokens / total_time) * batch_size
    logger.info(
        f"Decoded {generated_tokens} x {batch_size} tokens in {total_time:.2f}s ({tokens_per_second:.2f} tokens/s)"
    )


@torch.no_grad()
@torch.inference_mode()
def generate_agent(
    *,
    model: BaseTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    semantic_ids: list,
    im_end_id: int = 4,
    decode_one_token=decode_one_token_naive_agent,
    num_samples: int = 1,
    early_stop_threshold: float = 0.6,
    seed: Optional[int] = None,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype
    model_device = module_device(model)
    if device != model_device:
        raise ValueError(
            f"Prompt is on {device}, but text-semantic model is on {model_device}"
        )
    rng = SamplingRng(device, seed)

    codebook_dim = 1 + model.config.num_codebooks
    input_pos = torch.arange(0, T, device=device)

    # Use non-accelerated version for now, to avoid compilation overhead
    prefill_decode = (
        decode_one_token_naive_agent
        if isinstance(model, NaiveTransformer)
        else decode_one_token_ar_agent
    )
    next_token = prefill_decode(
        model,
        prompt,
        input_pos,
        semantic_ids=semantic_ids,
        rng=rng,
        **sampling_kwargs,
    ).view(num_samples, codebook_dim, -1)
    yield next_token.cpu()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    yield from decode_n_tokens_agent(
        model,
        next_token,
        input_pos,
        max_new_tokens - 1,
        im_end_id=im_end_id,
        semantic_ids=semantic_ids,
        decode_one_token=decode_one_token,
        early_stop_threshold=early_stop_threshold,
        rng=rng,
        **sampling_kwargs,
    )


def encode_tokens(
    tokenizer,
    string,
    device="cuda",
    prompt_tokens=None,
    num_codebooks=4,
):
    device = resolve_device(device)
    string = clean_text(string)

    messages = []
    messages.append(
        Message(
            role="user",
            parts=[TextPart(text=string)],
            cal_loss=False,
        )
    )

    if prompt_tokens is not None:
        if prompt_tokens.ndim == 3:
            assert (
                prompt_tokens.shape[0] == 1
            ), "3D prompt tokens should have shape (1, num_codebooks, seq_len)"
            prompt_tokens = prompt_tokens[0]

        assert prompt_tokens.ndim == 2, "Prompt tokens should be 2D tensor"

        if prompt_tokens.shape[0] > num_codebooks:
            logger.warning(
                f"Prompt tokens shape {prompt_tokens.shape} is larger than num_codebooks {num_codebooks}, getting first {num_codebooks} codebooks"
            )
            prompt_tokens = prompt_tokens[:num_codebooks]

        vq_part = VQPart(codes=prompt_tokens.to(device))

        messages.append(
            Message(
                role="assistant",
                parts=[TextPart(text="<|voice|>"), vq_part],
                cal_loss=False,
            )
        )
    else:
        messages.append(
            Message(
                role="assistant",
                parts=[TextPart(text="<|voice|>")],
                cal_loss=False,
                add_im_end=False,
            )
        )

    conversation = Conversation(messages=messages)
    # conversation.visualize(tokenizer)
    encoded = conversation.encode_for_inference(
        tokenizer=tokenizer,
        num_codebooks=num_codebooks,
    )

    return encoded.to(device)


def load_model(checkpoint_path, device, precision, compile=False, is_agent=False):
    device = resolve_device(device)
    model: Union[NaiveTransformer, DualARTransformer] = BaseTransformer.from_pretrained(
        checkpoint_path, load_weights=True, is_agent=is_agent
    )

    model = model.to(device=device, dtype=precision).eval()
    logger.info(f"Restored model from checkpoint on {device}")

    if isinstance(model, DualARTransformer):
        decode_impl = (
            _decode_one_token_ar_agent_impl if is_agent else _decode_one_token_ar_impl
        )
        noise_factory = _sampling_noise_ar_agent if is_agent else _sampling_noise_ar
        eager_decode = decode_one_token_ar_agent if is_agent else decode_one_token_ar
        logger.info("Using DualARTransformer")
    else:
        decode_impl = (
            _decode_one_token_naive_agent_impl
            if is_agent
            else _decode_one_token_naive_impl
        )
        noise_factory = (
            _sampling_noise_naive_agent if is_agent else _sampling_noise_naive
        )
        eager_decode = (
            decode_one_token_naive_agent if is_agent else decode_one_token_naive
        )
        logger.info("Using NaiveTransformer")

    if not compile:
        return model, eager_decode

    logger.info(f"Compiling full token decode path for {device}...")
    is_cuda = device.type == "cuda"
    compile_kwargs = {
        "fullgraph": True,
        "backend": "inductor" if is_cuda else "aot_eager",
        "mode": "reduce-overhead" if is_cuda else None,
    }
    compiled_impl = compile_on_device(decode_impl, device, **compile_kwargs)
    decode_one_token = _make_decode_callable(compiled_impl, noise_factory)
    return model, decode_one_token


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


def generate_long(
    *,
    model,
    decode_one_token: callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: int = 0.7,
    repetition_penalty: float = 1.5,
    temperature: float = 0.7,
    compile: bool = False,
    iterative_prompt: bool = True,
    max_length: int = 2048,
    chunk_length: int = 150,
    prompt_text: Optional[str | list[str]] = None,
    prompt_tokens: Optional[torch.Tensor | list[torch.Tensor]] = None,
    seed: Optional[int] = None,
):
    device = module_device(model)
    rng = SamplingRng(device, seed)

    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    use_prompt = prompt_text is not None and prompt_tokens is not None
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    assert use_prompt is False or len(prompt_text) == len(
        prompt_tokens
    ), "Prompt text and tokens must have the same length"

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer
    im_end_id = tokenizer.get_token_id("<|im_end|>")

    encoded = []
    texts = split_text(text, chunk_length) if iterative_prompt else [text]
    encoded_prompts = [
        Conversation(
            messages=[
                Message(
                    role="system",
                    parts=[TextPart(text="Speak out the provided text.")],
                    cal_loss=False,
                )
            ]
        )
        .encode_for_inference(
            tokenizer=tokenizer,
            num_codebooks=model.config.num_codebooks,
        )
        .to(device)
    ]

    if use_prompt:
        for idx, (t, c) in enumerate(zip(prompt_text, prompt_tokens)):
            encoded_prompts.append(
                encode_tokens(
                    tokenizer,
                    string=t,
                    device=device,
                    prompt_tokens=c,
                    num_codebooks=model.config.num_codebooks,
                )
            )

    for idx, text in enumerate(texts):
        encoded.append(
            encode_tokens(
                tokenizer,
                string=text,
                device=device,
                num_codebooks=model.config.num_codebooks,
            )
        )
        logger.info(f"Encoded text: {text}")

    # Move temperature, top_p, repetition_penalty to device
    # This is important so that changing params doesn't trigger recompile
    temperature = torch.tensor(temperature, device=device, dtype=torch.float)
    top_p = torch.tensor(top_p, device=device, dtype=torch.float)
    repetition_penalty = torch.tensor(
        repetition_penalty, device=device, dtype=torch.float
    )

    for sample_idx in range(num_samples):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        global_encoded = []
        seg_idx = 0

        while seg_idx < len(encoded):
            logger.info(
                f"Generating sentence {seg_idx + 1}/{len(encoded)} of sample {sample_idx + 1}/{num_samples}"
            )

            seg = encoded[seg_idx]
            global_encoded.append(seg)

            lengths = reversed([seg.size(1) for seg in global_encoded])

            # Pick last 2000 tokens
            count = 0
            for i, length in enumerate(lengths):
                count += length
                if count + length > max_length - 1024 - sum(
                    t.shape[1] for t in encoded_prompts
                ):
                    break

            if i != 0 and i % 2 == 0:
                i -= 1

            # Rotate the list, always make sure first segment is included to avoid drift
            if i < len(global_encoded) - 2:
                partial_encoded = global_encoded[:2] + global_encoded[-i:]
            else:
                partial_encoded = global_encoded

            if use_prompt:
                partial_encoded = encoded_prompts + partial_encoded

            cat_encoded = torch.cat(partial_encoded, dim=1)
            prompt_length = cat_encoded.size(1)

            t0 = time.perf_counter()
            y = generate(
                model=model,
                prompt=cat_encoded,
                max_new_tokens=max_new_tokens,
                decode_one_token=decode_one_token,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                rng=rng,
            )

            if sample_idx == 0 and seg_idx == 0 and compile:
                logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            t = time.perf_counter() - t0

            tokens_generated = y.size(1) - prompt_length
            tokens_sec = tokens_generated / t
            logger.info(
                f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_sec:.02f} tokens/sec"
            )
            logger.info(
                f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
            )

            if device.type == "cuda":
                logger.info(
                    f"GPU Memory used on {device}: "
                    f"{torch.cuda.max_memory_reserved(device) / 1e9:.02f} GB"
                )

            # Put the generated tokens
            # since there is <im_end>, we remove last token
            codes = y[1:, prompt_length + 1 :].clone()
            assert (codes >= 0).all(), f"Negative code found"

            decoded = y[:, prompt_length:].clone()
            # But for global encoding, we should keep the <im_end> token

            global_encoded.append(decoded)
            assert (codes >= 0).all(), f"Negative code found: {codes}"
            yield GenerateResponse(action="sample", codes=codes, text=texts[seg_idx])
            seg_idx += 1

        # This indicates the end of the current sample
        yield GenerateResponse(action="next")


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[GenerateResponse | Exception] = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


class TextSemanticWorker:
    """Owns a text-semantic request queue, worker thread, and model device."""

    def __init__(self, device: torch.device):
        self.device = resolve_device(device)
        self._requests: queue.Queue[GenerateRequest | None] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._lock = threading.Lock()

    def attach_thread(self, thread: threading.Thread) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("TextSemanticWorker thread is already attached")
            self._thread = thread

    def submit(self, request: GenerateRequest) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("TextSemanticWorker is closed")
            self._requests.put(request)

    def put(
        self, request: GenerateRequest, block: bool = True, timeout: Optional[float] = None
    ) -> None:
        """Compatibility alias for callers that previously received queue.Queue."""
        with self._lock:
            if self._closed:
                raise RuntimeError("TextSemanticWorker is closed")
            self._requests.put(request, block=block, timeout=timeout)

    def next_request(self) -> GenerateRequest | None:
        return self._requests.get()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        self._requests.put(None)
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    device = resolve_device(device)
    input_queue = TextSemanticWorker(device)
    init_event = threading.Event()
    init_error: list[Exception] = []

    def worker():
        try:
            model, decode_one_token = load_model(
                checkpoint_path, device, precision, compile=compile
            )
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=4096,  # model.config.max_seq_len
                dtype=next(model.parameters()).dtype,
            )
        except Exception as exc:
            init_error.append(exc)
            init_event.set()
            return

        init_event.set()

        while True:
            item = input_queue.next_request()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )
            except Exception as e:
                response_queue.put(WrappedGenerateResponse(status="error", response=e))

    thread = threading.Thread(target=worker, daemon=True)
    input_queue.attach_thread(thread)
    thread.start()
    init_event.wait()
    if init_error:
        thread.join()
        raise RuntimeError("Failed to initialize text-semantic worker") from init_error[0]

    return input_queue


def launch_thread_safe_queue_agent(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    device = resolve_device(device)
    input_queue = TextSemanticWorker(device)
    init_event = threading.Event()
    init_error: list[Exception] = []

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    config = BaseModelArgs.from_pretrained(checkpoint_path)

    def worker():
        try:
            model, decode_one_token = load_model(
                checkpoint_path, device, precision, compile=compile, is_agent=True
            )
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=4096,  # model.config.max_seq_len
                dtype=next(model.parameters()).dtype,
            )
        except Exception as exc:
            init_error.append(exc)
            init_event.set()
            return

        init_event.set()

        while True:
            item = input_queue.next_request()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for token in generate_agent(
                    model=model,
                    decode_one_token=decode_one_token,
                    **kwargs,
                ):
                    response_queue.put(token)

                response_queue.put("stop")
            except Exception as e:
                import traceback

                logger.exception(f"Error in worker: {traceback.format_exc()}")
                response_queue.put("error")

    thread = threading.Thread(target=worker, daemon=True)
    input_queue.attach_thread(thread)
    thread.start()
    init_event.wait()
    if init_error:
        thread.join()
        raise RuntimeError(
            "Failed to initialize text-semantic agent worker"
        ) from init_error[0]

    return input_queue, tokenizer, config


@click.command()
@click.option(
    "--text",
    type=str,
    default="你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
)
@click.option("--prompt-text", type=str, default=None, multiple=True)
@click.option(
    "--prompt-tokens",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option("--num-samples", type=int, default=1)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-p", type=float, default=0.7)
@click.option("--repetition-penalty", type=float, default=1.2)
@click.option("--temperature", type=float, default=0.7)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="checkpoints/fish-speech-1.5",
)
@click.option("--device", type=str, default="cuda")
@click.option("--compile/--no-compile", default=False)
@click.option("--seed", type=int, default=42)
@click.option("--half/--no-half", default=False)
@click.option("--iterative-prompt/--no-iterative-prompt", default=True)
@click.option("--chunk-length", type=int, default=100)
@click.option("--output-dir", type=Path, default="temp")
def main(
    text: str,
    prompt_text: Optional[list[str]],
    prompt_tokens: Optional[list[Path]],
    num_samples: int,
    max_new_tokens: int,
    top_p: int,
    repetition_penalty: float,
    temperature: float,
    checkpoint_path: Path,
    device: str,
    compile: bool,
    seed: int,
    half: bool,
    iterative_prompt: bool,
    chunk_length: int,
    output_dir: Path,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    precision = torch.half if half else torch.bfloat16
    device = resolve_device(device, fallback_to_cpu=True)

    if prompt_text is not None and len(prompt_text) != len(prompt_tokens):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
        )

    logger.info("Loading model ...")
    t0 = time.time()
    model, decode_one_token = load_model(
        checkpoint_path, device, precision, compile=compile
    )
    model.setup_caches(
        max_batch_size=1,
        max_seq_len=model.config.max_seq_len,
        dtype=next(model.parameters()).dtype,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    if prompt_tokens is not None:
        prompt_tokens = [torch.from_numpy(np.load(p)).to(device) for p in prompt_tokens]

    generator = generate_long(
        model=model,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
        seed=seed,
    )

    idx = 0
    codes = []

    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
            logger.info(f"Sampled text: {response.text}")
        elif response.action == "next":
            if codes:
                codes_npy_path = os.path.join(output_dir, f"codes_{idx}.npy")
                np.save(codes_npy_path, torch.cat(codes, dim=1).cpu().numpy())
                logger.info(f"Saved codes to {codes_npy_path}")
            logger.info(f"Next sample")
            codes = []
            idx += 1
        else:
            logger.error(f"Error: {response}")


if __name__ == "__main__":
    main()
