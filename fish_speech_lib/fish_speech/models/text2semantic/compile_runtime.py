# Copyright 2025-2026 Atm4x
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Callable, Iterator, TypeVar

import torch

from fish_speech_lib.fish_speech.device import DeviceLike, resolve_device
from fish_speech_lib.fish_speech.diagnostics import record_device_event


F = TypeVar("F", bound=Callable)


def _device_context(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.device(device)
    return nullcontext()


def _inductor_context(compile_kwargs: dict):
    if compile_kwargs.get("backend") != "inductor":
        return nullcontext()

    import torch._inductor.config as inductor_config

    settings = {
        "coordinate_descent_tuning": True,
        "triton.unique_kernel_names": True,
    }
    if hasattr(inductor_config, "fx_graph_cache"):
        settings["fx_graph_cache"] = True
    return inductor_config.patch(settings)


@contextmanager
def _compile_context(
    device: torch.device, compile_kwargs: dict
) -> Iterator[None]:
    with _device_context(device), _inductor_context(compile_kwargs):
        yield


def compile_on_device(function: F, device: DeviceLike, **compile_kwargs) -> F:
    """Compile and invoke a callable under the CUDA context owned by its model."""
    resolved = resolve_device(device)

    with _compile_context(resolved, compile_kwargs):
        compiled = torch.compile(function, **compile_kwargs)

    record_device_event("compiler_configured", device=resolved, options=compile_kwargs)

    @wraps(function)
    def invoke(*args, **kwargs):
        # torch.compile is lazy. Keep both first-time codegen/benchmarking and
        # subsequent CUDA-graph replay inside the selected model device scope.
        with _compile_context(resolved, compile_kwargs):
            if (
                resolved.type == "cuda"
                and compile_kwargs.get("mode") == "reduce-overhead"
            ):
                marker = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
                if callable(marker):
                    marker()
            return compiled(*args, **kwargs)

    invoke._fish_compile_device = resolved
    invoke._fish_compiled = True
    return invoke
