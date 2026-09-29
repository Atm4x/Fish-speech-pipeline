# Copyright 2025-2026 Atm4x
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import secrets
from typing import Optional, Sequence

import torch

from fish_speech_lib.fish_speech.device import DeviceLike, resolve_device


class SamplingRng:
    """Request-local sampling RNG that never mutates process or device RNG state."""

    def __init__(self, device: DeviceLike, seed: Optional[int] = None) -> None:
        self.device = resolve_device(device)
        self.seed = self._normalize_seed(seed)

        generator_device = (
            self.device
            if self.device.type in {"cpu", "cuda"}
            else torch.device("cpu")
        )
        self._generator = torch.Generator(device=generator_device)
        self._generator.manual_seed(self.seed)
        self._generator_device = generator_device

    @staticmethod
    def _normalize_seed(seed: Optional[int]) -> int:
        if seed is None:
            return secrets.randbits(63)
        return abs(int(seed)) % (1 << 63)

    def _resolved_target(self, device: DeviceLike) -> torch.device:
        requested = resolve_device(device)
        if requested.type in {"cpu", "cuda"} and requested != self._generator_device:
            raise RuntimeError(
                f"Sampling RNG belongs to {self._generator_device}, "
                f"but sampling tensor is on {requested}"
            )
        return requested

    def exponential(
        self,
        shape: Sequence[int],
        *,
        device: DeviceLike,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create exponential sampling noise from this request's private generator."""
        requested = self._resolved_target(device)
        normalized_shape = tuple(int(size) for size in shape)

        if requested.type in {"cpu", "cuda"}:
            return torch.empty(
                normalized_shape, device=requested, dtype=dtype
            ).exponential_(1, generator=self._generator)

        noise = torch.empty(
            normalized_shape, dtype=torch.float32, device="cpu"
        ).exponential_(1, generator=self._generator)
        return noise.to(device=requested, dtype=dtype)

    def exponential_parts(
        self,
        shapes: Sequence[Sequence[int]],
        *,
        device: DeviceLike,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        """Generate all per-token sampling noise with one RNG kernel/transfer."""
        normalized = [tuple(int(size) for size in shape) for shape in shapes]
        sizes = [math.prod(shape) for shape in normalized]
        flat = self.exponential(
            (sum(sizes),),
            device=device,
            dtype=dtype,
        )
        return tuple(
            part.view(shape)
            for part, shape in zip(flat.split(sizes), normalized)
        )

    def exponential_like(self, reference: torch.Tensor) -> torch.Tensor:
        return self.exponential(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
        )
