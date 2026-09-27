from __future__ import annotations

from typing import Union

import torch


DeviceLike = Union[str, torch.device]


def resolve_device(
    device: DeviceLike, *, fallback_to_cpu: bool = False
) -> torch.device:
    """Normalize and validate an inference device without current-device state."""
    requested = torch.device(device)

    if requested.type == "cuda":
        if not torch.cuda.is_available():
            if fallback_to_cpu and requested.index is None:
                return torch.device("cpu")
            raise RuntimeError(
                f"CUDA device requested but CUDA is unavailable: {device}"
            )

        index = 0 if requested.index is None else requested.index
        device_count = torch.cuda.device_count()
        if index < 0 or index >= device_count:
            raise ValueError(
                f"CUDA device index {index} is unavailable; "
                f"detected {device_count} CUDA device(s)"
            )
        return torch.device("cuda", index)

    if requested.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            if fallback_to_cpu and requested.index is None:
                return torch.device("cpu")
            raise RuntimeError(
                f"MPS device requested but MPS is unavailable: {device}"
            )
        if requested.index not in (None, 0):
            raise ValueError(f"Unsupported MPS device index: {requested.index}")
        return torch.device("mps")

    if requested.type == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported inference device: {device}")


def module_device(module: torch.nn.Module) -> torch.device:
    """Return the single device owned by a non-sharded module."""
    devices = {parameter.device for parameter in module.parameters()}
    devices.update(buffer.device for buffer in module.buffers())

    if not devices:
        raise RuntimeError(
            "Cannot determine device for a module without parameters or buffers"
        )
    if len(devices) != 1:
        formatted = ", ".join(sorted(str(device) for device in devices))
        raise RuntimeError(
            f"Expected a single-device model, found tensors on: {formatted}"
        )
    return next(iter(devices))
