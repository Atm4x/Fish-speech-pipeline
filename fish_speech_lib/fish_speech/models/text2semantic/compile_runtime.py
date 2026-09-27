from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import wraps
import inspect
import sys
from threading import RLock
from typing import Any, Callable, Iterator, TypeVar

import torch
from loguru import logger

from fish_speech_lib.fish_speech.device import DeviceLike, resolve_device


F = TypeVar("F", bound=Callable)
_MISSING = object()
_COMPILER_STATE_LOCK = RLock()
_SHARED_MEMORY_IMPORTS = (
    "torch._inductor.autoheuristic.autoheuristic",
    "torch._inductor.kernel.mm_grouped",
)


def _device_context(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.device(device)
    return nullcontext()


def _is_inductor(compile_kwargs: dict[str, Any]) -> bool:
    return compile_kwargs.get("backend", "inductor") == "inductor"


def _cuda_arch(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return str(major * 10 + minor)


def _inductor_options(
    device: torch.device, compile_kwargs: dict[str, Any]
) -> dict[str, Any]:
    from torch._inductor import config as inductor_config

    if compile_kwargs.get("mode") is not None and compile_kwargs.get("options") is not None:
        raise RuntimeError("Either mode or options can be specified, but not both.")

    options = dict(compile_kwargs.get("options") or {})
    if hasattr(inductor_config, "coordinate_descent_tuning"):
        options.setdefault("coordinate_descent_tuning", True)

    triton_config = getattr(inductor_config, "triton", None)
    if triton_config is not None and hasattr(triton_config, "unique_kernel_names"):
        options.setdefault("triton.unique_kernel_names", True)

    if hasattr(inductor_config, "fx_graph_cache"):
        options.setdefault("fx_graph_cache", True)

    cuda_config = getattr(inductor_config, "cuda", None)
    if (
        device.type == "cuda"
        and cuda_config is not None
        and hasattr(cuda_config, "arch")
    ):
        options["cuda.arch"] = _cuda_arch(device)

    return options


def _fallback_inductor_config_patches(
    compile_kwargs: dict[str, Any], options: dict[str, Any]
) -> dict[str, Any]:
    patches: dict[str, Any] = {}
    mode = compile_kwargs.get("mode")
    if mode and mode != "default":
        from torch._inductor import list_mode_options

        dynamic = compile_kwargs.get("dynamic")
        try:
            mode_options = list_mode_options(mode, dynamic)
        except TypeError:
            mode_options = list_mode_options(mode)
        if mode_options:
            patches.update(mode_options)
    patches.update(options)
    return patches


def _clear_inductor_cuda_caches() -> None:
    try:
        from torch._inductor.codegen.cuda import cuda_env
    except (ImportError, AttributeError):
        return

    for name in ("get_cuda_arch", "is_datacenter_blackwell_arch"):
        cached = getattr(cuda_env, name, None)
        cache_clear = getattr(cached, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()


@contextmanager
def _compiler_worker_device_context(device: torch.device) -> Iterator[None]:
    """Align TorchInductor's worker-side CUDA device cache with Fish's device."""
    if device.type != "cuda":
        yield
        return

    try:
        from torch._dynamo import device_interface

        cache = getattr(device_interface, "caching_worker_current_devices", None)
        worker = device_interface.get_interface_for_device("cuda").Worker
        set_device = getattr(worker, "set_device", None)
    except (AttributeError, ImportError, NotImplementedError):
        yield
        return

    if not isinstance(cache, dict) or not callable(set_device):
        yield
        return

    previous = cache.get("cuda", _MISSING)
    set_device(device.index)
    try:
        yield
    finally:
        if previous is _MISSING:
            cache.pop("cuda", None)
        else:
            set_device(previous)


def _selected_gpu_shared_memory(device: torch.device) -> int:
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(device.index)
    return properties.get("max_shared_mem", 0)


@contextmanager
def _inductor_device_zero_compat_context(device: torch.device) -> Iterator[None]:
    """Scope workarounds for Inductor helpers that still query physical device 0."""
    if device.type != "cuda":
        yield
        return

    try:
        import torch._inductor.utils as inductor_utils
    except ImportError:
        yield
        return

    original = getattr(inductor_utils, "get_gpu_shared_memory", None)
    if not callable(original):
        yield
        return

    def selected_gpu_shared_memory() -> int:
        return _selected_gpu_shared_memory(device)

    setattr(inductor_utils, "get_gpu_shared_memory", selected_gpu_shared_memory)
    try:
        for module_name in _SHARED_MEMORY_IMPORTS:
            module = sys.modules.get(module_name)
            if module is not None and getattr(module, "get_gpu_shared_memory", None) is original:
                setattr(module, "get_gpu_shared_memory", selected_gpu_shared_memory)
        yield
    finally:
        setattr(inductor_utils, "get_gpu_shared_memory", original)
        for module_name in _SHARED_MEMORY_IMPORTS:
            module = sys.modules.get(module_name)
            if (
                module is not None
                and getattr(module, "get_gpu_shared_memory", None)
                is selected_gpu_shared_memory
            ):
                setattr(module, "get_gpu_shared_memory", original)


def _inductor_worker_current_device() -> int | None:
    try:
        from torch._dynamo import device_interface

        worker = device_interface.get_interface_for_device("cuda").Worker
        current_device = getattr(worker, "current_device", None)
        if callable(current_device):
            return int(current_device())
    except (AttributeError, ImportError, NotImplementedError, RuntimeError):
        pass
    return None


def _assert_cuda_compiler_device(device: torch.device, cuda_arch: str | None) -> None:
    if device.type != "cuda":
        return

    expected = device.index
    runtime_device = int(torch.cuda.current_device())
    if runtime_device != expected:
        raise RuntimeError(
            "Fish Speech compiler CUDA context mismatch: "
            f"requested cuda:{expected}, torch current device is cuda:{runtime_device}."
        )

    worker_device = _inductor_worker_current_device()
    if worker_device is not None and worker_device != expected:
        raise RuntimeError(
            "Fish Speech TorchInductor worker device mismatch: "
            f"requested cuda:{expected}, Inductor worker uses cuda:{worker_device}."
        )

    logger.info(
        "Fish Speech compiler device: requested={}, torch_current=cuda:{}, "
        "inductor_worker={}, cuda_arch={}, gpu={}",
        device,
        runtime_device,
        f"cuda:{worker_device}" if worker_device is not None else "unavailable",
        cuda_arch or "default",
        torch.cuda.get_device_name(device),
    )


def _validate_example_input_devices(
    example_inputs: Any, device: torch.device
) -> None:
    if device.type != "cuda":
        return

    cuda_indices: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if value.device.type == "cuda":
                cuda_indices.add(value.device.index)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, (tuple, list, set)):
            for item in value:
                visit(item)

    visit(example_inputs)
    unexpected = {index for index in cuda_indices if index != device.index}
    if unexpected:
        raise RuntimeError(
            "Fish Speech compile graph contains tensors on a different CUDA device: "
            f"requested {device}, graph devices={sorted(cuda_indices)}."
        )


@contextmanager
def _compile_backend_context(device: torch.device) -> Iterator[None]:
    """Scope process-global compiler compatibility state to backend compilation only."""
    if device.type != "cuda":
        yield
        return

    with _COMPILER_STATE_LOCK:
        with (
            _device_context(device),
            _compiler_worker_device_context(device),
            _inductor_device_zero_compat_context(device),
        ):
            _clear_inductor_cuda_caches()
            try:
                yield
            finally:
                _clear_inductor_cuda_caches()


class _DeviceBoundInductorBackend:
    """Inductor adapter that binds every initial compile/recompile to one GPU."""

    compiler_name = "fish_inductor"

    def __init__(
        self,
        device: torch.device,
        compile_kwargs: dict[str, Any],
    ) -> None:
        self.device = device
        self.options = _inductor_options(device, compile_kwargs)
        self.mode = compile_kwargs.get("mode")
        self.dynamic = compile_kwargs.get("dynamic")

        wrapper_type = getattr(torch, "_TorchCompileInductorWrapper", None)
        if wrapper_type is not None:
            self._compiler = wrapper_type(self.mode, self.options, self.dynamic)
            self.config_patches = dict(getattr(self._compiler, "config", self.options))
        else:
            self._compiler = None
            self.config_patches = _fallback_inductor_config_patches(
                compile_kwargs, self.options
            )
        self.cuda_arch = self.config_patches.get("cuda.arch")

    def __call__(self, graph_module, example_inputs):
        with _compile_backend_context(self.device):
            _assert_cuda_compiler_device(self.device, self.cuda_arch)
            _validate_example_input_devices(example_inputs, self.device)

            if self._compiler is not None:
                return self._compiler(graph_module, example_inputs)

            from torch._inductor.async_compile import maybe_warm_pool
            from torch._inductor.compile_fx import compile_fx

            maybe_warm_pool()
            parameters = inspect.signature(compile_fx).parameters
            if "config_patches" in parameters:
                return compile_fx(
                    graph_module,
                    example_inputs,
                    config_patches=self.config_patches,
                )

            import torch._inductor.config as inductor_config

            with inductor_config.patch(self.config_patches):
                return compile_fx(graph_module, example_inputs)

    def reset(self) -> None:
        reset = getattr(self._compiler, "reset", None)
        if callable(reset):
            reset()

    def get_compiler_config(self):
        get_config = getattr(self._compiler, "get_compiler_config", None)
        if callable(get_config):
            return get_config()
        return dict(self.config_patches)


def _mark_cudagraph_step(device: torch.device, compile_kwargs: dict[str, Any]) -> None:
    if device.type != "cuda" or compile_kwargs.get("mode") != "reduce-overhead":
        return

    marker = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if callable(marker):
        marker()


def compile_on_device(function: F, device: DeviceLike, **compile_kwargs) -> F:
    """Compile a callable with compiler and runtime placement bound to Fish's device."""
    resolved = resolve_device(device)
    original_compile_kwargs = dict(compile_kwargs)

    if _is_inductor(compile_kwargs):
        backend = _DeviceBoundInductorBackend(resolved, compile_kwargs)

        torch_compile_kwargs = dict(compile_kwargs)
        torch_compile_kwargs.pop("mode", None)
        torch_compile_kwargs.pop("options", None)
        torch_compile_kwargs["backend"] = backend
    else:
        torch_compile_kwargs = compile_kwargs

    with _device_context(resolved):
        compiled = torch.compile(function, **torch_compile_kwargs)

    @wraps(function)
    def invoke(*args, **kwargs):
        with _device_context(resolved):
            _mark_cudagraph_step(resolved, original_compile_kwargs)
            return compiled(*args, **kwargs)

    invoke._fish_compile_device = resolved
    invoke._fish_compiled = True
    if _is_inductor(original_compile_kwargs):
        invoke._fish_inductor_options = dict(backend.config_patches)
    return invoke
