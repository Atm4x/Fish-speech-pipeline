import ast
import importlib.util
import sys
import types
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


for package_name in (
    "fish_speech_lib",
    "fish_speech_lib.fish_speech",
    "fish_speech_lib.fish_speech.models",
    "fish_speech_lib.fish_speech.models.text2semantic",
):
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules.setdefault(package_name, package)

DEVICE_MODULE = load_module(
    "fish_speech_lib.fish_speech.device",
    ROOT / "fish_speech_lib/fish_speech/device.py",
)
sys.modules["fish_speech_lib.fish_speech.device"] = DEVICE_MODULE
SAMPLING_MODULE = load_module(
    "fish_speech_lib.fish_speech.models.text2semantic.sampling",
    ROOT / "fish_speech_lib/fish_speech/models/text2semantic/sampling.py",
)
sys.modules["fish_speech_lib.fish_speech.models.text2semantic.sampling"] = SAMPLING_MODULE
COMPILE_MODULE = load_module(
    "fish_speech_lib.fish_speech.models.text2semantic.compile_runtime",
    ROOT / "fish_speech_lib/fish_speech/models/text2semantic/compile_runtime.py",
)


class DeviceResolutionTests(unittest.TestCase):
    @patch.object(torch.cuda, "device_count", return_value=2)
    @patch.object(torch.cuda, "is_available", return_value=True)
    def test_bare_cuda_is_normalized_to_first_visible_gpu(self, _available, _count):
        self.assertEqual(DEVICE_MODULE.resolve_device("cuda"), torch.device("cuda:0"))

    @patch.object(torch.cuda, "device_count", return_value=2)
    @patch.object(torch.cuda, "is_available", return_value=True)
    def test_explicit_cuda_index_is_preserved(self, _available, _count):
        self.assertEqual(DEVICE_MODULE.resolve_device("cuda:1"), torch.device("cuda:1"))

    @patch.object(torch.cuda, "device_count", return_value=1)
    @patch.object(torch.cuda, "is_available", return_value=True)
    def test_invalid_cuda_index_is_rejected(self, _available, _count):
        with self.assertRaises(ValueError):
            DEVICE_MODULE.resolve_device("cuda:1")

    @patch.object(torch.cuda, "is_available", return_value=False)
    def test_implicit_cuda_can_fallback_to_cpu(self, _available):
        self.assertEqual(
            DEVICE_MODULE.resolve_device("cuda", fallback_to_cpu=True),
            torch.device("cpu"),
        )

    @patch.object(torch.cuda, "is_available", return_value=False)
    def test_explicit_cuda_does_not_silently_fallback(self, _available):
        with self.assertRaises(RuntimeError):
            DEVICE_MODULE.resolve_device("cuda:1", fallback_to_cpu=True)


class SamplingRngTests(unittest.TestCase):
    def test_seeded_requests_are_reproducible(self):
        reference = torch.ones(32)
        first = SAMPLING_MODULE.SamplingRng("cpu", 123).exponential_like(reference)
        second = SAMPLING_MODULE.SamplingRng("cpu", 123).exponential_like(reference)
        self.assertTrue(torch.equal(first, second))

    def test_partitioned_sampling_noise_is_reproducible(self):
        first = SAMPLING_MODULE.SamplingRng("cpu", 555).exponential_parts(
            [(7,), (2, 3)], device="cpu", dtype=torch.float32
        )
        second = SAMPLING_MODULE.SamplingRng("cpu", 555).exponential_parts(
            [(7,), (2, 3)], device="cpu", dtype=torch.float32
        )
        self.assertEqual([tuple(t.shape) for t in first], [(7,), (2, 3)])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))

    def test_sampling_does_not_mutate_default_torch_rng(self):
        before = torch.random.get_rng_state().clone()
        rng = SAMPLING_MODULE.SamplingRng("cpu", 321)
        rng.exponential_like(torch.ones(32))
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))


class CompileRuntimeTests(unittest.TestCase):
    def test_cuda_compile_and_lazy_invocation_use_scoped_selected_device(self):
        events = []

        class DeviceContext:
            def __enter__(self):
                events.append("enter:cuda:1")

            def __exit__(self, exc_type, exc, tb):
                events.append("exit:cuda:1")

        def fake_compile(function, **kwargs):
            events.append("compile")

            def compiled(*args, **call_kwargs):
                events.append("compiled-call")
                return function(*args, **call_kwargs)

            return compiled

        with (
            patch.object(COMPILE_MODULE, "resolve_device", return_value=torch.device("cuda:1")),
            patch.object(
                COMPILE_MODULE,
                "_inductor_options",
                return_value={"cuda.arch": "89"},
            ),
            patch.object(COMPILE_MODULE.torch.cuda, "device", return_value=DeviceContext()),
            patch.object(COMPILE_MODULE.torch, "compile", side_effect=fake_compile),
            patch.object(
                COMPILE_MODULE.torch.compiler,
                "cudagraph_mark_step_begin",
                side_effect=lambda: events.append("mark-step"),
            ),
        ):
            compiled = COMPILE_MODULE.compile_on_device(
                lambda value: value + 1,
                "cuda:1",
                fullgraph=True,
                backend="inductor",
                mode="reduce-overhead",
            )
            self.assertEqual(compiled(4), 5)

        self.assertEqual(
            events,
            [
                "enter:cuda:1",
                "compile",
                "exit:cuda:1",
                "enter:cuda:1",
                "mark-step",
                "compiled-call",
                "exit:cuda:1",
            ],
        )

    def test_real_cpu_compile_accepts_external_sampling_noise(self):
        def token_math(logits, noise):
            probs = torch.softmax(logits, dim=-1)
            return torch.argmax(probs / noise, dim=-1)

        compiled = COMPILE_MODULE.compile_on_device(
            token_math,
            "cpu",
            fullgraph=True,
            backend="aot_eager",
            mode=None,
        )
        logits = torch.randn(2, 32)
        noise = torch.rand(2, 32).clamp_min_(1e-4)
        self.assertTrue(torch.equal(token_math(logits, noise), compiled(logits, noise)))

    def test_inductor_worker_device_cache_is_scoped_and_restored(self):
        from torch._dynamo import device_interface

        cache = device_interface.caching_worker_current_devices
        original = dict(cache)
        try:
            cache.clear()
            cache["cuda"] = 0
            with COMPILE_MODULE._compiler_worker_device_context(
                torch.device("cuda:1")
            ):
                self.assertEqual(cache["cuda"], 1)
            self.assertEqual(cache["cuda"], 0)

            cache.pop("cuda")
            with COMPILE_MODULE._compiler_worker_device_context(
                torch.device("cuda:1")
            ):
                self.assertEqual(cache["cuda"], 1)
            self.assertNotIn("cuda", cache)
        finally:
            cache.clear()
            cache.update(original)

    @patch.object(torch.cuda, "get_device_capability", return_value=(8, 9))
    def test_inductor_config_pins_selected_cuda_arch(self, _capability):
        backend = COMPILE_MODULE._DeviceBoundInductorBackend(
            torch.device("cuda:1"),
            {"backend": "inductor", "mode": "reduce-overhead"},
        )
        self.assertEqual(backend.config_patches["cuda.arch"], "89")
        self.assertTrue(backend.config_patches.get("triton.cudagraphs"))

    def test_device_bound_backend_scopes_every_backend_compile(self):
        import torch._inductor.async_compile as async_compile
        import torch._inductor.compile_fx as compile_fx_module

        events = []

        class BackendContext:
            def __enter__(self):
                events.append("enter-backend")

            def __exit__(self, exc_type, exc, tb):
                events.append("exit-backend")

        def fake_compile_fx(graph_module, example_inputs, config_patches=None):
            events.append(("compile-fx", dict(config_patches or {})))
            return "compiled-graph"

        with patch.object(
            COMPILE_MODULE,
            "_inductor_options",
            return_value={"cuda.arch": "89"},
        ):
            backend = COMPILE_MODULE._DeviceBoundInductorBackend(
                torch.device("cuda:1"),
                {"backend": "inductor", "mode": "reduce-overhead"},
            )

        with (
            patch.object(COMPILE_MODULE, "_compile_backend_context", return_value=BackendContext()),
            patch.object(COMPILE_MODULE, "_assert_cuda_compiler_device"),
            patch.object(COMPILE_MODULE, "_validate_example_input_devices"),
            patch.object(
                async_compile,
                "maybe_warm_pool",
                side_effect=lambda: events.append("warm-pool"),
            ),
            patch.object(compile_fx_module, "compile_fx", new=fake_compile_fx),
        ):
            self.assertEqual(backend(object(), []), "compiled-graph")
            self.assertEqual(backend(object(), []), "compiled-graph")

        self.assertEqual(events.count("enter-backend"), 2)
        self.assertEqual(events.count("exit-backend"), 2)
        compile_events = [event for event in events if isinstance(event, tuple)]
        self.assertEqual(len(compile_events), 2)
        self.assertTrue(all(event[1]["cuda.arch"] == "89" for event in compile_events))

    def test_compiler_device_mismatch_fails_before_codegen(self):
        with (
            patch.object(COMPILE_MODULE.torch.cuda, "current_device", return_value=0),
            patch.object(COMPILE_MODULE, "_inductor_worker_current_device", return_value=1),
        ):
            with self.assertRaisesRegex(RuntimeError, "compiler CUDA context mismatch"):
                COMPILE_MODULE._assert_cuda_compiler_device(
                    torch.device("cuda:1"), "89"
                )

    def test_device_zero_compat_patch_is_scoped(self):
        import torch._inductor.utils as inductor_utils

        module_name = COMPILE_MODULE._SHARED_MEMORY_IMPORTS[0]
        previous_module = sys.modules.get(module_name)
        fake_module = types.ModuleType(module_name)
        original = inductor_utils.get_gpu_shared_memory
        fake_module.get_gpu_shared_memory = original
        sys.modules[module_name] = fake_module
        try:
            with COMPILE_MODULE._inductor_device_zero_compat_context(
                torch.device("cuda:1")
            ):
                patched = inductor_utils.get_gpu_shared_memory
                self.assertIsNot(patched, original)
                self.assertIs(fake_module.get_gpu_shared_memory, patched)
            self.assertIs(inductor_utils.get_gpu_shared_memory, original)
            self.assertIs(fake_module.get_gpu_shared_memory, original)
        finally:
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module

    def test_real_cpu_inductor_backend_smoke(self):
        compiled = COMPILE_MODULE.compile_on_device(
            lambda value: torch.sin(value) + torch.cos(value),
            "cpu",
            fullgraph=True,
            backend="inductor",
        )
        value = torch.randn(8)
        expected = torch.sin(value) + torch.cos(value)
        self.assertTrue(torch.allclose(compiled(value), expected))

    def test_compile_runtime_never_calls_global_cuda_set_device(self):
        source = (
            ROOT
            / "fish_speech_lib/fish_speech/models/text2semantic/compile_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("torch.cuda.set_device", source)
        self.assertIn("torch.cuda.device", source)


class DevicePipelineStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inference_path = (
            ROOT / "fish_speech_lib/fish_speech/models/text2semantic/inference.py"
        )
        cls.inference_source = cls.inference_path.read_text(encoding="utf-8")
        cls.inference_tree = ast.parse(cls.inference_source)

    def test_text_semantic_pipeline_has_no_current_device_mutation(self):
        for node in ast.walk(self.inference_tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "set_device":
                self.fail("text-semantic pipeline must not mutate current CUDA device")
            if node.func.attr in {"synchronize", "max_memory_reserved"}:
                self.assertGreaterEqual(
                    len(node.args),
                    1,
                    f"{node.func.attr} must receive an explicit device",
                )

        self.assertNotIn("torch._inductor.config", self.inference_source)
        self.assertNotIn("cudagraph_mark_step_begin", self.inference_source)
        self.assertNotIn("with torch.device", self.inference_source)
        self.assertNotIn("TOKENIZERS_PARALLELISM", self.inference_source)

    def test_sampling_rng_does_not_write_global_rng_state(self):
        source = (
            ROOT
            / "fish_speech_lib/fish_speech/models/text2semantic/sampling.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("set_rng_state", source)
        self.assertNotIn("manual_seed_all", source)
        self.assertNotIn("torch.manual_seed", source)

    def test_compile_boundary_is_full_token_decode_path(self):
        self.assertIn("compiled_impl = compile_on_device(decode_impl, device", self.inference_source)
        self.assertIn('"mode": "reduce-overhead" if is_cuda else None', self.inference_source)
        self.assertNotIn("model.forward_generate = torch.compile", self.inference_source)
        self.assertNotIn("model.forward_generate_fast = torch.compile", self.inference_source)

        for name in (
            "_decode_one_token_ar_impl",
            "_decode_one_token_naive_impl",
            "_decode_one_token_ar_agent_impl",
            "_decode_one_token_naive_agent_impl",
        ):
            self.assertIn(name, self.inference_source)
        self.assertIn("sampling_noise", self.inference_source)

    def test_worker_owns_queue_by_composition(self):
        worker = next(
            node
            for node in self.inference_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TextSemanticWorker"
        )
        self.assertEqual(worker.bases, [])
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "queue"
                and node.func.attr == "Queue"
                for node in ast.walk(worker)
            )
        )

    def test_kv_cache_allocations_receive_explicit_device(self):
        source_path = ROOT / "fish_speech_lib/fish_speech/models/text2semantic/llama.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        kv_cache = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "KVCache"
        )
        init = next(
            node
            for node in kv_cache.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        zeros_calls = [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "zeros"
        ]
        self.assertEqual(len(zeros_calls), 2)
        for call in zeros_calls:
            self.assertIn("device", {keyword.arg for keyword in call.keywords})

    def test_engine_does_not_flush_process_wide_cuda_cache(self):
        source_path = ROOT / "fish_speech_lib/fish_speech/inference_engine/__init__.py"
        source = source_path.read_text(encoding="utf-8")
        self.assertNotIn("empty_cache", source)
        self.assertIn("llama_worker.submit", source)

    def test_vqgan_does_not_require_project_root_marker(self):
        vq_source = (
            ROOT / "fish_speech_lib/fish_speech/models/vqgan/inference.py"
        ).read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("pyrootutils", vq_source)
        self.assertNotIn("setup_root", vq_source)
        self.assertNotIn("pyrootutils", pyproject)


if __name__ == "__main__":
    unittest.main()
