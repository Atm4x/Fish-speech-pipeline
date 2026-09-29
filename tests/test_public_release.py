# Copyright 2025-2026 Atm4x
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "fish_speech_lib"


class ApacheLicenseTests(unittest.TestCase):
    def test_license_is_apache_2(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", text)
        self.assertIn("Version 2.0, January 2004", text)
        self.assertIn("END OF TERMS AND CONDITIONS", text)

    def test_every_python_source_has_apache_spdx_marker(self):
        missing = []
        for path in PACKAGE.rglob("*.py"):
            head = path.read_text(encoding="utf-8")[:1000]
            if "SPDX-License-Identifier: Apache-2.0" not in head:
                missing.append(str(path.relative_to(ROOT)))
        self.assertEqual(missing, [])

    def test_notice_keeps_upstream_attribution(self):
        notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("Atm4x", notice)
        self.assertIn("Fish Audio Authors", notice)
        self.assertIn("Apache License", notice)


class CleanCompilerIntegrationTests(unittest.TestCase):
    def test_no_compat_monkey_patch_package_is_shipped(self):
        self.assertFalse((PACKAGE / "fish_speech/compat").exists())

    def test_no_cuda_visible_devices_remapping(self):
        sources = "\n".join(
            p.read_text(encoding="utf-8") for p in PACKAGE.rglob("*.py")
        )
        self.assertNotIn("CUDA_VISIBLE_DEVICES", sources)

    def test_no_cuda_set_device_calls(self):
        offenders = []
        for path in PACKAGE.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "set_device"
                ):
                    offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_no_assignments_into_torch_compiler_private_modules(self):
        offenders = []
        for path in PACKAGE.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                elif isinstance(node, ast.AugAssign):
                    targets = [node.target]
                for target in targets:
                    if not isinstance(target, ast.Attribute):
                        continue
                    parts = []
                    value = target
                    while isinstance(value, ast.Attribute):
                        parts.append(value.attr)
                        value = value.value
                    if isinstance(value, ast.Name):
                        parts.append(value.id)
                    dotted = ".".join(reversed(parts))
                    if dotted.startswith("torch._inductor") or dotted.startswith("torch._dynamo"):
                        offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_compile_runtime_is_scoped_not_process_global(self):
        source = (PACKAGE / "fish_speech/models/text2semantic/compile_runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("torch.cuda.device", source)
        self.assertIn("inductor_config.patch", source)
        self.assertNotIn("set_device", source)
        self.assertNotIn("get_device_capability(0)", source)
        self.assertNotIn("get_device_properties(0)", source)


class PublicPackageTests(unittest.TestCase):
    def test_python_requirement_matches_source_syntax(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.10"', pyproject)

    def test_broken_legacy_console_entrypoint_is_not_published(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("fish_speech_pipe.inference:main", pyproject)
        self.assertNotIn("[project.scripts]", pyproject)

    def test_package_init_does_not_mutate_environment(self):
        source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("os.environ", source)
        self.assertNotIn("import *", source)

    def test_legacy_modules_use_bundled_fish_speech(self):
        for relative in ("model.py", "utils.py"):
            source = (PACKAGE / relative).read_text(encoding="utf-8")
            self.assertNotIn("from fish_speech.", source)
            self.assertNotIn("import fish_speech.", source)

    def test_legacy_model_manager_does_not_collapse_cuda_index(self):
        source = (PACKAGE / "model.py").read_text(encoding="utf-8")
        self.assertIn("resolve_device", source)
        self.assertNotIn('"cuda" if', source)
        self.assertNotIn("startswith(\"cuda\")", source)

    def test_debug_audio_array_print_is_removed(self):
        source = (PACKAGE / "inference.py").read_text(encoding="utf-8")
        self.assertNotIn('print("test:"', source)


if __name__ == "__main__":
    unittest.main()
