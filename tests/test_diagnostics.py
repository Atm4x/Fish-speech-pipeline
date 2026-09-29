# Copyright 2025-2026 Atm4x
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class DeviceSnapshotTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "fish_speech_lib/fish_speech/diagnostics.py"
        spec = importlib.util.spec_from_file_location("_diagnostics_test", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.module.DIAGNOSTIC_LOG = Path(self.directory.name) / "devices.json"

    def test_snapshot_replaces_events_and_preserves_each_device(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            for index in range(100):
                self.module.record_device_event("configured", device="cuda:0", value=index)
            self.module.record_device_event("configured", device="cuda:1", value=123)
        data = json.loads(self.module.DIAGNOSTIC_LOG.read_text(encoding="utf-8"))
        self.assertEqual(set(data["devices"]), {"cuda:0", "cuda:1"})
        self.assertEqual(data["devices"]["cuda:0"]["fields"]["value"], 99)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(list(Path(self.directory.name).iterdir()), [self.module.DIAGNOSTIC_LOG])

    def test_unwritable_temp_does_not_interrupt_inference(self):
        with patch.object(self.module.tempfile, "NamedTemporaryFile", side_effect=OSError("denied")):
            self.module.record_device_event("configured", device="cpu")
        self.assertFalse(self.module.DIAGNOSTIC_LOG.exists())
