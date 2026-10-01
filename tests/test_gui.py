import io
import os
import sys
import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gui import runtime, server, worker


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.calibration = self.root / "calibration.json"
        self.calibration.write_text("[]", encoding="utf-8")
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.patch_runs = patch.object(server, "RUNS_DIR", self.runs)
        self.patch_runs.start()

    def tearDown(self):
        self.patch_runs.stop()
        self.temp_dir.cleanup()

    def base_config(self):
        return {
            "model_path": str(self.model),
            "calibration_path": str(self.calibration),
            "output_path": "gui/runs/model.rkllm",
            "platform": "RK3588",
            "quantization": "W8A8",
            "device": "cpu",
            "dtype": "float32",
            "max_context": 4096,
        }

    def test_relative_paths_are_resolved_from_project_root(self):
        config, _ = server.make_config(self.base_config())
        self.assertEqual(config["output_path"], str(server.ROOT_DIR / "gui/runs/model.rkllm"))

    def test_platform_default_quantization_is_used(self):
        data = self.base_config()
        data.update({"platform": "RK3576", "quantization": ""})
        config, _ = server.make_config(data)
        self.assertEqual(config["quantized_dtype"], "w4a16")
        self.assertEqual(config["optimization_level"], 0)

    def test_oversized_integer_is_rejected_as_user_input(self):
        with self.assertRaisesRegex(ValueError, "最大上下文长度"):
            server.integer_value(10**400, "最大上下文长度")

    def test_output_cannot_be_inside_model_directory(self):
        data = self.base_config()
        data["output_path"] = str(self.model / "result.rkllm")
        with self.assertRaisesRegex(ValueError, "模型目录"):
            server.make_config(data)

    def test_output_cannot_overwrite_calibration_file(self):
        data = self.base_config()
        data["output_path"] = str(self.calibration)
        with self.assertRaisesRegex(ValueError, "校准数据"):
            server.make_config(data)

    def test_output_symlink_is_rejected(self):
        data = self.base_config()
        target = self.root / "actual-output.rkllm"
        target.write_bytes(b"existing")
        link = self.root / "output-link.rkllm"
        link.symlink_to(target)
        data["output_path"] = str(link)
        with self.assertRaisesRegex(ValueError, "普通文件"):
            server.make_config(data)

    def test_visual_outputs_cannot_be_hardlinks_to_each_other(self):
        data = self.base_config()
        first = self.root / "vision.onnx"
        second = self.root / "vision.rknn"
        first.write_bytes(b"existing")
        second.hardlink_to(first)
        data.update(
            {
                "conversion_mode": "视觉模型",
                "vision_model": "Qwen3-VL",
                "vision_batch_size": 1,
                "vision_height": 448,
                "vision_width": 448,
                "vision_onnx_path": str(first),
                "vision_rknn_path": str(second),
            }
        )
        with self.assertRaisesRegex(ValueError, "三个不同的文件"):
            server.make_config(data)

    def test_visual_config_has_three_distinct_outputs(self):
        data = self.base_config()
        data.update(
            {
                "conversion_mode": "视觉模型",
                "vision_model": "Qwen3-VL",
                "vision_batch_size": 1,
                "vision_height": 448,
                "vision_width": 448,
                "vision_onnx_path": "gui/runs/vision/model.onnx",
                "vision_rknn_path": "gui/runs/vision/model.rknn",
            }
        )
        config, _ = server.make_config(data)
        self.assertEqual(len(server.output_paths(config)), 3)


class JobOutputTests(unittest.TestCase):
    def make_job(self, config, process, before=None, stop_requested=False):
        return {
            "id": "test-job",
            "process": process,
            "lines": [],
            "config": config,
            "config_path": "test-config.json",
            "status": "转换中",
            "status_code": "running",
            "stop_requested": stop_requested,
            "outputs_before": before or {},
            "output": "",
            "outputs": [],
        }

    def test_completed_job_reports_all_visual_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = [root / "model.rkllm", root / "vision.onnx", root / "vision.rknn"]
            config = {
                "conversion_mode": "vision",
                "output_path": str(outputs[0]),
                "vision_onnx_path": str(outputs[1]),
                "vision_rknn_path": str(outputs[2]),
            }
            for output in outputs:
                output.write_bytes(b"generated")
            process = type("Process", (), {"stdout": io.StringIO("done\n"), "wait": lambda self: 0})()
            old_job = server._job
            server._job = self.make_job(config, process)
            try:
                server.read_job_output(process)
                snapshot = server.job_snapshot()
            finally:
                server._job = old_job
            self.assertEqual(snapshot["status_code"], "completed")
            self.assertEqual(snapshot["outputs"], [str(output) for output in outputs])

    def test_unchanged_old_output_does_not_count_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.rkllm"
            output.write_bytes(b"old")
            config = {"conversion_mode": "text", "output_path": str(output)}
            before = {str(output): server.file_signature(output)}
            process = type("Process", (), {"stdout": io.StringIO(), "wait": lambda self: 0})()
            old_job = server._job
            server._job = self.make_job(config, process, before=before)
            try:
                server.read_job_output(process)
                snapshot = server.job_snapshot()
            finally:
                server._job = old_job
            self.assertEqual(snapshot["status_code"], "failed")
            self.assertIn("未生成", snapshot["log"])

    def test_stopped_job_is_not_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"conversion_mode": "text", "output_path": str(Path(directory) / "model.rkllm")}
            process = type("Process", (), {"stdout": io.StringIO(), "wait": lambda self: -15})()
            old_job = server._job
            server._job = self.make_job(config, process, stop_requested=True)
            try:
                server.read_job_output(process)
                snapshot = server.job_snapshot()
            finally:
                server._job = old_job
        self.assertEqual(snapshot["status_code"], "cancelled")
        self.assertEqual(snapshot["status"], "已停止")

    def test_job_log_is_returned_incrementally(self):
        job = self.make_job({"conversion_mode": "text", "output_path": "result.rkllm"}, None)
        old_job = server._job
        server._job = job
        try:
            server.append_job_log(job, "first\nsecond")
            first = server.job_snapshot()
            server.append_job_log(job, "third")
            second = server.job_snapshot(after=first["log_offset"])
        finally:
            server._job = old_job
        self.assertEqual((first["log"], first["log_offset"]), ("first\nsecond", 2))
        self.assertEqual((second["log"], second["log_offset"]), ("third", 3))

    def test_stop_handles_parent_exit_before_log_reader_finishes(self):
        process = type("Process", (), {"poll": lambda self: 0})()
        job = self.make_job({"conversion_mode": "text", "output_path": "result.rkllm"}, process)
        job["finished"] = threading.Event()
        old_job = server._job
        server._job = job
        try:
            with patch.object(server, "kill_process_group") as kill_group:
                snapshot = server.stop_job()
            kill_group.assert_called_once_with(process, wait=False)
            self.assertEqual(snapshot["status_code"], "stopping")
        finally:
            server._job = old_job

    def test_start_job_does_not_write_config_when_busy(self):
        process = type("Process", (), {"poll": lambda self: None})()
        job = self.make_job({"conversion_mode": "text", "output_path": "result.rkllm"}, process)
        old_job = server._job
        server._job = job
        try:
            with self.assertRaisesRegex(ValueError, "已有转换任务"):
                server.start_job({})
        finally:
            server._job = old_job


class RequestAndCleanupTests(unittest.TestCase):
    def test_only_local_browser_requests_are_accepted(self):
        headers = {"Host": "localhost:7860", "Sec-Fetch-Site": "same-origin"}
        server.validate_local_request(headers, 7860)
        with self.assertRaisesRegex(ValueError, "本机"):
            server.validate_local_request({**headers, "Host": "example.com:7860"}, 7860)
        with self.assertRaisesRegex(ValueError, "本机"):
            server.validate_local_request({**headers, "Origin": "https://example.com"}, 7860)
        with self.assertRaisesRegex(ValueError, "跨站"):
            server.validate_local_request({**headers, "Sec-Fetch-Site": "cross-site"}, 7860)

    def test_cleanup_keeps_recent_run_metadata_and_visual_workspaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            visual = root / "visual"
            visual.mkdir()
            configs = [root / f"{index}.json" for index in range(12)]
            workspaces = [visual / f"{index}" for index in range(5)]
            for path in configs:
                path.write_text("{}", encoding="utf-8")
            for path in workspaces:
                path.mkdir()
            outputs = root / "outputs"
            outputs.mkdir()
            old_runs = server.RUNS_DIR
            server.RUNS_DIR = root
            try:
                server.cleanup_runs()
            finally:
                server.RUNS_DIR = old_runs
            self.assertEqual(len(list(root.glob("*.json"))), 10)
            self.assertEqual(len(list(visual.iterdir())), 3)
            self.assertTrue(outputs.exists())


class RuntimeTests(unittest.TestCase):
    def test_explicit_env_vars_override_defaults(self):
        with patch.dict(os.environ, {"RKLLM_WORKBENCH_LLM_PYTHON": "/custom/llm/python", "RKLLM_WORKBENCH_VISION_PYTHON": "/custom/vision/python"}):
            self.assertEqual(runtime.llm_python(), Path("/custom/llm/python"))
            self.assertEqual(runtime.vision_python(), Path("/custom/vision/python"))

    def test_managed_env_is_detected_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            llm_bin = home / "micromamba/envs/rknn-llm-workbench-llm/bin/python"
            vision_bin = home / "micromamba/envs/rknn-llm-workbench-vision/bin/python"
            llm_bin.parent.mkdir(parents=True)
            vision_bin.parent.mkdir(parents=True)
            llm_bin.write_text("")
            vision_bin.write_text("")
            with patch.dict(os.environ, {"RKLLM_WORKBENCH_HOME": str(home)}, clear=False):
                with patch.dict(os.environ, {"RKLLM_WORKBENCH_LLM_PYTHON": "", "RKLLM_WORKBENCH_VISION_PYTHON": "", "RKLLM_WORKBENCH_PYTHON": ""}):
                    self.assertEqual(runtime.llm_python(), llm_bin)
                    self.assertEqual(runtime.vision_python(), vision_bin)

    def test_falls_back_to_sys_executable_when_no_managed_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"RKLLM_WORKBENCH_HOME": temp_dir}, clear=False):
                with patch.dict(os.environ, {"RKLLM_WORKBENCH_LLM_PYTHON": "", "RKLLM_WORKBENCH_VISION_PYTHON": "", "RKLLM_WORKBENCH_PYTHON": ""}):
                    self.assertEqual(runtime.llm_python(), Path(sys.executable))
                    self.assertEqual(runtime.vision_python(), Path(sys.executable))


class WorkerTests(unittest.TestCase):
    def test_visual_workspace_cleaned_up_on_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir) / "runs"
            runs_dir.mkdir()
            fake_llm = Path(temp_dir) / "python_llm"
            fake_llm.write_text("")
            fake_model = Path(temp_dir) / "fake_model"
            fake_model.mkdir()

            config = {
                "model_path": str(fake_model),
                "vision_model_name": "qwen2_5-vl-3b",
                "vision_model_type": "qwen2.5vl",
                "vision_batch_size": 1,
                "vision_height": 448,
                "vision_width": 448,
                "device": "cpu",
                "target_platform": "rk3588",
                "vision_onnx_path": str(Path(temp_dir) / "out.onnx"),
                "vision_rknn_path": str(Path(temp_dir) / "out.rknn"),
            }

            created_workspaces = []

            def fake_prepare():
                ws = runs_dir / "visual/test_ws"
                ws.mkdir(parents=True, exist_ok=True)
                created_workspaces.append(ws)
                return ws

            with patch.object(worker, "RUNS_DIR", runs_dir), \
                 patch.object(worker, "llm_python", return_value=fake_llm), \
                 patch.object(worker, "vision_python", return_value=fake_llm), \
                 patch.object(worker, "_check_visual_dependencies"), \
                 patch.object(worker, "_prepare_visual_workspace", side_effect=fake_prepare), \
                 patch.object(worker, "_run_command", side_effect=RuntimeError("export failed")):
                with self.assertRaisesRegex(RuntimeError, "export failed"):
                    worker._run_visual(config)

            self.assertEqual(len(created_workspaces), 1)
            self.assertFalse(created_workspaces[0].exists())


if __name__ == "__main__":
    unittest.main()
