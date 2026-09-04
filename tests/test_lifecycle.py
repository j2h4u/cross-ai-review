import asyncio
import os
import signal
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cross_ai


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "cross-ai"


def _write_fake_opencode(path: Path, *, hang: bool) -> None:
    behavior = (
        "marker = os.environ.get('CROSS_AI_TEST_MARKER')\n"
        "if marker:\n"
        "    Path(marker).write_text(os.environ['TMPDIR'], encoding='utf-8')\n"
        "child = subprocess.Popen(['sleep', '60'])\n"
        "Path(os.environ['CROSS_AI_TEST_CHILD_PID']).write_text(str(child.pid), encoding='utf-8')\n"
        "time.sleep(60)\n"
        if hang
        else ""
    )
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import subprocess\n"
        "import time\n"
        "from pathlib import Path\n"
        "assert os.environ['TMPDIR'] == os.environ['BUN_TMPDIR']\n"
        "Path(os.environ['TMPDIR'], 'libopentui.so').write_bytes(b'x' * 4096)\n"
        f"{behavior}\n"
        "message = 'Finding: lifecycle test output ' + ('x' * 240)\n"
        "print(json.dumps({'type': 'text', 'part': {'messageID': 'm1', 'text': message}}))\n"
        "print(json.dumps({'type': 'step_finish', 'part': {'messageID': 'm1', 'reason': 'stop'}}))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _process_is_live(pid: int) -> bool:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return False
    return "State:\tZ" not in status_path.read_text(encoding="utf-8")


class CrossAiLifecycleTests(unittest.TestCase):
    def test_permission_contract(self) -> None:
        permissions = cross_ai._opencode_permissions(Path("/srv/example"))
        self.assertEqual(permissions["read"]["*"], "allow")
        for tool in ("glob", "grep", "list"):
            self.assertEqual(permissions[tool], "allow")
        for tool in (
            "edit",
            "write",
            "task",
            "question",
            "webfetch",
            "websearch",
            "skill",
        ):
            self.assertEqual(permissions[tool], "deny")
        self.assertEqual(permissions["external_directory"], {"*": "deny"})
        bash = permissions["bash"]
        self.assertEqual(bash["*"], "deny")
        for command in ("diff", "status", "log", "show"):
            self.assertEqual(bash[f"git {command}*"], "allow")
            self.assertEqual(
                bash[f"git -C /srv/example {command}*"],
                "allow",
            )

    def test_temp_environment_limit_and_cleanup(self) -> None:
        path = cross_ai._create_temporary_directory()
        try:
            self.assertEqual(path.parent, Path("/tmp"))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
            env = cross_ai._opencode_environment(Path("/tmp/config"), path)
            self.assertEqual(env["TMPDIR"], str(path))
            self.assertEqual(env["BUN_TMPDIR"], str(path))
            (path / "oversized.so").write_bytes(b"xx")
            with self.assertRaises(cross_ai.TemporaryDirectoryLimitExceeded):
                asyncio.run(
                    cross_ai._monitor_temporary_directory(path, limit_bytes=1)
                )
        finally:
            cross_ai._remove_temporary_directory(path)
        self.assertFalse(path.exists())

    def test_success_preserves_report_and_removes_run_temp(self) -> None:
        before = set(Path("/tmp").glob(f"cross-ai-{os.getuid()}-*"))
        with tempfile.TemporaryDirectory(prefix="cross-ai-test-") as fixture:
            root = Path(fixture)
            fake = root / "opencode"
            context = root / "context.md"
            context.write_text("review me", encoding="utf-8")
            _write_fake_opencode(fake, hang=False)
            env = os.environ.copy()
            env["OPENCODE_BIN"] = str(fake)
            result = subprocess.run(
                [
                    str(LAUNCHER),
                    "--reviewer",
                    "deepseek-v4-pro",
                    "--no-shared-server",
                    "--repo-root",
                    str(root),
                    "--output-dir",
                    "reports",
                    str(context),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            reports = list((root / "reports").glob("*/deepseek-v4-pro.md"))
            self.assertEqual(len(reports), 1)
        self.assertEqual(
            set(Path("/tmp").glob(f"cross-ai-{os.getuid()}-*")),
            before,
        )

    def test_signals_clean_temp_and_process_group(self) -> None:
        for signum, returncode in (
            (signal.SIGHUP, 129),
            (signal.SIGINT, 130),
            (signal.SIGQUIT, 131),
            (signal.SIGTERM, 143),
        ):
            with self.subTest(signum=signum):
                self._assert_signal_cleanup(signum, returncode)

    def _assert_signal_cleanup(self, signum: int, returncode: int) -> None:
        before = set(Path("/tmp").glob(f"cross-ai-{os.getuid()}-*"))
        with tempfile.TemporaryDirectory(prefix="cross-ai-test-") as fixture:
            root = Path(fixture)
            fake = root / "opencode"
            context = root / "context.md"
            marker = root / "temp-path"
            child_pid_path = root / "child-pid"
            context.write_text("review me", encoding="utf-8")
            _write_fake_opencode(fake, hang=True)
            env = os.environ.copy()
            env.update(
                OPENCODE_BIN=str(fake),
                CROSS_AI_TEST_MARKER=str(marker),
                CROSS_AI_TEST_CHILD_PID=str(child_pid_path),
            )
            process = subprocess.Popen(
                [
                    str(LAUNCHER),
                    "--reviewer",
                    "deepseek-v4-pro",
                    "--no-shared-server",
                    "--repo-root",
                    str(root),
                    "--output-dir",
                    "reports",
                    str(context),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            try:
                deadline = time.monotonic() + 10
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(marker.exists(), "fake runtime did not start")
                child_pid = int(child_pid_path.read_text(encoding="utf-8"))
                temporary_dir = Path(marker.read_text(encoding="utf-8"))
                process.send_signal(signum)
                _, stderr = process.communicate(timeout=15)
                self.assertEqual(process.returncode, returncode, stderr)
                self.assertFalse(temporary_dir.exists())
                deadline = time.monotonic() + 3
                while _process_is_live(child_pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(_process_is_live(child_pid))
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
        self.assertEqual(
            set(Path("/tmp").glob(f"cross-ai-{os.getuid()}-*")),
            before,
        )

    def test_partial_server_startup_kills_process_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cross-ai-test-") as fixture:
            root = Path(fixture)
            fake = root / "opencode"
            marker = root / "temp-path"
            child_pid_path = root / "child-pid"
            temporary_dir = cross_ai._create_temporary_directory()
            _write_fake_opencode(fake, hang=True)
            environment = {
                "CROSS_AI_TEST_MARKER": str(marker),
                "CROSS_AI_TEST_CHILD_PID": str(child_pid_path),
            }
            try:
                with mock.patch.dict(os.environ, environment):
                    with self.assertRaises(TimeoutError):
                        asyncio.run(
                            cross_ai._start_opencode_server(
                                opencode_bin=str(fake),
                                repo_root=root,
                                output_dir=root,
                                config_path=root / "config.json",
                                temporary_dir=temporary_dir,
                                timeout_seconds=1,
                            )
                        )
                child_pid = int(child_pid_path.read_text(encoding="utf-8"))
                deadline = time.monotonic() + 3
                while _process_is_live(child_pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(_process_is_live(child_pid))
            finally:
                cross_ai._remove_temporary_directory(temporary_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
