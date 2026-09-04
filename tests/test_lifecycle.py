import asyncio
import contextlib
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
SCRIPT = REPOSITORY_ROOT / "cross_ai.py"


def _write_fake_opencode(path: Path, *, hang: bool) -> None:
    behavior = (
        "runtime_pid = os.environ.get('CROSS_AI_TEST_RUNTIME_PID')\n"
        "if runtime_pid:\n"
        "    Path(runtime_pid).write_text(str(os.getpid()), encoding='utf-8')\n"
        "child = subprocess.Popen(['sleep', '60'])\n"
        "Path(os.environ['CROSS_AI_TEST_CHILD_PID']).write_text(str(child.pid), encoding='utf-8')\n"
        "marker = os.environ.get('CROSS_AI_TEST_MARKER')\n"
        "if marker:\n"
        "    Path(marker).write_text(os.environ['TMPDIR'], encoding='utf-8')\n"
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
    with contextlib.suppress(FileNotFoundError):
        return "State:\tZ" not in status_path.read_text(encoding="utf-8")
    return False


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def _process_cmdline(pid: int) -> bytes | None:
    if not _process_is_live(pid):
        return None
    with contextlib.suppress(FileNotFoundError, OSError):
        return Path(f"/proc/{pid}/cmdline").read_bytes()
    return None


def _fake_group_identity(fake: Path, runtime_pid: int, child_pid: int | None) -> bool:
    runtime_cmdline = _process_cmdline(runtime_pid)
    if runtime_cmdline is not None:
        with contextlib.suppress(FileNotFoundError, OSError):
            if os.getpgid(runtime_pid) == runtime_pid and str(fake).encode() in runtime_cmdline:
                return True
    if child_pid is None:
        return False
    child_cmdline = _process_cmdline(child_pid)
    if child_cmdline is None:
        return False
    with contextlib.suppress(FileNotFoundError, OSError):
        return os.getpgid(child_pid) == runtime_pid and b"sleep" in child_cmdline and b"60" in child_cmdline
    return False


def _terminate_fake_process_group(fake: Path, runtime_pid_path: Path, child_pid_path: Path) -> None:
    runtime_pid = _read_pid(runtime_pid_path)
    child_pid = _read_pid(child_pid_path)
    if runtime_pid is None or runtime_pid <= 1 or not _fake_group_identity(fake, runtime_pid, child_pid):
        return

    with contextlib.suppress(ProcessLookupError):
        os.killpg(runtime_pid, signal.SIGTERM)
    deadline = time.monotonic() + 3
    while (
        _process_is_live(runtime_pid) or (child_pid is not None and _process_is_live(child_pid))
    ) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_is_live(runtime_pid) or (child_pid is not None and _process_is_live(child_pid)):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(runtime_pid, signal.SIGKILL)


def _drain_subprocess(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.kill()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)


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
                asyncio.run(cross_ai._monitor_temporary_directory(path, limit_bytes=1))
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
                    str(SCRIPT),
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
            runtime_pid_path = root / "runtime-pid"
            context.write_text("review me", encoding="utf-8")
            _write_fake_opencode(fake, hang=True)
            env = os.environ.copy()
            env.update(
                OPENCODE_BIN=str(fake),
                CROSS_AI_TEST_MARKER=str(marker),
                CROSS_AI_TEST_CHILD_PID=str(child_pid_path),
                CROSS_AI_TEST_RUNTIME_PID=str(runtime_pid_path),
            )
            process = subprocess.Popen(
                [
                    str(SCRIPT),
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
                while (
                    not marker.exists() or not child_pid_path.exists() or not runtime_pid_path.exists()
                ) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(marker.exists(), "fake runtime did not start")
                self.assertTrue(child_pid_path.exists(), "fake runtime child was not ready")
                self.assertTrue(runtime_pid_path.exists(), "fake runtime PID was not ready")
                child_pid = int(child_pid_path.read_text(encoding="utf-8"))
                runtime_pid = int(runtime_pid_path.read_text(encoding="utf-8"))
                temporary_dir = Path(marker.read_text(encoding="utf-8"))
                process.send_signal(signum)
                _, stderr = process.communicate(timeout=15)
                self.assertEqual(process.returncode, returncode, stderr)
                self.assertFalse(temporary_dir.exists())
                deadline = time.monotonic() + 3
                while _process_is_live(child_pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(_process_is_live(child_pid))
                self.assertFalse(_process_is_live(runtime_pid))
            finally:
                _terminate_fake_process_group(fake, runtime_pid_path, child_pid_path)
                _drain_subprocess(process)
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
            runtime_pid_path = root / "runtime-pid"
            temporary_dir = cross_ai._create_temporary_directory()
            _write_fake_opencode(fake, hang=True)
            environment = {
                "CROSS_AI_TEST_MARKER": str(marker),
                "CROSS_AI_TEST_CHILD_PID": str(child_pid_path),
                "CROSS_AI_TEST_RUNTIME_PID": str(runtime_pid_path),
            }
            try:
                with mock.patch.dict(os.environ, environment), self.assertRaises(TimeoutError):
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
                runtime_pid = int(runtime_pid_path.read_text(encoding="utf-8"))
                deadline = time.monotonic() + 3
                while _process_is_live(child_pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(_process_is_live(child_pid))
                self.assertFalse(_process_is_live(runtime_pid))
            finally:
                _terminate_fake_process_group(fake, runtime_pid_path, child_pid_path)
                cross_ai._remove_temporary_directory(temporary_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
