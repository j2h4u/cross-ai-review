import asyncio
import contextlib
import os
import signal
import stat
import subprocess
import time
from pathlib import Path
from unittest import mock

import pytest

import cross_ai

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "cross_ai.py"


def write_fake_opencode(path: Path, *, hang: bool) -> None:
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


def config_for(fake: Path, xdg: Path, *, include_unavailable_runtimes: bool = False) -> None:
    (xdg / "cross-ai").mkdir(parents=True)
    (xdg / "cross-ai/config.toml").write_text(
        f"""
default_profile = "standard"
default_timeout_seconds = 5

[runtimes.opencode]
binary = "{fake}"
"""
        + ("\n[runtimes.claude]\n[runtimes.codex]\n" if include_unavailable_runtimes else "")
        + """

[profiles.standard]
reviewers = ["oc-review"]

[reviewers.oc-review]
runtime = "opencode"
model = "provider/model"
reasoning = "max"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def process_is_live(pid: int) -> bool:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return False
    with contextlib.suppress(FileNotFoundError):
        return "State:\tZ" not in status_path.read_text(encoding="utf-8")
    return False


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def terminate_fake_group(fake: Path, runtime_path: Path, child_path: Path) -> None:
    runtime_pid = read_pid(runtime_path)
    child_pid = read_pid(child_path)
    if runtime_pid is None or runtime_pid <= 1:
        return
    with contextlib.suppress(FileNotFoundError, OSError, ProcessLookupError):
        if (
            os.getpgid(runtime_pid) == runtime_pid
            and str(fake).encode() in Path(f"/proc/{runtime_pid}/cmdline").read_bytes()
        ):
            os.killpg(runtime_pid, signal.SIGKILL)
    if child_pid is not None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)


def drain(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.communicate(timeout=5)


def run_direct(root: Path, xdg: Path, context: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(SCRIPT),
            "--reviewer",
            "oc-review",
            "--no-shared-server",
            "--repo-root",
            str(root),
            "--output-dir",
            "reports",
            str(context),
        ],
        cwd=root,
        env={**os.environ, "XDG_CONFIG_HOME": str(xdg)},
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def test_temp_environment_limit_and_cleanup() -> None:
    path = cross_ai._create_temporary_directory()
    try:
        assert path.parent == Path("/tmp")
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
        env = cross_ai._opencode_environment(Path("/tmp/config"), path)
        assert env["TMPDIR"] == str(path)
        assert env["BUN_TMPDIR"] == str(path)
        (path / "oversized.so").write_bytes(b"xx")
        with pytest.raises(cross_ai.TemporaryDirectoryLimitExceeded):
            asyncio.run(cross_ai._monitor_temporary_directory(path, limit_bytes=1))
    finally:
        cross_ai._remove_temporary_directory(path)
    assert not path.exists()


def test_temp_monitor_removes_only_stale_bun_artifacts() -> None:
    path = cross_ai._create_temporary_directory()
    try:
        stale_bun = path / ".9adf5afbebf9ef97-00000000.so"
        fresh_bun = path / ".9adf5afbebf9ef98-00000000.so"
        unrelated = path / "oversized.so"
        stale_bun.write_bytes(b"xx")
        fresh_bun.write_bytes(b"xx")
        unrelated.write_bytes(b"xx")
        old = time.time() - cross_ai.OPENCODE_BUN_ARTIFACT_GRACE_SECONDS - 1
        os.utime(stale_bun, (old, old))
        os.utime(unrelated, (old, old))

        cross_ai._remove_stale_bun_artifacts(
            path,
            older_than=cross_ai.OPENCODE_BUN_ARTIFACT_GRACE_SECONDS,
        )

        assert not stale_bun.exists()
        assert fresh_bun.exists()
        assert unrelated.exists()
    finally:
        cross_ai._remove_temporary_directory(path)


def test_success_preserves_report_and_removes_run_temp(tmp_path: Path) -> None:
    before = set(Path("/tmp").glob(f"cross-ai-{os.getuid()}-*"))
    fake = tmp_path / "opencode"
    context = tmp_path / "context.md"
    context.write_text("review me", encoding="utf-8")
    write_fake_opencode(fake, hang=False)
    xdg = tmp_path / "xdg"
    config_for(fake, xdg, include_unavailable_runtimes=True)
    result = run_direct(tmp_path, xdg, context)
    assert result.returncode == 0, result.stderr
    reports = list((tmp_path / "reports").glob("*/oc-review.md"))
    assert len(reports) == 1
    assert "Finding: lifecycle test output" in reports[0].read_text(encoding="utf-8")
    assert set(Path("/tmp").glob(f"cross-ai-{os.getuid()}-*")) == before


@pytest.mark.parametrize(
    ("signum", "returncode"),
    ((signal.SIGHUP, 129), (signal.SIGINT, 130), (signal.SIGQUIT, 131), (signal.SIGTERM, 143)),
)
def test_signals_clean_temp_and_process_group(tmp_path: Path, signum: signal.Signals, returncode: int) -> None:
    fake = tmp_path / "opencode"
    context = tmp_path / "context.md"
    marker = tmp_path / "temp-path"
    child_path = tmp_path / "child-pid"
    runtime_path = tmp_path / "runtime-pid"
    context.write_text("review me", encoding="utf-8")
    write_fake_opencode(fake, hang=True)
    xdg = tmp_path / "xdg"
    config_for(fake, xdg)
    process = subprocess.Popen(
        [str(SCRIPT), "--reviewer", "oc-review", "--no-shared-server", "--repo-root", str(tmp_path), str(context)],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "XDG_CONFIG_HOME": str(xdg),
            "CROSS_AI_TEST_MARKER": str(marker),
            "CROSS_AI_TEST_CHILD_PID": str(child_path),
            "CROSS_AI_TEST_RUNTIME_PID": str(runtime_path),
        },
    )
    try:
        deadline = time.monotonic() + 10
        while (
            not marker.exists() or not child_path.exists() or not runtime_path.exists()
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), "fake runtime did not start"
        assert child_path.exists()
        assert runtime_path.exists()
        temporary = Path(marker.read_text(encoding="utf-8"))
        process.send_signal(signum)
        _, stderr = process.communicate(timeout=15)
        assert process.returncode == returncode, stderr
        assert not temporary.exists()
        child_pid = read_pid(child_path)
        runtime_pid = read_pid(runtime_path)
        deadline = time.monotonic() + 3
        while (
            (child_pid and process_is_live(child_pid)) or (runtime_pid and process_is_live(runtime_pid))
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_pid is None or not process_is_live(child_pid)
        assert runtime_pid is None or not process_is_live(runtime_pid)
    finally:
        terminate_fake_group(fake, runtime_path, child_path)
        drain(process)


def test_partial_server_startup_kills_process_group(tmp_path: Path) -> None:
    fake = tmp_path / "opencode"
    marker = tmp_path / "temp-path"
    child_path = tmp_path / "child-pid"
    runtime_path = tmp_path / "runtime-pid"
    temporary = cross_ai._create_temporary_directory()
    write_fake_opencode(fake, hang=True)
    try:
        with (
            mock.patch.dict(
                os.environ,
                {
                    "CROSS_AI_TEST_MARKER": str(marker),
                    "CROSS_AI_TEST_CHILD_PID": str(child_path),
                    "CROSS_AI_TEST_RUNTIME_PID": str(runtime_path),
                },
            ),
            pytest.raises(TimeoutError),
        ):
            asyncio.run(
                cross_ai._start_opencode_server(
                    opencode_bin=str(fake),
                    repo_root=tmp_path,
                    output_dir=tmp_path,
                    opencode_config_path=tmp_path / "opencode.json",
                    temporary_dir=temporary,
                    timeout_seconds=1,
                )
            )
        child_pid = read_pid(child_path)
        runtime_pid = read_pid(runtime_path)
        assert child_pid is not None and runtime_pid is not None
        deadline = time.monotonic() + 3
        while (process_is_live(child_pid) or process_is_live(runtime_pid)) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not process_is_live(child_pid)
        assert not process_is_live(runtime_pid)
    finally:
        terminate_fake_group(fake, runtime_path, child_path)
        cross_ai._remove_temporary_directory(temporary)
