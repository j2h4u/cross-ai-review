import argparse
import asyncio
import json
from pathlib import Path
from unittest import mock

import pytest

import cross_ai


def request(root: Path, runtime: str = "codex") -> cross_ai.ReviewRequest:
    return cross_ai.ReviewRequest(
        runtime_bin=f"/bin/{runtime}",
        review_model=cross_ai.ReviewerSpec(runtime, runtime, "test-model", 1),
        repo_root=root,
        context_files=(root / "context.md",),
        prompt="review",
        output_dir=root,
        temporary_dir=root,
        timeout_seconds=1,
        attach_url=None,
        min_output_chars=5,
        require_review_markers=False,
    )


def opencode_output(text: str) -> bytes:
    return "\n".join(
        (
            json.dumps({"type": "text", "part": {"messageID": "m1", "text": text}}),
            json.dumps({"type": "step_finish", "part": {"messageID": "m1", "reason": "stop"}}),
        )
    ).encode()


def codex_output(text: str) -> bytes:
    return json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": text}}).encode()


def result(root: Path, slug: str, *, valid: bool = True) -> cross_ai.ReviewResult:
    return cross_ai.ReviewResult(
        cross_ai.ReviewerSpec(slug, "codex", "test-model", 1),
        0 if valid else 1,
        root / f"{slug}.md",
        valid,
        1,
        None if valid else "invalid",
    )


def test_output_parsers_and_validation_cover_all_runtimes(tmp_path: Path) -> None:
    assert (
        cross_ai._extract_final_review_text(opencode_output("Finding: OpenCode answer")) == "Finding: OpenCode answer"
    )
    assert (
        cross_ai._extract_claude_review_text(json.dumps({"result": "Finding: Claude answer"}).encode())
        == "Finding: Claude answer"
    )
    assert cross_ai._extract_codex_review_text(codex_output("Finding: Codex answer")) == "Finding: Codex answer"
    for runtime, output in (
        ("opencode", opencode_output("Finding: valid answer")),
        ("claude", json.dumps({"result": "Finding: valid answer"}).encode()),
        ("codex", codex_output("Finding: valid answer")),
    ):
        assert cross_ai._validate_runtime_output(request(tmp_path, runtime), output) == ("Finding: valid answer", None)
    with pytest.raises(ValueError, match="not valid JSON"):
        cross_ai._extract_claude_review_text(b"not-json")
    with pytest.raises(ValueError, match="completed agent message"):
        cross_ai._extract_codex_review_text(b"{}")


def test_opencode_output_is_strict_and_codex_tolerates_unrelated_lines() -> None:
    malformed = opencode_output("Finding: valid") + b"\nnot-json\n"
    with pytest.raises(ValueError, match="line 3 is not valid JSON"):
        cross_ai._extract_final_review_text(malformed)
    tolerant = b"not-json\n" + codex_output("Finding: valid Codex") + b"\n{}\n"
    assert cross_ai._extract_codex_review_text(tolerant) == "Finding: valid Codex"


def test_bash_permission_failure_accepts_only_permission_errors() -> None:
    events = (
        {},
        {"part": "not-a-table"},
        {"part": {"type": "text", "tool": "bash"}},
        {"part": {"type": "tool", "tool": "bash", "state": "not-a-table"}},
        {"part": {"type": "tool", "tool": "bash", "state": {"status": "ok"}}},
        {"part": {"type": "tool", "tool": "bash", "state": {"status": "error", "error": "failed"}}},
        {"part": {"type": "tool", "tool": "bash", "state": {"status": "error", "error": 3}}},
    )
    for event in events:
        assert cross_ai._bash_permission_failure(event) is None
    state = {"status": "error", "error": "Permission denied by policy", "input": {"command": "make"}}
    assert cross_ai._bash_permission_failure({"part": {"type": "tool", "tool": "bash", "state": state}}) == (
        state,
        "Permission denied by policy",
    )


def test_shared_opencode_invalid_output_retries_directly(tmp_path: Path) -> None:
    calls: list[str | None] = []

    async def attempt(
        current: cross_ai.ReviewRequest, *, attach_url: str | None, opencode_config_path: Path
    ) -> cross_ai.ReviewAttempt:
        del current, opencode_config_path
        calls.append(attach_url)
        shared = attach_url is not None
        return cross_ai.ReviewAttempt(
            command=["fake", "shared" if shared else "direct"],
            stdout=b"",
            timed_out=False,
            returncode=0,
            final_text=None if shared else "Finding: direct retry",
            validation_error="shared output invalid" if shared else None,
            permission_denials=(),
            mode="shared" if shared else "direct",
        )

    current = cross_ai.ReviewRequest(
        **{**request(tmp_path, "opencode").__dict__, "attach_url": "http://127.0.0.1:1234"}
    )
    with mock.patch.object(cross_ai, "_review_attempt", side_effect=attempt):
        attempts = asyncio.run(cross_ai._collect_review_attempts(current, opencode_config_path=tmp_path / "config"))
    assert calls == ["http://127.0.0.1:1234", None]
    assert [item.mode for item in attempts] == ["shared", "direct"]
    assert attempts[-1].final_text == "Finding: direct retry"


def test_review_report_keeps_retry_diagnostics(tmp_path: Path) -> None:
    current = request(tmp_path, "opencode")
    attempts = [
        cross_ai.ReviewAttempt(["fake", "shared"], b"raw", False, 0, None, "invalid shared", (), "shared"),
        cross_ai.ReviewAttempt(["fake", "direct"], b"raw", False, 0, "Finding: final", None, (), "direct"),
    ]
    with mock.patch.object(cross_ai, "_collect_review_attempts", return_value=attempts):
        result = asyncio.run(cross_ai._run_review(current, opencode_config_path=tmp_path / "config"))
    report = (tmp_path / "opencode.md").read_text(encoding="utf-8")
    assert result.valid_output
    assert "A direct retry was performed" in report
    assert "Finding: final" in report


def test_dispatch_resolves_before_async_run_or_creating_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_text = (
        """
default_profile = "standard"
default_timeout_seconds = 77
[runtimes.opencode]
binary = "/missing/opencode"
[profiles.standard]
reviewers = ["oc-review"]
[reviewers.oc-review]
runtime = "opencode"
model = "provider/model"
""".strip()
        + "\n"
    )
    config_path = tmp_path / "cross-ai/config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(config_text, encoding="utf-8")
    context = tmp_path / "context.md"
    context.write_text("context", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    called = False

    async def unexpected_async(*args: object) -> int:
        del args
        nonlocal called
        called = True
        return 0

    with (
        mock.patch.object(cross_ai, "_main_async", side_effect=unexpected_async),
        pytest.raises(cross_ai.ConfigurationError, match="not an executable file"),
    ):
        cross_ai._dispatch(
            ["--reviewer", "oc-review", "--repo-root", str(tmp_path), "--output-dir", "reports", str(context)]
        )
    assert not called
    assert not (tmp_path / "reports").exists()


def test_permission_policy_and_summary_preserve_contract(tmp_path: Path) -> None:
    permissions = cross_ai._opencode_permissions(tmp_path)
    assert permissions["read"]["*"] == "allow"
    assert permissions["bash"]["*"] == "deny"
    for tool in ("edit", "write", "task", "question", "webfetch", "websearch", "skill"):
        assert permissions[tool] == "deny"
    denial = cross_ai.PermissionDenial("make", "denied", "rule")
    result = cross_ai.ReviewResult(
        cross_ai.ReviewerSpec("reviewer", "codex", "model", 1),
        1,
        tmp_path / "reviewer.md",
        False,
        1,
        "bad",
        (denial,),
    )
    summary = cross_ai.RunSummary(tmp_path, "review", tmp_path, None, tmp_path / "opencode.json", (), [result], 1)
    path = tmp_path / "summary.md"
    cross_ai._write_run_summary(path, summary)
    text = path.read_text(encoding="utf-8")
    assert "invalid output (bad)" in text
    assert "denied command" in text


def test_review_attempt_success_timeout_and_exception(tmp_path: Path) -> None:
    current = request(tmp_path, "opencode")

    class Process:
        returncode = 0

        def __init__(self, *outputs: object) -> None:
            self.outputs = list(outputs)

        async def communicate(self) -> tuple[bytes, None]:
            output = self.outputs.pop(0)
            if isinstance(output, BaseException):
                raise output
            return output, None

    with mock.patch.object(asyncio, "create_subprocess_exec", return_value=Process(opencode_output("Finding: answer"))):
        attempt = asyncio.run(
            cross_ai._review_attempt(current, attach_url=None, opencode_config_path=tmp_path / "config")
        )
    assert attempt.returncode == 0
    assert attempt.final_text == "Finding: answer"
    assert not attempt.timed_out

    timed = Process(TimeoutError(), opencode_output("Finding: recovered"))
    with (
        mock.patch.object(asyncio, "create_subprocess_exec", return_value=timed),
        mock.patch.object(cross_ai, "_terminate_process", new_callable=mock.AsyncMock) as terminate,
    ):
        attempt = asyncio.run(
            cross_ai._review_attempt(current, attach_url=None, opencode_config_path=tmp_path / "config")
        )
    assert attempt.timed_out
    assert attempt.returncode == 124
    assert attempt.final_text is None
    terminate.assert_awaited_once_with(timed)

    failed = Process(RuntimeError("broken process"))
    with (
        mock.patch.object(asyncio, "create_subprocess_exec", return_value=failed),
        mock.patch.object(cross_ai, "_terminate_process", new_callable=mock.AsyncMock) as terminate,
        pytest.raises(RuntimeError, match="broken process"),
    ):
        asyncio.run(cross_ai._review_attempt(current, attach_url=None, opencode_config_path=tmp_path / "config"))
    terminate.assert_awaited_once_with(failed)


def test_execute_reviews_orders_and_reports_invalid_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    models = (
        cross_ai.ReviewerSpec("first", "codex", "test-model", 1),
        cross_ai.ReviewerSpec("second", "codex", "test-model", 1),
    )
    run = cross_ai.PreparedRun(
        {}, tmp_path, "review", tmp_path, tmp_path, tmp_path / "config", (), "prompt", models, False
    )

    async def completed(value: cross_ai.ReviewResult) -> cross_ai.ReviewResult:
        await asyncio.sleep(0)
        return value

    with mock.patch.object(
        cross_ai,
        "_review_tasks",
        return_value=[completed(result(tmp_path, "second", valid=False)), completed(result(tmp_path, "first"))],
    ):
        values = asyncio.run(cross_ai._execute_reviews(run, None))
    assert [value.model.slug for value in values] == ["first", "second"]
    output = capsys.readouterr().out
    assert "status=valid" in output
    assert "status=invalid" in output


def test_main_async_success_cleans_temporary_directory(tmp_path: Path) -> None:
    args = argparse.Namespace(
        repo_root=str(tmp_path),
        context_file=[str(tmp_path / "context.md")],
        output_dir="reports",
        mode="review",
        goal=None,
        no_shared_server=True,
        premium=False,
    )
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    prepared = cross_ai.PreparedRun(
        {}, tmp_path, "review", tmp_path, temporary, tmp_path / "config", (), "prompt", (), False
    )

    async def execute(run: cross_ai.PreparedRun, *, started_at: float) -> int:
        assert run is prepared
        assert started_at > 0
        return 0

    with (
        mock.patch.object(cross_ai, "_create_temporary_directory", return_value=temporary),
        mock.patch.object(cross_ai, "_prepare_run", return_value=prepared) as prepare,
        mock.patch.object(cross_ai, "_print_startup"),
        mock.patch.object(cross_ai, "_execute_with_temporary_limit", side_effect=execute),
        mock.patch.object(cross_ai, "_remove_temporary_directory") as remove,
    ):
        value = asyncio.run(cross_ai._main_async(args, (), {}, False))
    assert value == 0
    prepare.assert_called_once_with(args, (), {}, False, temporary_dir=temporary)
    remove.assert_called_once_with(temporary)
