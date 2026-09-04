import argparse
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cross_ai


def _args(**overrides: object) -> argparse.Namespace:
    values = {"reviewer": None, "premium": False, "all": False}
    values.update(overrides)
    return argparse.Namespace(**values)


def _request(root: Path, runtime: str = "codex") -> cross_ai.ReviewRequest:
    return cross_ai.ReviewRequest(
        runtime_bin=runtime,
        review_model=cross_ai.ReviewerSpec(slug=runtime, runtime=runtime, model="test"),
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


def _result(root: Path, slug: str = "codex") -> cross_ai.ReviewResult:
    return cross_ai.ReviewResult(
        model=cross_ai.ReviewerSpec(slug=slug, runtime=slug, model="test"),
        returncode=0,
        output_path=root / f"{slug}.md",
        valid_output=True,
        elapsed_seconds=1,
    )


def _opencode_output(text: str) -> bytes:
    events = [
        {"type": "text", "part": {"messageID": "m1", "text": text}},
        {"type": "step_finish", "part": {"messageID": "m1", "reason": "stop"}},
    ]
    return "\n".join(json.dumps(event) for event in events).encode()


def _codex_output(text: str) -> bytes:
    event = {"type": "item.completed", "item": {"type": "agent_message", "text": text}}
    return json.dumps(event).encode()


class CrossAiQualityTests(unittest.TestCase):
    def test_reviewer_selection_paths(self) -> None:
        self.assertEqual([item.slug for item in cross_ai._selected_reviewers(_args())], ["deepseek-v4-pro", "glm-5-3"])
        self.assertEqual([item.slug for item in cross_ai._selected_reviewers(_args(premium=True))], ["claude-opus-5"])
        self.assertEqual(len(cross_ai._selected_reviewers(_args(all=True))), 3)
        self.assertEqual(
            [item.slug for item in cross_ai._selected_reviewers(_args(reviewer=["deepseek-v4-pro"]))],
            ["deepseek-v4-pro"],
        )
        disabled = cross_ai.ReviewerSpec("disabled", "codex", "test", enabled=False)
        with (
            mock.patch.dict(cross_ai.REVIEWER_REGISTRY, {"disabled": disabled}, clear=True),
            self.assertRaises(SystemExit),
        ):
            cross_ai._selected_reviewers(_args(reviewer=["disabled"]))

    def test_all_profile_follows_registry_order_and_identity(self) -> None:
        profile_slugs = cross_ai.REVIEW_PROFILES["all"]
        self.assertEqual(profile_slugs, tuple(cross_ai.REVIEWER_REGISTRY))
        profile_reviewers = cross_ai._reviewers_for_profile("all")
        self.assertEqual(
            [id(reviewer) for reviewer in profile_reviewers],
            [id(cross_ai.REVIEWER_REGISTRY[slug]) for slug in profile_slugs],
        )

    def test_executable_discovery_uses_explicit_path_then_path_then_home(self) -> None:
        runtimes = (
            ("opencode", cross_ai._find_opencode, "OPENCODE_BIN", ".opencode/bin/opencode"),
            ("claude", cross_ai._find_claude, "CLAUDE_BIN", ".local/bin/claude"),
            ("codex", cross_ai._find_codex, "CODEX_BIN", ".local/bin/codex"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for runtime, finder, variable, home_suffix in runtimes:
                with self.subTest(runtime=runtime, source="explicit"):
                    explicit = root / f"{runtime}-explicit"
                    explicit.touch(mode=0o755)
                    path_candidate = root / f"{runtime}-path"
                    path_candidate.touch(mode=0o755)
                    with (
                        mock.patch.dict(os.environ, {variable: str(explicit)}),
                        mock.patch.object(cross_ai.shutil, "which", return_value=str(path_candidate)),
                        mock.patch.object(cross_ai.Path, "home", return_value=root),
                    ):
                        self.assertEqual(finder(), str(explicit))
                with self.subTest(runtime=runtime, source="PATH"):
                    path_candidate = root / f"{runtime}-path-only"
                    path_candidate.touch(mode=0o755)
                    with (
                        mock.patch.dict(os.environ, {}, clear=False),
                        mock.patch.object(cross_ai.shutil, "which", return_value=str(path_candidate)),
                        mock.patch.object(cross_ai.Path, "home", return_value=root),
                    ):
                        os.environ.pop(variable, None)
                        self.assertEqual(finder(), str(path_candidate))
                with self.subTest(runtime=runtime, source="home"):
                    home_candidate = root / home_suffix
                    home_candidate.parent.mkdir(parents=True, exist_ok=True)
                    home_candidate.touch(mode=0o755)
                    with (
                        mock.patch.dict(os.environ, {}, clear=False),
                        mock.patch.object(cross_ai.shutil, "which", return_value=None),
                        mock.patch.object(cross_ai.Path, "home", return_value=root),
                    ):
                        os.environ.pop(variable, None)
                        self.assertEqual(finder(), str(home_candidate))

    def test_executable_discovery_rejects_unusable_candidates_with_runtime_errors(self) -> None:
        runtimes = (
            (
                "opencode",
                cross_ai._find_opencode,
                "OPENCODE_BIN",
                "opencode binary not found. Set OPENCODE_BIN or install it at ~/.opencode/bin/opencode.",
            ),
            (
                "claude",
                cross_ai._find_claude,
                "CLAUDE_BIN",
                "claude binary not found. Set CLAUDE_BIN or install Claude Code.",
            ),
            (
                "codex",
                cross_ai._find_codex,
                "CODEX_BIN",
                "codex binary not found. Set CODEX_BIN or install Codex CLI.",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for runtime, finder, variable, expected_error in runtimes:
                with self.subTest(runtime=runtime):
                    non_executable = root / f"{runtime}-not-executable"
                    non_executable.touch(mode=0o644)
                    with (
                        mock.patch.dict(os.environ, {variable: str(non_executable)}),
                        mock.patch.object(cross_ai.shutil, "which", return_value=str(root / "missing")),
                        mock.patch.object(cross_ai.Path, "home", return_value=root),
                        self.assertRaises(SystemExit) as raised,
                    ):
                        finder()
                    self.assertEqual(
                        str(raised.exception),
                        expected_error,
                    )

    def test_prepare_run_resolves_reviewers_and_binaries_before_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = root / "context.md"
            context.write_text("context", encoding="utf-8")
            args = argparse.Namespace(
                repo_root=str(root),
                context_file=[str(context)],
                output_dir="reports",
                mode="review",
                goal=None,
                reviewer=["deepseek-v4-pro"],
                premium=False,
                all=False,
                no_shared_server=True,
            )
            for failure, patch_target in (
                ("reviewer failure", "_selected_reviewers"),
                ("binary failure", "_runtime_bins"),
            ):
                with self.subTest(failure=failure):
                    with (
                        mock.patch.object(cross_ai, patch_target, side_effect=SystemExit(failure)),
                        mock.patch.object(cross_ai, "_create_run_directory") as create_run_directory,
                        mock.patch.object(cross_ai, "_write_opencode_config") as write_config,
                        mock.patch.object(cross_ai, "_copy_context_files") as copy_context,
                        self.assertRaisesRegex(SystemExit, failure),
                    ):
                        cross_ai._prepare_run(args, temporary_dir=root / "temporary")
                    create_run_directory.assert_not_called()
                    write_config.assert_not_called()
                    copy_context.assert_not_called()
                    self.assertFalse((root / "reports").exists())

    def test_review_report_preserves_attempts_retry_reason_validation_and_final_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root, "opencode")
            attempts = [
                cross_ai.ReviewAttempt(
                    command=["opencode", "run", "--attach", "http://127.0.0.1:1234"],
                    stdout=b"shared raw output",
                    timed_out=False,
                    returncode=0,
                    final_text=None,
                    validation_error="shared response had no completed step",
                    permission_denials=(),
                    mode="shared",
                ),
                cross_ai.ReviewAttempt(
                    command=["opencode", "run", "--dir", str(root)],
                    stdout=b"direct raw output",
                    timed_out=False,
                    returncode=0,
                    final_text="Finding: direct final text",
                    validation_error=None,
                    permission_denials=(),
                    mode="direct",
                ),
            ]
            with mock.patch.object(cross_ai, "_collect_review_attempts", return_value=attempts):
                result = asyncio.run(cross_ai._run_review(request, config_path=root / "config"))
            report = request.output_dir / "opencode.md"
            text = report.read_text(encoding="utf-8")

        self.assertTrue(result.valid_output)
        self.assertFalse(hasattr(result, "command"))
        self.assertFalse(hasattr(result, "attempts"))
        self.assertIn("shared: opencode run --attach http://127.0.0.1:1234", text)
        self.assertIn(f"direct: opencode run --dir {root}", text)
        self.assertIn(
            "A direct retry was performed because the attached shared-server output failed validation: "
            "shared response had no completed step",
            text,
        )
        self.assertIn(f"Command:\n\n```bash\nopencode run --dir {root}\n```", text)
        self.assertIn("Finding: direct final text", text)

    def test_json_event_parsing_and_permission_detection(self) -> None:
        events = [
            {"type": "text", "part": {"messageID": "m1", "text": "Finding: okay"}},
            {"type": "step_finish", "part": {"messageID": "m1", "reason": "stop"}},
            {
                "part": {
                    "type": "tool",
                    "tool": "bash",
                    "state": {"status": "error", "error": "Permission denied", "input": {"command": "make"}},
                }
            },
        ]
        stdout = "\n".join(json.dumps(event) for event in events).encode()
        self.assertEqual(cross_ai._extract_final_review_text(stdout), "Finding: okay")
        self.assertEqual(cross_ai._permission_denials(cross_ai._decode_review_output(stdout))[0].command, "make")
        self.assertIsNone(cross_ai._bash_permission_failure({}))
        with self.assertRaises(ValueError):
            cross_ai._decode_review_output(b"not-json")
        with self.assertRaises(ValueError):
            cross_ai._decode_review_output(b"[]")

    def test_codex_and_runtime_output_validation(self) -> None:
        lines = [
            b"not-json",
            json.dumps({"type": "other"}).encode(),
            json.dumps({"type": "item.completed", "item": {"type": "other"}}).encode(),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "first answer"}}).encode(),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "final answer"}}).encode(),
        ]
        stdout = b"\n".join(lines)
        self.assertEqual(cross_ai._extract_codex_review_text(stdout), "final answer")
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory))
            self.assertEqual(cross_ai._validate_runtime_output(request, stdout), ("final answer", None))
            short_request = cross_ai.ReviewRequest(**{**request.__dict__, "min_output_chars": 100})
            self.assertIn("too short", cross_ai._validate_runtime_output(short_request, stdout)[1] or "")
            unknown = cross_ai.ReviewRequest(
                **{**request.__dict__, "review_model": cross_ai.ReviewerSpec("x", "x", "x")}
            )
            self.assertIn("unsupported", cross_ai._validate_runtime_output(unknown, stdout)[1] or "")
        with self.assertRaises(ValueError):
            cross_ai._extract_codex_review_text(b"{}")

    def test_successful_output_extraction_for_each_runtime(self) -> None:
        self.assertEqual(
            cross_ai._extract_final_review_text(_opencode_output("Finding: OpenCode answer")),
            "Finding: OpenCode answer",
        )
        self.assertEqual(
            cross_ai._extract_claude_review_text(json.dumps({"result": "Finding: Claude answer"}).encode()),
            "Finding: Claude answer",
        )
        self.assertEqual(
            cross_ai._extract_codex_review_text(_codex_output("Finding: Codex answer")),
            "Finding: Codex answer",
        )

    def test_minimum_length_and_review_marker_rules_apply_to_each_runtime(self) -> None:
        outputs = {
            "opencode": _opencode_output("Finding: a sufficiently long answer"),
            "claude": json.dumps({"result": "Finding: a sufficiently long answer"}).encode(),
            "codex": _codex_output("Finding: a sufficiently long answer"),
        }
        markerless_outputs = {
            "opencode": _opencode_output("a sufficiently long answer without markers"),
            "claude": json.dumps({"result": "a sufficiently long answer without markers"}).encode(),
            "codex": _codex_output("a sufficiently long answer without markers"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for runtime, stdout in outputs.items():
                with self.subTest(runtime=runtime, rule="success"):
                    request = _request(root, runtime)
                    self.assertEqual(
                        cross_ai._validate_runtime_output(request, stdout),
                        (
                            "Finding: a sufficiently long answer",
                            None,
                        ),
                    )
                with self.subTest(runtime=runtime, rule="minimum length"):
                    request = cross_ai.ReviewRequest(**{**_request(root, runtime).__dict__, "min_output_chars": 100})
                    text, error = cross_ai._validate_runtime_output(request, stdout)
                    self.assertIsNone(text)
                    self.assertIn("too short", error or "")
                with self.subTest(runtime=runtime, rule="review marker"):
                    request = cross_ai.ReviewRequest(
                        **{**_request(root, runtime).__dict__, "require_review_markers": True}
                    )
                    text, error = cross_ai._validate_runtime_output(request, markerless_outputs[runtime])
                    self.assertIsNone(text)
                    self.assertIn("expected review markers", error or "")

    def test_malformed_output_keeps_opencode_strict_and_codex_tolerant(self) -> None:
        malformed_opencode = _opencode_output("Finding: valid event") + b"\nnot-json\n"
        with self.assertRaisesRegex(ValueError, r"line 3 is not valid JSON"):
            cross_ai._extract_final_review_text(malformed_opencode)
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory), "opencode")
            text, error = cross_ai._validate_runtime_output(request, malformed_opencode)
        self.assertIsNone(text)
        self.assertIn("not valid JSON", error or "")

        with self.assertRaisesRegex(ValueError, "Claude output is not valid JSON"):
            cross_ai._extract_claude_review_text(b"not-json")

        malformed_codex = b"not-json\n" + _codex_output("Finding: valid Codex event") + b"\n{}\n"
        self.assertEqual(
            cross_ai._extract_codex_review_text(malformed_codex),
            "Finding: valid Codex event",
        )
        with self.assertRaisesRegex(ValueError, "no completed agent message"):
            cross_ai._extract_codex_review_text(b"not-json\n{}\n")

    def test_opencode_shared_validation_failure_retries_directly(self) -> None:
        calls: list[str | None] = []

        async def attempt(
            request: cross_ai.ReviewRequest,
            *,
            attach_url: str | None,
            config_path: Path,
        ) -> cross_ai.ReviewAttempt:
            del request, config_path
            calls.append(attach_url)
            shared = attach_url is not None
            return cross_ai.ReviewAttempt(
                command=["fake-opencode", "shared" if shared else "direct"],
                stdout=b"",
                timed_out=False,
                returncode=0,
                final_text=None if shared else "Finding: direct retry",
                validation_error="shared output was invalid" if shared else None,
                permission_denials=(),
                mode="shared" if shared else "direct",
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = cross_ai.ReviewRequest(
                **{**_request(root, "opencode").__dict__, "attach_url": "http://127.0.0.1:1234"}
            )
            with mock.patch.object(cross_ai, "_review_attempt", side_effect=attempt):
                attempts = asyncio.run(cross_ai._collect_review_attempts(request, config_path=root / "config"))

        self.assertEqual(calls, ["http://127.0.0.1:1234", None])
        self.assertEqual([item.mode for item in attempts], ["shared", "direct"])
        self.assertEqual(attempts[-1].final_text, "Finding: direct retry")

    def test_command_construction_preserves_runtime_flags_and_shared_context_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_files = (root / "copied context.md", root / "second.md")
            base_prompt = "base review prompt"
            opencode_request = cross_ai.ReviewRequest(
                runtime_bin="/bin/opencode",
                review_model=cross_ai.ReviewerSpec(
                    "oc", "opencode", "provider/model", cross_ai.DeepSeekV4ProReasoning.MAX
                ),
                repo_root=root,
                context_files=context_files,
                prompt=base_prompt,
                output_dir=root,
                temporary_dir=root,
                timeout_seconds=1,
                attach_url="http://127.0.0.1:1234",
                min_output_chars=5,
                require_review_markers=True,
            )
            claude_request = cross_ai.ReviewRequest(
                **{
                    **opencode_request.__dict__,
                    "runtime_bin": "/bin/claude",
                    "review_model": cross_ai.ReviewerSpec(
                        "claude", "claude", "opus", cross_ai.ClaudeOpus5Reasoning.HIGH
                    ),
                    "attach_url": None,
                }
            )
            codex_request = cross_ai.ReviewRequest(
                **{
                    **opencode_request.__dict__,
                    "runtime_bin": "/bin/codex",
                    "review_model": cross_ai.ReviewerSpec(
                        "codex", "codex", "gpt-test", cross_ai.CodexSolReasoning.XHIGH
                    ),
                    "attach_url": None,
                }
            )

            opencode = cross_ai._review_command(opencode_request, attach_url=opencode_request.attach_url)
            claude = cross_ai._review_command(claude_request, attach_url=None)
            codex = cross_ai._review_command(codex_request, attach_url=None)

        self.assertEqual(opencode[:2], ["/bin/opencode", "run"])
        self.assertEqual(opencode[opencode.index("--model") + 1], "provider/model")
        self.assertEqual(opencode[opencode.index("--agent") + 1], "plan")
        self.assertEqual(opencode[opencode.index("--format") + 1], "json")
        self.assertEqual(opencode[opencode.index("--variant") + 1], "max")
        self.assertEqual(opencode[opencode.index("--attach") + 1], "http://127.0.0.1:1234")
        self.assertEqual(opencode[opencode.index("--dir") + 1], str(root))
        self.assertEqual(opencode[opencode.index("--title") + 1], "cross-ai-oc")
        self.assertEqual(opencode[-2:], ["--", base_prompt])
        for path in context_files:
            self.assertIn(str(path), opencode)

        self.assertEqual(claude[:2], ["/bin/claude", "-p"])
        claude_prompt = claude[2]
        self.assertIn(base_prompt, claude_prompt)
        self.assertIn("Review the following copied context files", claude_prompt)
        for path in context_files:
            self.assertIn(str(path), claude_prompt)
        self.assertEqual(claude[claude.index("--model") + 1], "opus")
        self.assertEqual(claude[claude.index("--permission-mode") + 1], "plan")
        self.assertEqual(claude[claude.index("--output-format") + 1], "json")
        self.assertEqual(claude[claude.index("--effort") + 1], "high")

        self.assertEqual(codex[:2], ["/bin/codex", "exec"])
        codex_prompt = codex[-1]
        self.assertEqual(claude_prompt, codex_prompt)
        self.assertIn(base_prompt, codex_prompt)
        for option in ("--sandbox", "--ephemeral", "--json", "--skip-git-repo-check", "--cd", "--config"):
            self.assertIn(option, codex)
        self.assertEqual(codex[codex.index("--model") + 1], "gpt-test")
        self.assertEqual(codex[codex.index("--sandbox") + 1], "read-only")
        self.assertEqual(codex[codex.index("--cd") + 1], str(root))
        self.assertEqual(codex[codex.index("--config") + 1], 'model_reasoning_effort="xhigh"')

    def test_registry_table_and_summary_cover_status_paths(self) -> None:
        table = cross_ai._reviewer_registry_table()
        self.assertIn("deepseek-v4-pro", table)
        self.assertIn("off", table)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            denial = cross_ai.PermissionDenial("make", "denied", "rule")
            failed = cross_ai.ReviewResult(
                **{
                    **_result(root, "failed").__dict__,
                    "valid_output": False,
                    "validation_error": "bad",
                    "permission_denials": (denial,),
                }
            )
            path = root / "summary.md"
            cross_ai._write_run_summary(
                path,
                cross_ai.RunSummary(
                    root, "review", root, None, root / "config", (root / "input",), [_result(root), failed], 61
                ),
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("invalid output (bad)", text)
            self.assertIn("denied command", text)

    def test_execute_reviews_orders_results(self) -> None:
        async def completed(result: cross_ai.ReviewResult) -> cross_ai.ReviewResult:
            return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = (
                cross_ai.ReviewerSpec("first", "codex", "test"),
                cross_ai.ReviewerSpec("second", "codex", "test"),
            )
            run = cross_ai.PreparedRun({}, root, "review", root, root, root / "config", (), "prompt", models, False)
            tasks = [completed(_result(root, "second")), completed(_result(root, "first"))]
            with mock.patch.object(cross_ai, "_review_tasks", return_value=tasks):
                results = asyncio.run(cross_ai._execute_reviews(run, None))
            self.assertEqual([item.model.slug for item in results], ["first", "second"])

    def test_review_attempt_success(self) -> None:
        class Process:
            returncode = 0

            async def communicate(self) -> tuple[bytes, None]:
                event = {"type": "item.completed", "item": {"type": "agent_message", "text": "valid answer"}}
                return json.dumps(event).encode(), None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)
            with mock.patch.object(asyncio, "create_subprocess_exec", return_value=Process()):
                attempt = asyncio.run(cross_ai._review_attempt(request, attach_url=None, config_path=root / "config"))
        self.assertEqual(attempt.returncode, 0)
        self.assertEqual(attempt.final_text, "valid answer")
        self.assertFalse(attempt.timed_out)

    def test_main_async_happy_path_cleans_temporary_directory(self) -> None:
        async def execute(run: cross_ai.PreparedRun, *, started_at: float) -> int:
            del run, started_at
            return 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / "run-temp"
            temporary.mkdir()
            prepared = cross_ai.PreparedRun(
                {}, root, "review", root, temporary, root / "config", (), "prompt", (), False
            )
            with (
                mock.patch.object(cross_ai, "_create_temporary_directory", return_value=temporary),
                mock.patch.object(cross_ai, "_prepare_run", return_value=prepared),
                mock.patch.object(cross_ai, "_print_startup"),
                mock.patch.object(cross_ai, "_execute_with_temporary_limit", side_effect=execute),
                mock.patch.object(cross_ai, "_remove_temporary_directory") as remove,
            ):
                self.assertEqual(asyncio.run(cross_ai._main_async(_args())), 0)
            remove.assert_called_once_with(temporary)


if __name__ == "__main__":
    unittest.main()
