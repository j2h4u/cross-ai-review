import argparse
import inspect
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest import mock

import pytest

import cross_ai

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "cross_ai.py"

CONFIG = (
    """
default_profile = "standard"
default_timeout_seconds = 77

[runtimes.opencode]
[runtimes.claude]
[runtimes.codex]

[profiles.standard]
reviewers = ["oc-review"]
[profiles.custom]
reviewers = ["codex-review"]
[profiles.premium]
reviewers = ["claude-review"]

[reviewers.oc-review]
runtime = "opencode"
model = "provider/model"
reasoning = "max"

[reviewers.claude-review]
runtime = "claude"
model = "opus"
reasoning = "low"

[reviewers.codex-review]
runtime = "codex"
model = "gpt-test"
reasoning = "xhigh"
timeout_seconds = 12

[reviewers.orphan-review]
runtime = "opencode"
model = "orphan-model"
""".strip()
    + "\n"
)


def write_config(path: Path, text: str = CONFIG) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def args(**overrides: object) -> argparse.Namespace:
    values = {
        "reviewer": None,
        "premium": False,
        "all": False,
        "profile": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_active_config_uses_xdg_path_and_home_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert cross_ai._active_config_path() == tmp_path / "xdg/cross-ai/config.toml"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.setattr(cross_ai.Path, "home", lambda: tmp_path / "home")
    assert cross_ai._active_config_path() == tmp_path / "home/.config/cross-ai/config.toml"
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative-xdg")
    with pytest.raises(cross_ai.ConfigurationError, match="absolute path"):
        cross_ai._active_config_path()


def test_init_creates_active_template_and_refuses_overwrite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    destination = cross_ai._init_config()
    assert destination == config_home / "cross-ai/config.toml"
    assert destination.read_text(encoding="utf-8") == cross_ai._template_path().read_text(encoding="utf-8")
    destination.write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(cross_ai.ConfigurationError, match="not overwritten"):
        cross_ai._init_config()
    assert destination.read_text(encoding="utf-8") == "sentinel\n"


def test_init_refuses_dangling_symlink_without_touching_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    destination = config_home / "cross-ai/config.toml"
    target = tmp_path / "target-config.toml"
    destination.parent.mkdir(parents=True)
    destination.symlink_to(target)
    with pytest.raises(cross_ai.ConfigurationError, match="not overwritten"):
        cross_ai._init_config()
    assert destination.is_symlink()
    assert not target.exists()


def test_init_maps_exclusive_create_race_to_non_overwrite_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    destination = config_home / "cross-ai/config.toml"
    destination.parent.mkdir(parents=True)
    original_open = Path.open
    original_write_text = Path.write_text

    def race_open(path: Path, mode: str = "r", *args: object, **kwargs: object):
        if path == destination and mode == "x":
            original_write_text(path, "racing writer\n", **kwargs)
            raise FileExistsError(path)
        return original_open(path, mode, *args, **kwargs)

    with (
        mock.patch.object(Path, "open", autospec=True, side_effect=race_open),
        pytest.raises(cross_ai.ConfigurationError, match="not overwritten"),
    ):
        cross_ai._init_config()
    assert destination.read_text(encoding="utf-8") == "racing writer\n"


@pytest.mark.parametrize(
    ("label", "body", "message"),
    [
        ("unknown root key", CONFIG.replace("default_profile", "unknown = true\ndefault_profile", 1), "unknown key"),
        ("missing required root key", CONFIG.replace("default_timeout_seconds = 77\n", ""), "missing required"),
        ("wrong root type", CONFIG.replace('default_profile = "standard"', "default_profile = 3"), "default_profile"),
        (
            "unknown reviewer reference",
            CONFIG.replace('reviewers = ["oc-review"]', 'reviewers = ["missing-review"]', 1),
            "unknown reviewer",
        ),
        (
            "reviewer cross reference",
            CONFIG.replace('runtime = "opencode"', 'runtime = "missing-runtime"', 1),
            "unconfigured runtime",
        ),
        (
            "duplicate profile members",
            CONFIG.replace('reviewers = ["oc-review"]', 'reviewers = ["oc-review", "oc-review"]', 1),
            "duplicates",
        ),
        ("invalid name", CONFIG.replace("[profiles.standard]", "[profiles.Standard]"), "profile name"),
        ("invalid model", CONFIG.replace('model = "provider/model"', 'model = "bad model"'), "model"),
        ("invalid reasoning", CONFIG.replace('reasoning = "max"', 'reasoning = "Max!"'), "reasoning"),
        ("duplicate TOML key", CONFIG + '\n[reviewers.oc-review]\nmodel = "second"\n', "cannot read"),
    ],
)
def test_config_validation_is_strict(tmp_path: Path, label: str, body: str, message: str) -> None:
    with pytest.raises(cross_ai.ConfigurationError, match=message):
        cross_ai._load_config(write_config(tmp_path / f"{label.replace(' ', '-')}.toml", body))


def test_config_rejects_wrong_table_types_and_unknown_nested_keys(tmp_path: Path) -> None:
    cases = (
        CONFIG.replace("[runtimes.opencode]", "runtimes = []", 1),
        CONFIG.replace("[profiles.standard]", "[profiles.standard]\nunknown = true", 1),
        CONFIG.replace("[reviewers.oc-review]", "[reviewers.oc-review]\nenabled = false", 1),
        CONFIG.replace('default_profile = "standard"', 'version = 1\ndefault_profile = "standard"'),
    )
    for body in cases:
        with pytest.raises(cross_ai.ConfigurationError):
            cross_ai._load_config(write_config(tmp_path / f"{len(body)}.toml", body))


@pytest.mark.parametrize("binary", ["", "relative/claude"])
def test_load_rejects_empty_or_relative_binary_on_unselected_runtime(tmp_path: Path, binary: str) -> None:
    body = CONFIG.replace("[runtimes.claude]", f'[runtimes.claude]\nbinary = "{binary}"', 1)
    with pytest.raises(cross_ai.ConfigurationError, match="binary"):
        cross_ai._load_config(write_config(tmp_path / f"binary-{len(binary)}.toml", body))


def test_default_timeout_inheritance_override_and_bool_rejection(tmp_path: Path) -> None:
    config = cross_ai._load_config(write_config(tmp_path / "config.toml"))
    assert config.default_timeout_seconds == 77
    assert config.reviewers["oc-review"].timeout_seconds == 77
    assert config.reviewers["codex-review"].timeout_seconds == 12
    for body in (
        CONFIG.replace("default_timeout_seconds = 77", "default_timeout_seconds = true"),
        CONFIG.replace("timeout_seconds = 12", "timeout_seconds = false"),
    ):
        with pytest.raises(cross_ai.ConfigurationError, match="positive integer"):
            cross_ai._load_config(write_config(tmp_path / f"bool-{len(body)}.toml", body))


def test_reviewer_spec_has_no_code_default_timeout() -> None:
    assert inspect.signature(cross_ai.ReviewerSpec).parameters["timeout_seconds"].default is inspect.Parameter.empty


def test_no_enabled_or_version_fields_are_part_of_the_public_model(tmp_path: Path) -> None:
    config = cross_ai._load_config(write_config(tmp_path / "config.toml"))
    assert not hasattr(config, "version")
    assert not hasattr(config.reviewers["oc-review"], "enabled")


def test_profile_and_reviewer_selection_including_orphans(tmp_path: Path) -> None:
    config = cross_ai._load_config(write_config(tmp_path / "config.toml"))
    assert [item.slug for item in cross_ai._selected_reviewers(args(), config)] == ["oc-review"]
    assert [item.slug for item in cross_ai._selected_reviewers(args(profile="custom"), config)] == ["codex-review"]
    assert [item.slug for item in cross_ai._selected_reviewers(args(premium=True), config)] == ["claude-review"]
    assert [item.slug for item in cross_ai._selected_reviewers(args(all=True), config)] == [
        "oc-review",
        "claude-review",
        "codex-review",
        "orphan-review",
    ]
    assert [item.slug for item in cross_ai._selected_reviewers(args(reviewer=["orphan-review"]), config)] == [
        "orphan-review"
    ]
    with pytest.raises(cross_ai.ConfigurationError, match="cannot repeat"):
        cross_ai._selected_reviewers(args(reviewer=["oc-review", "oc-review"]), config)


def test_premium_focus_only_applies_to_premium_profile_selection(tmp_path: Path) -> None:
    config = cross_ai._load_config(write_config(tmp_path / "config.toml"))
    assert cross_ai._premium_focus(args(premium=True), config)
    assert cross_ai._premium_focus(args(profile="premium"), config)
    premium_default = cross_ai.CrossAIConfig(
        config.path,
        "premium",
        config.default_timeout_seconds,
        config.runtimes,
        config.profiles,
        config.reviewers,
    )
    assert cross_ai._premium_focus(args(), premium_default)
    assert not cross_ai._premium_focus(args(all=True), config)
    assert not cross_ai._premium_focus(args(reviewer=["claude-review"]), config)


def test_binary_resolution_configured_absolute_then_standard_then_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit = executable(tmp_path / "explicit")
    standard = executable(tmp_path / "standard")
    path_binary = executable(tmp_path / "path")
    runtime = cross_ai.RuntimeSpec("opencode", str(explicit))
    assert cross_ai._find_runtime(runtime) == str(explicit)
    with mock.patch.object(cross_ai, "STANDARD_RUNTIME_PATHS", {"opencode": standard}):
        with mock.patch.object(cross_ai.shutil, "which", return_value=str(path_binary)):
            assert cross_ai._find_runtime(cross_ai.RuntimeSpec("opencode")) == str(standard)
        standard.unlink()
        with mock.patch.object(cross_ai.shutil, "which", return_value=str(path_binary)):
            assert cross_ai._find_runtime(cross_ai.RuntimeSpec("opencode")) == str(path_binary)
    for runtime, variable in (("opencode", "OPENCODE_BIN"), ("claude", "CLAUDE_BIN"), ("codex", "CODEX_BIN")):
        monkeypatch.setenv(variable, str(explicit))
        with (
            mock.patch.object(cross_ai, "STANDARD_RUNTIME_PATHS", {runtime: tmp_path / "missing"}),
            mock.patch.object(cross_ai.shutil, "which", return_value=None),
            pytest.raises(cross_ai.ConfigurationError, match="binary not found"),
        ):
            cross_ai._find_runtime(cross_ai.RuntimeSpec(runtime))


def test_configured_binary_is_absolute_executable_or_fails(tmp_path: Path) -> None:
    for value, message in (
        ("relative/opencode", "must be absolute"),
        (str(tmp_path / "missing"), "not an executable file"),
        (str(tmp_path / "not-executable"), "not an executable file"),
    ):
        if value.endswith("not-executable"):
            Path(value).write_text("", encoding="utf-8")
        with pytest.raises(cross_ai.ConfigurationError, match=message):
            cross_ai._find_runtime(cross_ai.RuntimeSpec("opencode", value))


def test_runtime_bin_resolution_only_checks_selected_runtimes(tmp_path: Path) -> None:
    config = cross_ai._load_config(write_config(tmp_path / "config.toml"))
    with mock.patch.object(cross_ai, "_find_runtime", return_value="/bin/selected") as find:
        assert cross_ai._runtime_bins(config, (config.reviewers["oc-review"],)) == {"opencode": "/bin/selected"}
    find.assert_called_once_with(config.runtimes["opencode"])


def test_doctor_checks_every_configured_runtime_and_only_warns_about_orphans(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config = cross_ai._load_config(write_config(tmp_path / "config.toml"))
    with mock.patch.object(cross_ai, "_find_runtime", side_effect=lambda runtime: f"/bin/{runtime.name}") as find:
        assert cross_ai._doctor(config) == 0
    assert {call.args[0].name for call in find.call_args_list} == {"opencode", "claude", "codex"}
    output = capsys.readouterr().out
    assert "Warning: orphan reviewers" in output
    assert "Runtime claude: /bin/claude" in output

    no_orphan = cross_ai._load_config(
        write_config(
            tmp_path / "no-orphan.toml",
            CONFIG.replace('\n[reviewers.orphan-review]\nruntime = "opencode"\nmodel = "orphan-model"\n', ""),
        )
    )
    with mock.patch.object(cross_ai, "_find_runtime", return_value="/bin/runtime"):
        cross_ai._doctor(no_orphan)
    assert "Warning:" not in capsys.readouterr().out


def test_invalid_config_fails_before_artifacts_or_runtime(tmp_path: Path) -> None:
    xdg = tmp_path / "xdg"
    config_path = write_config(
        xdg / "cross-ai/config.toml", CONFIG.replace("default_profile", "unknown = true\ndefault_profile", 1)
    )
    context = tmp_path / "context.md"
    context.write_text("context", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(tmp_path), "--output-dir", "reports", str(context)],
        cwd=tmp_path,
        env={**os.environ, "XDG_CONFIG_HOME": str(xdg)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "configuration error" in result.stderr
    assert config_path.is_file()
    assert not (tmp_path / "reports").exists()


def test_no_args_protocol_and_cli_dispatch(capsys: pytest.CaptureFixture[str]) -> None:
    assert cross_ai._dispatch([]) == 0
    output = capsys.readouterr().out.lower()
    assert "one planning or review pass" in output
    assert "cheap" in output
    assert "convergence" in output
    assert "premium" in output
    assert "blocker" in output
    assert "one pass" in output or "one-pass" in output
    assert "no state" in output or "stateless" in output
    assert "automatic premium" in output
    assert "exit 0" in output
    assert "technical" in output
    cheap = output.index("default cheap profile")
    premium = output.index("explicitly run --premium")
    blockers = output.index("premium finds blockers")
    cheap_again = output.index("cheap-profile go", blockers)
    premium_again = output.index("explicitly run --premium again", cheap_again)
    assert cheap < premium < blockers < cheap_again < premium_again
    assert "--profile name" in output
    assert "--premium" in output
    assert "--all" in output
    assert "--reviewer slug" in output


def test_dispatch_init_and_doctor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert cross_ai._dispatch(["--init-config"]) == 0
    assert "Created Cross-AI configuration" in capsys.readouterr().out
    config = cross_ai._load_config()
    with mock.patch.object(cross_ai, "_doctor", return_value=0) as doctor:
        assert cross_ai._dispatch(["doctor"]) == 0
    doctor.assert_called_once_with(config)


def test_review_prompt_requires_explicit_go_or_no_go(tmp_path: Path) -> None:
    prompt = cross_ai._build_prompt("review", None, tmp_path, premium=False).lower()
    assert "go/no-go" in prompt or "go or no-go" in prompt


def test_premium_prompt_has_strategic_focus_without_semantic_go_evaluation(tmp_path: Path) -> None:
    fake = executable(tmp_path / "opencode")
    config = cross_ai.CrossAIConfig(
        path=tmp_path / "config.toml",
        default_profile="standard",
        default_timeout_seconds=77,
        runtimes={"opencode": cross_ai.RuntimeSpec("opencode", str(fake))},
        profiles={"standard": ("reviewer",), "premium": ("reviewer",)},
        reviewers={"reviewer": cross_ai.ReviewerSpec("reviewer", "opencode", "provider/model", 77)},
    )
    context = tmp_path / "context.md"
    context.write_text("context", encoding="utf-8")
    args = argparse.Namespace(
        repo_root=str(tmp_path),
        context_file=[str(context)],
        output_dir="reports",
        mode="review",
        goal=None,
        reviewer=None,
        premium=True,
        all=False,
        profile=None,
        no_shared_server=True,
    )
    run = cross_ai._prepare_run(
        args,
        (config.reviewers["reviewer"],),
        {"opencode": str(fake)},
        True,
        temporary_dir=tmp_path / "temporary",
    )
    prompt = run.prompt.lower()
    assert "strategic" in prompt
    assert "semantic go" not in prompt


def test_runtime_command_flags_remain_stable(tmp_path: Path) -> None:
    request = cross_ai.ReviewRequest(
        runtime_bin="/bin/opencode",
        review_model=cross_ai.ReviewerSpec("oc", "opencode", "provider/model", 1, "max"),
        repo_root=tmp_path,
        context_files=(tmp_path / "one.md", tmp_path / "two.md"),
        prompt="review prompt",
        output_dir=tmp_path,
        temporary_dir=tmp_path,
        timeout_seconds=1,
        attach_url="http://127.0.0.1:1234",
        min_output_chars=5,
        require_review_markers=True,
    )
    opencode = cross_ai._review_command(request, attach_url=request.attach_url)
    assert opencode[:2] == ["/bin/opencode", "run"]
    assert opencode[opencode.index("--model") + 1] == "provider/model"
    assert opencode[opencode.index("--agent") + 1] == "plan"
    assert opencode[opencode.index("--format") + 1] == "json"
    assert opencode[opencode.index("--variant") + 1] == "max"
    assert opencode[opencode.index("--attach") + 1] == request.attach_url
    assert opencode[opencode.index("--dir") + 1] == str(tmp_path)
    assert opencode[opencode.index("--title") + 1] == "cross-ai-oc"
    for context_file in request.context_files:
        assert str(context_file) in opencode
    assert opencode[-2:] == ["--", "review prompt"]

    for runtime, model, expected in (
        ("claude", "opus", ("--permission-mode", "plan", "--output-format", "json", "--effort", "low")),
        ("codex", "gpt-test", ("--sandbox", "read-only", "--ephemeral", "--json", "--skip-git-repo-check")),
    ):
        command = cross_ai._review_command(
            cross_ai.ReviewRequest(
                **{
                    **request.__dict__,
                    "runtime_bin": f"/bin/{runtime}",
                    "review_model": cross_ai.ReviewerSpec(runtime, runtime, model, 1, "low"),
                    "attach_url": None,
                }
            ),
            attach_url=None,
        )
        for flag in expected:
            assert flag in command
        if runtime == "claude":
            assert command[:2] == ["/bin/claude", "-p"]
            assert command[command.index("--model") + 1] == model
            assert command[command.index("--effort") + 1] == "low"
        else:
            assert command[:2] == ["/bin/codex", "exec"]
            assert command[command.index("--model") + 1] == model
            assert command[command.index("--cd") + 1] == str(tmp_path)
            assert command[command.index("--config") + 1] == 'model_reasoning_effort="low"'


def test_installed_python312_cli_initializes_from_foreign_cwd(tmp_path: Path) -> None:
    python312 = Path("/home/j2h4u/.local/bin/python3.12")
    if not python312.is_file():
        pytest.skip("Python 3.12 is unavailable")
    dist = tmp_path / "dist"
    dist.mkdir()
    build = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    wheel = next(dist.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert any(name.endswith("share/cross-ai-review/cross-ai.config.toml") for name in archive.namelist())
    venv = tmp_path / "venv"
    subprocess.run([str(python312), "-m", "venv", str(venv)], check=True, capture_output=True)
    installed_python = venv / "bin/python"
    install = subprocess.run(
        ["uv", "pip", "install", "--python", str(installed_python), "--no-deps", str(wheel)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert install.returncode == 0, install.stderr
    xdg = tmp_path / "foreign-xdg"
    foreign = tmp_path / "foreign-cwd"
    foreign.mkdir()
    init = subprocess.run(
        [str(venv / "bin/cross-ai"), "--init-config"],
        cwd=foreign,
        env={**os.environ, "XDG_CONFIG_HOME": str(xdg)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert init.returncode == 0, init.stderr
    loaded = subprocess.run(
        [str(installed_python), "-c", "import cross_ai; print(cross_ai._load_config().path)"],
        cwd=foreign,
        env={**os.environ, "XDG_CONFIG_HOME": str(xdg)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert loaded.returncode == 0, loaded.stderr
    assert loaded.stdout.strip() == str(xdg / "cross-ai/config.toml")
    assert (xdg / "cross-ai/config.toml").is_file()
