#!/usr/bin/env python3
"""Run parallel adversarial reviews from any repository."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import http.client
import json
import os
import re
import shlex
import shutil
import signal
import socket
import stat
import sys
import sysconfig
import tempfile
import tomllib
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from time import monotonic, time

DEFAULT_REVIEW_PROMPT = (
    "Adversarially review the attached context. Focus on correctness, design fit, "
    "security/privacy, operational risk, and whether this is ready to ship. "
    "Return Critical/High/Medium/Low findings first, ordered by severity, with "
    "file/line references where possible. End explicitly with GO or NO-GO, a brief "
    "rationale, and any remaining actionable blockers. Avoid nitpicks. Do not edit files."
)
DEFAULT_PLAN_PROMPT = (
    "Create an implementation-ready plan for the stated goal and attached context. "
    "Identify assumptions, affected files, ordered implementation steps, risks, and "
    "verification. Do not edit files."
)

DEFAULT_MIN_OUTPUT_CHARS = 200
OUTPUT_INVALID_RETURNCODE = 86
DEFAULT_OUTPUT_DIR = ".adversarial-reviews"
OPENCODE_TEMP_ROOT = "/tmp"
OPENCODE_TEMP_LIMIT_BYTES = 2 * 1024**3
OPENCODE_TEMP_POLL_SECONDS = 1.0
OPENCODE_BUN_ARTIFACT_GRACE_SECONDS = 60.0
OPENCODE_BUN_ARTIFACT_PATTERN = re.compile(r"^\.[0-9a-f]+-\d{8}\.so$")
PROCESS_TERMINATION_GRACE_SECONDS = 3.0
TERMINATION_SIGNALS = (signal.SIGHUP, signal.SIGTERM, signal.SIGQUIT)
REVIEW_MARKER_PATTERN = re.compile(
    r"\b(Critical|High|Medium|Low|Finding|Findings|Verdict|GO|NO-GO|SHIP|NO-SHIP|"
    r"No ship|ship blocker|blocker|issue|issues)\b",
    re.IGNORECASE,
)
MENTAL_MODEL = """Run bounded AI reviewers in one planning or review pass.

Caller protocol: run the default cheap profile, read and fix its actionable
findings, and repeat on the current changes until every default reviewer says
GO. Then explicitly run --premium. If premium finds blockers, fix them, restore
cheap-profile GO on the current changes, and explicitly run --premium again.

Cross-AI keeps no state and performs one pass only: it does not iterate, fix
code, retain convergence state, perform automatic premium runs, or interpret
GO/NO-GO as readiness. Exit 0 means only that every selected report was
technically valid.
Use --profile NAME for a named team, --reviewer for targeted work, --all for
every configured reviewer, `cross-ai doctor` for runtime diagnostics, and
`cross-ai --init-config` to create the one active configuration file.

Reviewers may read and search the chosen repository and use read-only Git, but
may not edit files, run tests or builds, access the network, or delegate work.
"""

CONFIG_FILE_NAME = "config.toml"
TEMPLATE_FILE_NAME = "cross-ai.config.toml"
CONFIG_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:+-]*$")
REASONING_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
KNOWN_RUNTIMES = frozenset(("opencode", "claude", "codex"))
STANDARD_RUNTIME_PATHS = {
    "opencode": Path.home() / ".opencode/bin/opencode",
    "claude": Path.home() / ".local/bin/claude",
    "codex": Path.home() / ".local/bin/codex",
}


@dataclass(frozen=True)
class ReviewerSpec:
    slug: str
    runtime: str
    model: str
    timeout_seconds: int
    reasoning: str | None = None


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    binary: str | None = None


@dataclass(frozen=True)
class CrossAIConfig:
    path: Path
    default_profile: str
    default_timeout_seconds: int
    runtimes: dict[str, RuntimeSpec]
    profiles: dict[str, tuple[str, ...]]
    reviewers: dict[str, ReviewerSpec]


class ConfigurationError(ValueError):
    """Raised when the active Cross-AI TOML configuration is invalid."""


@dataclass(frozen=True)
class ReviewResult:
    model: ReviewerSpec
    returncode: int
    output_path: Path
    valid_output: bool
    elapsed_seconds: float
    validation_error: str | None = None
    permission_denials: tuple[PermissionDenial, ...] = ()


@dataclass(frozen=True)
class ReviewRequest:
    runtime_bin: str
    review_model: ReviewerSpec
    repo_root: Path
    context_files: tuple[Path, ...]
    prompt: str
    output_dir: Path
    temporary_dir: Path
    timeout_seconds: int
    attach_url: str | None
    min_output_chars: int
    require_review_markers: bool


@dataclass(frozen=True)
class ReviewAttempt:
    command: list[str]
    stdout: bytes
    timed_out: bool
    returncode: int
    final_text: str | None
    validation_error: str | None
    permission_denials: tuple[PermissionDenial, ...]
    mode: str


@dataclass(frozen=True)
class PermissionDenial:
    command: str
    reason: str
    matched_rule: str


@dataclass
class OpenCodeServer:
    process: asyncio.subprocess.Process
    url: str
    log_path: Path


@dataclass(frozen=True)
class RunSummary:
    repo_root: Path
    mode: str
    output_dir: Path
    server: OpenCodeServer | None
    opencode_config_path: Path
    context_files: tuple[Path, ...]
    results: list[ReviewResult]
    elapsed_seconds: float


@dataclass(frozen=True)
class PreparedRun:
    runtime_bins: dict[str, str]
    repo_root: Path
    mode: str
    output_dir: Path
    temporary_dir: Path
    opencode_config_path: Path
    context_files: tuple[Path, ...]
    prompt: str
    models: tuple[ReviewerSpec, ...]
    shared_mode: bool


class TemporaryDirectoryLimitExceeded(RuntimeError):
    """Raised when one Cross-AI run consumes too much temporary storage."""


class TerminationSignalReceived(RuntimeError):
    """Raised after a termination signal has triggered orderly cleanup."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum}")
        self.signum = signum


def _render_command(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-") or "input"


def _existing_directory(raw_path: str, *, label: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"{label} is not an existing directory: {path}")
    return path


def _execution_guardrails(repo_root: Path, mode: str) -> str:
    return (
        "This is a headless, non-interactive session in a trusted workspace. "
        "The tool restrictions are a behavioral and resource boundary, not a "
        "hostile-code security sandbox. Never request permission or approval "
        "because an unanswered prompt deadlocks the run. Do not delegate to "
        "nested agents, invoke task, or use @explore. Your complete working "
        f"boundary is {repo_root}. Freely read and search files beneath that exact "
        "directory. Never inspect sibling directories, parent directories, or "
        "any other absolute path. If evidence is unavailable inside the boundary, "
        f"state that limitation and continue the {mode}. The process already "
        "runs from the repository root, so invoke git diff, git status, git show, "
        "or git log without git -C. If any tool call is denied, do not search for "
        "workarounds: state the denied command and limitation in your final "
        "response immediately. Do not edit files."
    )


def _opencode_permissions(repo_root: Path) -> dict[str, object]:
    # This policy limits reviewer behavior and resource use in a trusted workspace.
    # It intentionally permits repository reads and read-only Git, while blocking
    # edits, tests, builds, arbitrary shell commands, networking, and delegation.
    bash_permissions = {"*": "deny"}
    for command in ("diff", "status", "log", "show"):
        bash_permissions[f"git {command}*"] = "allow"
        bash_permissions[f"git -C {repo_root} {command}*"] = "allow"
        quoted_root = shlex.quote(str(repo_root))
        if quoted_root != str(repo_root):
            bash_permissions[f"git -C {quoted_root} {command}*"] = "allow"

    return {
        "*": "deny",
        "read": {
            "*": "allow",
            ".env": "deny",
            ".env.*": "deny",
            ".env.example": "allow",
        },
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "bash": bash_permissions,
        "edit": "deny",
        "write": "deny",
        "task": "deny",
        "question": "deny",
        "external_directory": {"*": "deny"},
        "doom_loop": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "skill": "deny",
    }


def _active_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        base = Path(config_home).expanduser()
        if not base.is_absolute():
            raise ConfigurationError("XDG_CONFIG_HOME must be an absolute path")
    else:
        base = Path.home() / ".config"
    return base / "cross-ai" / CONFIG_FILE_NAME


def _template_path() -> Path:
    installed = Path(sysconfig.get_path("data")) / "share" / "cross-ai-review" / TEMPLATE_FILE_NAME
    checkout = Path(__file__).with_name(TEMPLATE_FILE_NAME)
    for candidate in (installed, checkout):
        if candidate.is_file():
            return candidate
    raise ConfigurationError("shipped configuration template is unavailable; reinstall cross-ai-review")


def _required_table(document: dict[str, object], key: str) -> dict[str, object]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{key} must be a TOML table")
    return value


def _ensure_keys(table: dict[str, object], *, label: str, allowed: frozenset[str], required: frozenset[str]) -> None:
    unknown = set(table).difference(allowed)
    missing = required.difference(table)
    if unknown:
        raise ConfigurationError(f"{label} has unknown key(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigurationError(f"{label} is missing required key(s): {', '.join(sorted(missing))}")


def _safe_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not CONFIG_NAME_PATTERN.fullmatch(value):
        raise ConfigurationError(f"{label} must use lowercase letters, digits, and hyphens")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{label} must be a positive integer")
    return value


def _safe_model(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not MODEL_PATTERN.fullmatch(value):
        raise ConfigurationError(f"{label} must be a safe non-empty model identifier")
    return value


def _safe_reasoning(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not REASONING_PATTERN.fullmatch(value):
        raise ConfigurationError(f"{label} must be a safe reasoning identifier")
    return value


def _configured_binary(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigurationError(f"{label} must be a string")
    if not value:
        raise ConfigurationError(f"{label} must be a non-empty absolute path")
    if not Path(value).is_absolute():
        raise ConfigurationError(f"{label} must be an absolute path: {value}")
    return value


def _parse_runtimes(runtime_table: dict[str, object]) -> dict[str, RuntimeSpec]:
    runtimes: dict[str, RuntimeSpec] = {}
    for name, raw_runtime in runtime_table.items():
        runtime_name = _safe_name(name, label="runtime name")
        if runtime_name not in KNOWN_RUNTIMES:
            raise ConfigurationError(f"unsupported runtime: {runtime_name}")
        if not isinstance(raw_runtime, dict):
            raise ConfigurationError(f"runtimes.{runtime_name} must be a TOML table")
        _ensure_keys(
            raw_runtime, label=f"runtimes.{runtime_name}", allowed=frozenset(("binary",)), required=frozenset()
        )
        binary = _configured_binary(raw_runtime.get("binary"), label=f"runtimes.{runtime_name}.binary")
        runtimes[runtime_name] = RuntimeSpec(runtime_name, binary)
    if not runtimes:
        raise ConfigurationError("runtimes must configure at least one runtime")
    return runtimes


def _parse_reviewers(
    reviewer_table: dict[str, object], runtimes: dict[str, RuntimeSpec], default_timeout: int
) -> dict[str, ReviewerSpec]:
    reviewers: dict[str, ReviewerSpec] = {}
    for slug, raw_reviewer in reviewer_table.items():
        reviewer_slug = _safe_name(slug, label="reviewer name")
        if not isinstance(raw_reviewer, dict):
            raise ConfigurationError(f"reviewers.{reviewer_slug} must be a TOML table")
        _ensure_keys(
            raw_reviewer,
            label=f"reviewers.{reviewer_slug}",
            allowed=frozenset(("runtime", "model", "reasoning", "timeout_seconds")),
            required=frozenset(("runtime", "model")),
        )
        runtime = _safe_name(raw_reviewer["runtime"], label=f"reviewers.{reviewer_slug}.runtime")
        if runtime not in runtimes:
            raise ConfigurationError(f"reviewers.{reviewer_slug} references unconfigured runtime: {runtime}")
        timeout = _positive_int(
            raw_reviewer.get("timeout_seconds", default_timeout), label=f"reviewers.{reviewer_slug}.timeout_seconds"
        )
        reviewers[reviewer_slug] = ReviewerSpec(
            slug=reviewer_slug,
            runtime=runtime,
            model=_safe_model(raw_reviewer["model"], label=f"reviewers.{reviewer_slug}.model"),
            reasoning=_safe_reasoning(raw_reviewer.get("reasoning"), label=f"reviewers.{reviewer_slug}.reasoning"),
            timeout_seconds=timeout,
        )
    if not reviewers:
        raise ConfigurationError("reviewers must configure at least one reviewer")
    return reviewers


def _parse_profiles(profile_table: dict[str, object], reviewers: dict[str, ReviewerSpec]) -> dict[str, tuple[str, ...]]:
    profiles: dict[str, tuple[str, ...]] = {}
    for name, raw_profile in profile_table.items():
        profile_name = _safe_name(name, label="profile name")
        if profile_name == "all":
            raise ConfigurationError("profiles.all is reserved; --all selects every configured reviewer")
        if not isinstance(raw_profile, dict):
            raise ConfigurationError(f"profiles.{profile_name} must be a TOML table")
        _ensure_keys(
            raw_profile,
            label=f"profiles.{profile_name}",
            allowed=frozenset(("reviewers",)),
            required=frozenset(("reviewers",)),
        )
        members = raw_profile["reviewers"]
        if not isinstance(members, list) or not all(isinstance(member, str) for member in members):
            raise ConfigurationError(f"profiles.{profile_name}.reviewers must be an array of reviewer names")
        if not members:
            raise ConfigurationError(f"profiles.{profile_name}.reviewers must not be empty")
        if len(set(members)) != len(members):
            raise ConfigurationError(f"profiles.{profile_name}.reviewers contains duplicates")
        unknown_reviewers = set(members).difference(reviewers)
        if unknown_reviewers:
            raise ConfigurationError(
                f"profiles.{profile_name} references unknown reviewer(s): {', '.join(sorted(unknown_reviewers))}"
            )
        profiles[profile_name] = tuple(members)
    return profiles


def _load_config(path: Path | None = None) -> CrossAIConfig:
    config_path = path or _active_config_path()
    if not config_path.is_file():
        raise ConfigurationError(f"configuration file not found: {config_path}; run cross-ai --init-config")
    try:
        document = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"cannot read configuration {config_path}: {exc}") from None
    if not isinstance(document, dict):
        raise ConfigurationError("configuration root must be a TOML table")
    _ensure_keys(
        document,
        label="configuration",
        allowed=frozenset(("default_profile", "default_timeout_seconds", "runtimes", "profiles", "reviewers")),
        required=frozenset(("default_profile", "default_timeout_seconds", "runtimes", "profiles", "reviewers")),
    )
    default_profile = _safe_name(document["default_profile"], label="default_profile")
    default_timeout = _positive_int(document["default_timeout_seconds"], label="default_timeout_seconds")
    runtimes = _parse_runtimes(_required_table(document, "runtimes"))
    reviewers = _parse_reviewers(_required_table(document, "reviewers"), runtimes, default_timeout)
    profiles = _parse_profiles(_required_table(document, "profiles"), reviewers)
    if default_profile == "all" or default_profile not in profiles:
        raise ConfigurationError(f"default_profile must name a configured non-all profile: {default_profile}")
    return CrossAIConfig(config_path, default_profile, default_timeout, runtimes, profiles, reviewers)


def _init_config() -> Path:
    destination = _active_config_path()
    template = _template_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(template.read_text(encoding="utf-8"))
    except FileExistsError:
        raise ConfigurationError(f"configuration already exists and was not overwritten: {destination}") from None
    return destination


def _first_executable(*candidates: Path | str | None) -> str | None:
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _find_runtime(runtime: RuntimeSpec) -> str:
    if runtime.binary is not None:
        if not runtime.binary:
            raise ConfigurationError(f"runtimes.{runtime.name}.binary must be a non-empty absolute executable")
        configured = Path(runtime.binary)
        if not configured.is_absolute():
            raise ConfigurationError(f"runtimes.{runtime.name}.binary must be absolute: {runtime.binary}")
        if not configured.is_file() or not os.access(configured, os.X_OK):
            raise ConfigurationError(f"runtimes.{runtime.name}.binary is not an executable file: {runtime.binary}")
        return str(configured)
    candidate = _first_executable(STANDARD_RUNTIME_PATHS[runtime.name], shutil.which(runtime.name))
    if candidate is not None:
        return candidate
    raise ConfigurationError(f"{runtime.name} binary not found; configure runtimes.{runtime.name}.binary or install it")


def _reviewers_for_profile(config: CrossAIConfig, profile: str) -> list[ReviewerSpec]:
    return [config.reviewers[name] for name in config.profiles[profile]]


def _selected_reviewers(args: argparse.Namespace, config: CrossAIConfig) -> tuple[ReviewerSpec, ...]:
    if args.reviewer:
        if len(set(args.reviewer)) != len(args.reviewer):
            raise ConfigurationError("--reviewer cannot repeat the same reviewer slug")
        unknown = set(args.reviewer).difference(config.reviewers)
        if unknown:
            raise ConfigurationError(f"unknown reviewer(s): {', '.join(sorted(unknown))}")
        reviewers = [config.reviewers[name] for name in args.reviewer]
    else:
        if args.premium:
            profile = "premium"
        elif args.all:
            reviewers = list(config.reviewers.values())
            return tuple(reviewers)
        else:
            profile = args.profile or config.default_profile
        if profile not in config.profiles:
            raise ConfigurationError(f"unknown profile: {profile}")
        reviewers = _reviewers_for_profile(config, profile)
    return tuple(reviewers)


def _premium_focus(args: argparse.Namespace, config: CrossAIConfig) -> bool:
    if args.reviewer or args.all:
        return False
    return args.premium or (args.profile or config.default_profile) == "premium"


def _runtime_bins(config: CrossAIConfig, reviewers: tuple[ReviewerSpec, ...]) -> dict[str, str]:
    runtimes = {reviewer.runtime for reviewer in reviewers}
    return {runtime: _find_runtime(config.runtimes[runtime]) for runtime in runtimes}


def _create_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(1000):
        name = stamp if suffix == 0 else f"{stamp}-{suffix + 1}"
        candidate = output_root / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"could not allocate unique run directory under {output_root}")


def _temporary_directory_prefix() -> str:
    return f"cross-ai-{os.getuid()}-"


def _create_temporary_directory() -> Path:
    path = Path(
        tempfile.mkdtemp(
            prefix=_temporary_directory_prefix(),
            dir=OPENCODE_TEMP_ROOT,
        )
    )
    path.chmod(0o700)
    return path


def _remove_temporary_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if (
        path.parent != Path(OPENCODE_TEMP_ROOT)
        or not path.name.startswith(_temporary_directory_prefix())
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
    ):
        raise RuntimeError(f"refusing to remove unowned temporary directory: {path}")
    shutil.rmtree(path)


def _temporary_directory_size(path: Path) -> int:
    total = 0
    for directory, _, filenames in os.walk(path, followlinks=False):
        directory_path = Path(directory)
        for filename in filenames:
            try:
                total += (directory_path / filename).lstat().st_size
            except FileNotFoundError:
                continue
    return total


def _remove_stale_bun_artifacts(path: Path, *, older_than: float) -> None:
    cutoff = time() - older_than
    for candidate in path.iterdir():
        if not OPENCODE_BUN_ARTIFACT_PATTERN.fullmatch(candidate.name):
            continue
        try:
            metadata = candidate.lstat()
            if stat.S_ISREG(metadata.st_mode) and metadata.st_mtime <= cutoff:
                candidate.unlink()
        except FileNotFoundError:
            continue


async def _monitor_temporary_directory(path: Path, *, limit_bytes: int) -> None:
    while True:
        _remove_stale_bun_artifacts(
            path,
            older_than=OPENCODE_BUN_ARTIFACT_GRACE_SECONDS,
        )
        size_bytes = _temporary_directory_size(path)
        if size_bytes > limit_bytes:
            raise TemporaryDirectoryLimitExceeded(
                f"temporary directory exceeded {limit_bytes} bytes: {size_bytes} bytes in {path}"
            )
        await asyncio.sleep(OPENCODE_TEMP_POLL_SECONDS)


def _existing_file(raw_path: str, *, label: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return path


def _copy_context_files(files: list[Path], output_dir: Path) -> tuple[Path, ...]:
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    seen: dict[str, int] = {}
    for source in files:
        base = _slugify(source.name)
        count = seen.get(base, 0)
        seen[base] = count + 1
        name = base if count == 0 else f"{count + 1}-{base}"
        destination = input_dir / name
        shutil.copy2(source, destination)
        copied.append(destination)
    return tuple(copied)


def _write_opencode_config(output_dir: Path, repo_root: Path) -> Path:
    opencode_config_path = output_dir / "opencode.json"
    permissions = _opencode_permissions(repo_root)
    config = {
        "$schema": "https://opencode.ai/config.json",
        "permission": permissions,
        "agent": {
            "plan": {
                "mode": "primary",
                "permission": permissions,
            }
        },
    }
    opencode_config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    return opencode_config_path


def _opencode_environment(
    opencode_config_path: Path,
    temporary_dir: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")
    env["OPENCODE_CONFIG"] = str(opencode_config_path)
    env["TMPDIR"] = str(temporary_dir)
    env["BUN_TMPDIR"] = str(temporary_dir)
    return env


def _review_returncode(returncode: int | None, timed_out: bool) -> int:
    if timed_out:
        return 124
    return returncode if returncode is not None else 1


def _decode_review_output(stdout: bytes) -> list[dict[str, object]]:
    try:
        raw_text = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"review output is not valid UTF-8: {exc}") from exc

    events: list[dict[str, object]] = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"review output line {line_number} is not valid JSON: {exc.msg}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"review output line {line_number} is not a JSON object")
        events.append(event)
    return events


def _bash_permission_failure(
    event: dict[str, object],
) -> tuple[dict[str, object], str] | None:
    part = event.get("part")
    if not isinstance(part, dict):
        return None
    if part.get("type") != "tool" or part.get("tool") != "bash":
        return None
    state = part.get("state")
    if not isinstance(state, dict) or state.get("status") != "error":
        return None
    error = state.get("error")
    if not isinstance(error, str) or "permission" not in error.lower():
        return None
    return state, error


def _permission_denial(event: dict[str, object]) -> PermissionDenial | None:
    failure = _bash_permission_failure(event)
    if failure is None:
        return None
    state, error = failure
    tool_input = state.get("input")
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if not isinstance(command, str):
        command = "<command unavailable in OpenCode event>"
    return PermissionDenial(
        command=command,
        reason=error,
        matched_rule='bash "*" -> deny (no read-only allow rule matched)',
    )


def _permission_denials(
    events: list[dict[str, object]],
) -> tuple[PermissionDenial, ...]:
    return tuple(denial for event in events if (denial := _permission_denial(event)) is not None)


def _collect_review_parts(
    events: list[dict[str, object]],
) -> tuple[dict[str, list[str]], list[dict[str, object]]]:
    text_by_message: dict[str, list[str]] = {}
    finish_parts: list[dict[str, object]] = []
    for event in events:
        part = event.get("part")
        if event.get("type") == "step_finish":
            if isinstance(part, dict):
                finish_parts.append(part)
        elif event.get("type") == "text" and isinstance(part, dict):
            message_id = part.get("messageID")
            text = part.get("text")
            if isinstance(message_id, str) and isinstance(text, str):
                text_by_message.setdefault(message_id, []).append(text)
    return text_by_message, finish_parts


def _extract_final_review_text(stdout: bytes) -> str:
    text_by_message, finish_parts = _collect_review_parts(_decode_review_output(stdout))
    if not finish_parts:
        raise ValueError("review output has no completed assistant step")
    final_part = finish_parts[-1]
    if final_part.get("reason") != "stop":
        raise ValueError(f"review output has no final assistant response: reason={final_part.get('reason')!r}")
    message_id = final_part.get("messageID")
    if not isinstance(message_id, str):
        raise ValueError("final review step has no valid messageID")
    final_text = "\n".join(text_by_message.get(message_id, ())).strip()
    if not final_text:
        raise ValueError("final assistant response did not contain text")
    return final_text


def _validate_final_review_text(final_text: str, *, min_output_chars: int, require_review_markers: bool) -> str:
    if len(final_text) < min_output_chars:
        raise ValueError(
            f"final assistant response too short: {len(final_text)} chars, expected at least {min_output_chars}"
        )
    if require_review_markers and not REVIEW_MARKER_PATTERN.search(final_text):
        raise ValueError("final assistant response did not contain expected review markers")
    return final_text


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_healthcheck(port: int) -> bool:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("GET", "/global/health")
        response = connection.getresponse()
        return response.status == HTTPStatus.OK
    except OSError:
        return False
    finally:
        connection.close()


async def _start_opencode_server(
    *,
    opencode_bin: str,
    repo_root: Path,
    output_dir: Path,
    opencode_config_path: Path,
    temporary_dir: Path,
    timeout_seconds: int = 30,
) -> OpenCodeServer:
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"
    log_path = output_dir / "opencode-serve.log"
    command = [
        opencode_bin,
        "serve",
        "--hostname",
        "127.0.0.1",
        "--port",
        str(port),
        "--print-logs",
        "--log-level",
        "INFO",
    ]
    env = _opencode_environment(opencode_config_path, temporary_dir)
    with log_path.open("wb") as log_handle:
        log_handle.write(
            (
                f"# OpenCode shared server\n\nCommand:\n\n```bash\n{_render_command(command)}\n```\n\nOutput:\n\n"
            ).encode()
        )
        log_handle.flush()
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=repo_root,
            stdout=log_handle,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    try:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                raise RuntimeError(f"opencode serve exited before becoming healthy; see {log_path}")
            if await asyncio.to_thread(_server_healthcheck, port):
                return OpenCodeServer(process=process, url=url, log_path=log_path)
            await asyncio.sleep(0.25)
    except BaseException:
        await _terminate_process(process)
        raise

    await _terminate_process(process)
    raise TimeoutError(f"opencode serve did not become healthy; see {log_path}")


async def _stop_opencode_server(server: OpenCodeServer) -> None:
    await _terminate_process(server.process)


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_group(process_group: int, signum: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_group, signum)


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    process_group = process.pid
    _signal_process_group(process_group, signal.SIGTERM)
    deadline = asyncio.get_running_loop().time() + PROCESS_TERMINATION_GRACE_SECONDS
    while _process_group_exists(process_group) and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    if _process_group_exists(process_group):
        _signal_process_group(process_group, signal.SIGKILL)
    if process.returncode is None:
        await process.wait()


def _opencode_review_command(request: ReviewRequest, *, attach_url: str | None) -> list[str]:
    review_model = request.review_model
    command = [
        request.runtime_bin,
        "run",
        "--model",
        review_model.model,
    ]
    command.extend(["--agent", "plan"])
    command.extend(["--format", "json"])
    if review_model.reasoning:
        command.extend(["--variant", review_model.reasoning])
    if attach_url is not None:
        command.extend(["--attach", attach_url])
    command.extend(["--dir", str(request.repo_root)])
    for file_path in request.context_files:
        command.extend(["-f", str(file_path)])
    command.extend(["--title", f"cross-ai-{review_model.slug}"])
    command.extend(["--", request.prompt])
    return command


def _copied_context_prompt(request: ReviewRequest) -> str:
    context = "\n".join(f"- {path}" for path in request.context_files)
    return (
        f"{request.prompt}\n\nReview the following copied context files in addition to repository evidence:\n{context}"
    )


def _claude_review_command(request: ReviewRequest) -> list[str]:
    prompt = _copied_context_prompt(request)
    command = [
        request.runtime_bin,
        "-p",
        prompt,
        "--model",
        request.review_model.model,
        "--permission-mode",
        "plan",
        "--output-format",
        "json",
    ]
    if request.review_model.reasoning:
        command.extend(["--effort", request.review_model.reasoning])
    return command


def _codex_review_command(request: ReviewRequest) -> list[str]:
    prompt = _copied_context_prompt(request)
    command = [
        request.runtime_bin,
        "exec",
        "--model",
        request.review_model.model,
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--json",
        "--skip-git-repo-check",
        "--cd",
        str(request.repo_root),
    ]
    if request.review_model.reasoning:
        command.extend(
            [
                "--config",
                f'model_reasoning_effort="{request.review_model.reasoning}"',
            ]
        )
    command.append(prompt)
    return command


def _review_command(request: ReviewRequest, *, attach_url: str | None) -> list[str]:
    if request.review_model.runtime == "opencode":
        return _opencode_review_command(request, attach_url=attach_url)
    if request.review_model.runtime == "claude":
        return _claude_review_command(request)
    if request.review_model.runtime == "codex":
        return _codex_review_command(request)
    raise ValueError(f"unsupported reviewer runtime: {request.review_model.runtime}")


def _extract_claude_review_text(stdout: bytes) -> str:
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Claude output is not valid JSON: {exc}") from exc
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, str) or not result.strip():
        raise ValueError("Claude output has no non-empty result")
    return result.strip()


def _extract_codex_review_text(stdout: bytes) -> str:
    try:
        raw_text = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Codex output is not valid UTF-8: {exc}") from exc

    messages: list[str] = []
    for line in raw_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            messages.append(text.strip())
    if not messages:
        raise ValueError("Codex output has no completed agent message")
    return messages[-1]


def _validate_runtime_output(request: ReviewRequest, stdout: bytes) -> tuple[str | None, str | None]:
    extractors = {
        "opencode": _extract_final_review_text,
        "claude": _extract_claude_review_text,
        "codex": _extract_codex_review_text,
    }
    if request.review_model.runtime not in extractors:
        return None, f"unsupported output runtime: {request.review_model.runtime}"
    try:
        final_text = _validate_final_review_text(
            extractors[request.review_model.runtime](stdout),
            min_output_chars=request.min_output_chars,
            require_review_markers=request.require_review_markers,
        )
    except ValueError as exc:
        return None, str(exc)
    return final_text, None


async def _review_attempt(
    request: ReviewRequest, *, attach_url: str | None, opencode_config_path: Path
) -> ReviewAttempt:
    command = _review_command(request, attach_url=attach_url)

    env = (
        _opencode_environment(opencode_config_path, request.temporary_dir)
        if request.review_model.runtime == "opencode"
        else os.environ.copy()
    )
    env.setdefault("NO_COLOR", "1")
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=request.repo_root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=request.timeout_seconds)
    except TimeoutError:
        timed_out = True
        await _terminate_process(process)
        stdout, _ = await process.communicate()
    except BaseException:
        await _terminate_process(process)
        raise
    returncode = _review_returncode(process.returncode, timed_out)
    final_text: str | None = None
    validation_error: str | None = None
    permission_denials: tuple[PermissionDenial, ...] = ()
    if request.review_model.runtime == "opencode":
        with contextlib.suppress(ValueError):
            permission_denials = _permission_denials(_decode_review_output(stdout))
    if returncode == 0:
        final_text, validation_error = _validate_runtime_output(request, stdout)
    return ReviewAttempt(
        command=command,
        stdout=stdout,
        timed_out=timed_out,
        returncode=returncode,
        final_text=final_text,
        validation_error=validation_error,
        permission_denials=permission_denials,
        mode="shared" if attach_url is not None else "direct",
    )


async def _collect_review_attempts(request: ReviewRequest, *, opencode_config_path: Path) -> list[ReviewAttempt]:
    attempts = [
        await _review_attempt(
            request,
            attach_url=request.attach_url,
            opencode_config_path=opencode_config_path,
        )
    ]
    if (
        request.review_model.runtime == "opencode"
        and request.attach_url is not None
        and attempts[0].returncode == 0
        and attempts[0].validation_error is not None
    ):
        attempts.append(await _review_attempt(request, attach_url=None, opencode_config_path=opencode_config_path))
    return attempts


def _result_returncode(attempt: ReviewAttempt) -> int:
    if attempt.returncode == 0 and attempt.validation_error is not None:
        return OUTPUT_INVALID_RETURNCODE
    return attempt.returncode


def _review_result_header(
    review_model: ReviewerSpec,
    attempts: list[ReviewAttempt],
) -> str:
    attempt = attempts[-1]
    retry_note = ""
    if len(attempts) > 1:
        retry_note = (
            "A direct retry was performed because the attached shared-server "
            f"output failed validation: {attempts[0].validation_error}\n\n"
        )
    commands = "\n".join(f"- {item.mode}: {_render_command(item.command)}" for item in attempts)
    return (
        f"# Cross-AI Result: {review_model.slug}\n\n"
        f"Execution attempts:\n\n{commands}\n\n"
        f"{retry_note}"
        f"Command:\n\n```bash\n{_render_command(attempt.command)}\n```\n\n"
        "Output:\n\n"
    )


def _review_result_body(
    request: ReviewRequest,
    attempt: ReviewAttempt,
    *,
    valid_output: bool,
) -> bytes:
    sections: list[bytes] = []
    if attempt.timed_out:
        sections.append(f"ERROR: review timed out after {request.timeout_seconds} seconds.\n\n".encode())
    if attempt.validation_error is not None:
        sections.append((f"ERROR: review output failed validation: {attempt.validation_error}\n\n").encode())
    if valid_output and attempt.final_text is not None:
        sections.append(f"{attempt.final_text}\n".encode())
    else:
        sections.append(attempt.stdout)
    return b"".join(sections)


def _write_review_result(
    request: ReviewRequest,
    output_path: Path,
    attempts: list[ReviewAttempt],
    *,
    valid_output: bool,
) -> None:
    attempt = attempts[-1]
    with output_path.open("wb") as handle:
        handle.write(_review_result_header(request.review_model, attempts).encode())
        handle.write(_review_result_body(request, attempt, valid_output=valid_output))


async def _run_review(request: ReviewRequest, *, opencode_config_path: Path) -> ReviewResult:
    started_at = monotonic()
    attempts = await _collect_review_attempts(request, opencode_config_path=opencode_config_path)
    attempt = attempts[-1]
    returncode = _result_returncode(attempt)
    valid_output = returncode == 0 and attempt.validation_error is None
    output_path = request.output_dir / f"{request.review_model.slug}.md"
    _write_review_result(request, output_path, attempts, valid_output=valid_output)

    return ReviewResult(
        model=request.review_model,
        returncode=returncode,
        output_path=output_path,
        valid_output=valid_output,
        elapsed_seconds=monotonic() - started_at,
        validation_error=attempt.validation_error,
        permission_denials=tuple(denial for item in attempts for denial in item.permission_denials),
    )


def _resolve_output_root(repo_root: Path, output_dir: str) -> Path:
    output_root = Path(output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    else:
        output_root = output_root.resolve()
    if not output_root.is_relative_to(repo_root):
        raise SystemExit(f"output directory must be inside repository root: {output_root}")
    return output_root


def _build_prompt(mode: str, goal: str | None, repo_root: Path, *, premium: bool = False) -> str:
    prompt = DEFAULT_REVIEW_PROMPT if mode == "review" else DEFAULT_PLAN_PROMPT
    if premium:
        prompt = (
            f"{prompt}\n\n"
            "Premium strategic focus: assess long-term architecture, product consequences, maintainability, "
            "and decisions that would be difficult to reverse."
        )
    if goal:
        prompt = f"Goal / decision to support: {goal}\n\n{prompt}"
    return f"{prompt}\n\n{_execution_guardrails(repo_root, mode)}"


def _result_status(result: ReviewResult) -> str:
    if result.valid_output:
        return "ok"
    if result.validation_error is not None:
        return f"invalid output ({result.validation_error})"
    return f"exit {result.returncode}"


def _write_run_summary(
    summary_path: Path,
    summary: RunSummary,
) -> None:
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Cross-AI Run Summary\n\n")
        handle.write(f"- repo root: `{summary.repo_root}`\n")
        handle.write(f"- mode: `{summary.mode}`\n")
        handle.write(f"- output directory: `{summary.output_dir}`\n")
        handle.write(f"- total wall time: `{_format_duration(summary.elapsed_seconds)}`\n")
        server_log = summary.server.log_path if summary.server else "not started (direct mode)"
        handle.write(f"- execution mode: `{'shared' if summary.server else 'direct'}`\n")
        handle.write(f"- shared opencode server: `{server_log}`\n")
        handle.write(f"- generated OpenCode permission config: `{summary.opencode_config_path}`\n")
        handle.write("- context files:\n")
        for file_path in summary.context_files:
            handle.write(f"  - `{file_path}`\n")
        handle.write("\n")
        for result in summary.results:
            handle.write(
                f"- {result.model.slug}: {_result_status(result)}, "
                f"elapsed `{_format_duration(result.elapsed_seconds)}`, "
                f"`{result.output_path}`\n"
            )
            for denial in result.permission_denials:
                handle.write(f"  - denied command: `{denial.command}`\n")
                handle.write(f"  - permission rule: `{denial.matched_rule}`\n")
                handle.write(f"  - reason: `{denial.reason}`\n")


def _print_run_summary(
    output_dir: Path,
    server: OpenCodeServer | None,
    results: list[ReviewResult],
    summary_path: Path,
    elapsed_seconds: float,
) -> None:
    sys.stdout.write(f"Cross-AI output: {output_dir}\n")
    if server is not None:
        sys.stdout.write(f"- shared opencode server log: {server.log_path}\n")
    else:
        sys.stdout.write("- execution mode: direct (no shared server)\n")
    for result in results:
        sys.stdout.write(
            f"- {result.model.slug}: exit={result.returncode} "
            f"elapsed={_format_duration(result.elapsed_seconds)} "
            f"{result.output_path}\n"
        )
        for denial in result.permission_denials:
            sys.stdout.write(f"  denied command: {denial.command}\n")
            sys.stdout.write(f"  permission rule: {denial.matched_rule}\n")
            sys.stdout.write(f"  reason: {denial.reason}\n")
    sys.stdout.write(f"- summary: {summary_path}\n")
    sys.stdout.write(f"- total wall time: {_format_duration(elapsed_seconds)}\n")


def _prepare_run(
    args: argparse.Namespace,
    models: tuple[ReviewerSpec, ...],
    runtime_bins: dict[str, str],
    premium_focus: bool,
    *,
    temporary_dir: Path,
) -> PreparedRun:
    repo_root = _existing_directory(args.repo_root, label="repository root")
    raw_files = [_existing_file(path, label="context file") for path in args.context_file]
    output_root = _resolve_output_root(repo_root, args.output_dir)
    prompt = _build_prompt(args.mode, args.goal, repo_root, premium=premium_focus)

    output_dir = _create_run_directory(output_root)
    opencode_config_path = _write_opencode_config(output_dir, repo_root)
    context_files = _copy_context_files(raw_files, output_dir)
    return PreparedRun(
        runtime_bins=runtime_bins,
        repo_root=repo_root,
        mode=args.mode,
        output_dir=output_dir,
        temporary_dir=temporary_dir,
        opencode_config_path=opencode_config_path,
        context_files=context_files,
        prompt=prompt,
        models=models,
        shared_mode=not args.no_shared_server,
    )


def _print_startup(run: PreparedRun) -> None:
    sys.stdout.write("Cross-AI startup\n")
    sys.stdout.write(f"- repo root: {run.repo_root}\n")
    sys.stdout.write(f"- mode: {run.mode}; execution: {'shared' if run.shared_mode else 'direct'}\n")
    sys.stdout.write(f"- output directory: {run.output_dir}\n")
    sys.stdout.write(f"- temporary directory: {run.temporary_dir}; limit={OPENCODE_TEMP_LIMIT_BYTES} bytes\n")
    for model in run.models:
        sys.stdout.write(
            f"- reviewer: {model.slug}; runtime={model.runtime}; "
            f"model={model.model}:"
            f"{model.reasoning or 'default'}; "
            f"timeout={model.timeout_seconds}s\n"
        )
    sys.stdout.flush()


def _reviewer_profile_names(config: CrossAIConfig, reviewer_slug: str) -> str:
    profiles = [profile for profile, reviewer_slugs in config.profiles.items() if reviewer_slug in reviewer_slugs]
    return ",".join(profiles) or "-"


def _format_timeout(seconds: int) -> str:
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _reviewer_registry_table(config: CrossAIConfig) -> str:
    headers = (
        "reviewer",
        "profiles",
        "runtime",
        "model",
        "reasoning",
        "timeout",
    )
    rows = [
        (
            reviewer.slug,
            _reviewer_profile_names(config, reviewer.slug),
            reviewer.runtime,
            reviewer.model,
            reviewer.reasoning or "default",
            _format_timeout(reviewer.timeout_seconds),
        )
        for reviewer in config.reviewers.values()
    ]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows)
    return "\n".join(lines)


async def _start_run_server(run: PreparedRun) -> OpenCodeServer | None:
    if not run.shared_mode or not any(reviewer.runtime == "opencode" for reviewer in run.models):
        return None
    return await _start_opencode_server(
        opencode_bin=run.runtime_bins["opencode"],
        repo_root=run.repo_root,
        output_dir=run.output_dir,
        opencode_config_path=run.opencode_config_path,
        temporary_dir=run.temporary_dir,
    )


def _review_tasks(run: PreparedRun, server: OpenCodeServer | None) -> list[Coroutine[object, object, ReviewResult]]:
    return [
        _run_review(
            ReviewRequest(
                runtime_bin=run.runtime_bins[review_model.runtime],
                review_model=review_model,
                repo_root=run.repo_root,
                context_files=run.context_files,
                prompt=run.prompt,
                output_dir=run.output_dir,
                temporary_dir=run.temporary_dir,
                timeout_seconds=review_model.timeout_seconds,
                attach_url=(server.url if server and review_model.runtime == "opencode" else None),
                min_output_chars=DEFAULT_MIN_OUTPUT_CHARS,
                require_review_markers=run.mode == "review",
            ),
            opencode_config_path=run.opencode_config_path,
        )
        for review_model in run.models
    ]


async def _execute_reviews(run: PreparedRun, server: OpenCodeServer | None) -> list[ReviewResult]:
    results: list[ReviewResult] = []
    tasks = [asyncio.create_task(task) for task in _review_tasks(run, server)]
    try:
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            results.append(result)
            status = "valid" if result.valid_output else "invalid"
            sys.stdout.write(
                f"Cross-AI reviewer completed: {result.model.slug} "
                f"elapsed={_format_duration(result.elapsed_seconds)} "
                f"exit={result.returncode} status={status} "
                f"{result.output_path}\n"
            )
            sys.stdout.flush()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    model_order = {model.slug: index for index, model in enumerate(run.models)}
    return sorted(results, key=lambda result: model_order[result.model.slug])


def _finish_run(
    run: PreparedRun,
    server: OpenCodeServer | None,
    results: list[ReviewResult],
    elapsed_seconds: float,
) -> int:
    summary_path = run.output_dir / "summary.md"
    _write_run_summary(
        summary_path,
        RunSummary(
            repo_root=run.repo_root,
            mode=run.mode,
            output_dir=run.output_dir,
            server=server,
            opencode_config_path=run.opencode_config_path,
            context_files=run.context_files,
            results=results,
            elapsed_seconds=elapsed_seconds,
        ),
    )
    _print_run_summary(
        run.output_dir,
        server,
        results,
        summary_path,
        elapsed_seconds,
    )
    return 0 if all(result.valid_output for result in results) else 1


async def _execute_prepared_run(run: PreparedRun, *, started_at: float) -> int:
    server: OpenCodeServer | None = None
    try:
        server = await _start_run_server(run)
        results = await _execute_reviews(run, server)
    finally:
        if server is not None:
            await _stop_opencode_server(server)
    return _finish_run(run, server, results, monotonic() - started_at)


async def _execute_with_temporary_limit(
    run: PreparedRun,
    *,
    started_at: float,
) -> int:
    run_task = asyncio.create_task(_execute_prepared_run(run, started_at=started_at))
    monitor_task = asyncio.create_task(
        _monitor_temporary_directory(
            run.temporary_dir,
            limit_bytes=OPENCODE_TEMP_LIMIT_BYTES,
        )
    )
    try:
        done, _ = await asyncio.wait(
            (run_task, monitor_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if monitor_task in done:
            monitor_task.result()
            raise RuntimeError("temporary directory monitor stopped unexpectedly")
        return run_task.result()
    finally:
        for task in (run_task, monitor_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(run_task, monitor_task, return_exceptions=True)


async def _main_async(
    args: argparse.Namespace,
    models: tuple[ReviewerSpec, ...],
    runtime_bins: dict[str, str],
    premium_focus: bool,
) -> int:
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    if main_task is None:
        raise RuntimeError("Cross-AI main task is unavailable")
    termination_signal: int | None = None

    def request_termination(signum: int) -> None:
        nonlocal termination_signal
        termination_signal = signum
        main_task.cancel()

    registered_signals: list[signal.Signals] = []
    temporary_dir: Path | None = None
    try:
        for signum in TERMINATION_SIGNALS:
            loop.add_signal_handler(signum, request_termination, signum)
            registered_signals.append(signum)
        temporary_dir = _create_temporary_directory()
        started_at = monotonic()
        run = _prepare_run(args, models, runtime_bins, premium_focus, temporary_dir=temporary_dir)
        _print_startup(run)
        return await _execute_with_temporary_limit(run, started_at=started_at)
    except asyncio.CancelledError:
        if termination_signal is not None:
            raise TerminationSignalReceived(termination_signal) from None
        raise
    finally:
        if temporary_dir is not None:
            _remove_temporary_directory(temporary_dir)
        for signum in registered_signals:
            loop.remove_signal_handler(signum)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run parallel planning or review with registered AI runtimes. "
            "Context files are copied into the review output directory before "
            "being attached, so files from /tmp work reliably."
        )
    )
    parser.add_argument(
        "context_file",
        nargs="*",
        help="Context file(s) to attach to every selected reviewer.",
    )
    parser.add_argument(
        "--goal",
        help="Task-specific goal prepended to the built-in prompt for the selected mode.",
    )
    parser.add_argument(
        "--repo-root",
        help="Exact existing project directory used for both OpenCode cwd and --dir.",
    )
    parser.add_argument(
        "--mode",
        choices=("review", "plan"),
        default="review",
        help="Work to perform. Defaults to review.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for review outputs, relative to repo root by default.",
    )
    reviewer_selection = parser.add_mutually_exclusive_group()
    reviewer_selection.add_argument(
        "--reviewer",
        action="append",
        default=None,
        metavar="SLUG",
        help=(
            "Run only this registered reviewer. Repeat for multiple reviewers. "
            "Useful for diagnosis, retries, and timing calibration."
        ),
    )
    reviewer_selection.add_argument(
        "--premium",
        action="store_true",
        help="Run the configured premium profile.",
    )
    reviewer_selection.add_argument(
        "--all",
        action="store_true",
        help="Run every configured reviewer in one intentional full pass.",
    )
    reviewer_selection.add_argument(
        "--profile",
        metavar="NAME",
        help="Run one named configured profile.",
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Create the active XDG configuration from the shipped template without overwriting it.",
    )
    parser.add_argument(
        "--no-shared-server",
        action="store_true",
        help="Run each reviewer directly without a shared OpenCode server (debug mode).",
    )
    return parser


def _parse_args(argv: list[str], parser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    return (parser or _build_parser()).parse_args(argv)


def _build_doctor_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Check Cross-AI configuration and configured runtime binaries.")


def _validate_run_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.repo_root:
        parser.error("--repo-root is required for a review or planning run")
    if not args.context_file:
        parser.error("at least one context file is required for a review or planning run")


def _doctor(config: CrossAIConfig) -> int:
    sys.stdout.write(f"Cross-AI configuration: {config.path}\n")
    sys.stdout.write(f"Default profile: {config.default_profile}\n")
    unavailable = False
    for runtime in config.runtimes.values():
        try:
            binary = _find_runtime(runtime)
        except ConfigurationError as exc:
            unavailable = True
            sys.stdout.write(f"Runtime {runtime.name}: unavailable ({exc})\n")
        else:
            sys.stdout.write(f"Runtime {runtime.name}: {binary}\n")
    profiled = {reviewer for members in config.profiles.values() for reviewer in members}
    orphans = sorted(set(config.reviewers).difference(profiled))
    if orphans:
        sys.stdout.write(f"Warning: orphan reviewers selectable with --reviewer/--all: {', '.join(orphans)}\n")
    sys.stdout.write(f"\n{_reviewer_registry_table(config)}\n")
    return 1 if unavailable else 0


def _dispatch(argv: list[str]) -> int:
    if not argv:
        parser = _build_parser()
        sys.stdout.write(MENTAL_MODEL)
        sys.stdout.write("\n")
        sys.stdout.write(parser.format_help())
        return 0
    if argv[0] == "doctor":
        _build_doctor_parser().parse_args(argv[1:])
        return _doctor(_load_config())
    parser = _build_parser()
    args = _parse_args(argv, parser)
    if args.init_config:
        if (
            args.context_file
            or args.repo_root
            or args.goal
            or args.mode != "review"
            or args.output_dir != DEFAULT_OUTPUT_DIR
            or args.reviewer
            or args.premium
            or args.all
            or args.profile
            or args.no_shared_server
        ):
            parser.error("--init-config cannot be combined with review or planning arguments")
        destination = _init_config()
        sys.stdout.write(f"Created Cross-AI configuration: {destination}\n")
        return 0
    _validate_run_arguments(args, parser)
    config = _load_config()
    models = _selected_reviewers(args, config)
    runtime_bins = _runtime_bins(config, models)
    premium_focus = _premium_focus(args, config)
    sys.stdout.write(f"Starting cross-ai in {args.mode} mode. Run cross-ai without arguments for usage guidance.\n")
    sys.stdout.flush()
    return asyncio.run(_main_async(args, models, runtime_bins, premium_focus))


def main() -> int:
    try:
        return _dispatch(sys.argv[1:])
    except ConfigurationError as exc:
        sys.stderr.write(f"Cross-AI configuration error: {exc}\n")
        sys.stderr.flush()
        return 2
    except TemporaryDirectoryLimitExceeded as exc:
        sys.stderr.write(f"Cross-AI stopped: {exc}\n")
        sys.stderr.flush()
        return 75
    except TerminationSignalReceived as exc:
        sys.stderr.write("Cross-AI terminated; reviewer processes and temporary files were cleaned up.\n")
        sys.stderr.flush()
        return 128 + exc.signum
    except KeyboardInterrupt:
        sys.stderr.write("Cross-AI interrupted; reviewer processes and temporary files were cleaned up.\n")
        sys.stderr.flush()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
