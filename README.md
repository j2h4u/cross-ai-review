# Cross-AI Review

> ### Consensus Cosplay
>
> Five copies of the same model receive nearly identical context, agree with
> each other, and get presented as independent expert judgment.

Cross-AI Review helps avoid that trap. It runs adversarial review or fusion
planning across deliberately different AI command-line tools, then preserves
one inspectable Markdown report per reviewer. Agreement is still evidence to
evaluate—not proof of an independent consensus.

## What it does

- Supports two workflows: adversarial review and fusion planning.
- Runs OpenCode, Claude Code, and Codex reviewers through explicit profiles.
- Executes reviewers concurrently with per-model timeouts.
- Keeps repository access read-only and blocks editing, builds, web access, and
  nested delegation.
- Validates runtime-specific output and records failed or denied attempts.
- Cleans up process groups and bounded per-run temporary storage while keeping
  the review reports.

The permission model assumes a trusted operator, workspace, repository, and
reviewer. It is a behavioral and resource boundary, not a hostile-code security
sandbox. Reviewers may freely read and search the selected repository and use
read-only `git diff`, `git status`, `git show`, and `git log`. Editing, tests,
builds, general shell commands, web access, and nested delegation are disabled.

## Requirements

- Python 3.12 or newer
- [OpenCode](https://opencode.ai/) for the standard reviewers
- Claude Code and Codex only when selected reviewers use those runtimes

Provider authentication and model access are managed by the corresponding AI
CLI. No credentials are stored in this repository.

## Configuration

Cross-AI has one active TOML file: `$XDG_CONFIG_HOME/cross-ai/config.toml`, or
`~/.config/cross-ai/config.toml` when `XDG_CONFIG_HOME` is unset. It never
searches the current directory or other fallback locations. Create it from the
shipped template with:

```bash
cross-ai --init-config
```

The command refuses to overwrite an existing file. The complete schema is:

```toml
default_profile = "standard"
default_timeout_seconds = 900

[runtimes.opencode]
binary = "/absolute/path/to/opencode" # optional

[profiles.standard]
reviewers = ["reviewer-slug"]

[reviewers.reviewer-slug]
runtime = "opencode"
model = "provider/model"
reasoning = "max"       # optional
timeout_seconds = 1200   # optional; inherits the top-level default
```

The top-level tables are exactly `runtimes`, `profiles`, and `reviewers`;
unknown keys, missing keys, invalid names, types, and references are errors.
There is no schema `version` and no reviewer `enabled` flag. Reviewer names use
lowercase letters, digits, and hyphens. Runtime names are `opencode`, `claude`,
and `codex`; profiles contain one or more reviewer slugs. The default profile
must name a configured profile other than the reserved `all` name.

Use `cross-ai doctor` to validate the configuration and inspect every runtime
in the configured catalog. Missing binaries are reported as unavailable and do
not prevent the orphan-reviewer warning. A reviewer not listed in any profile
is an orphan: it remains selectable explicitly or through `--all`.

Runtime binaries resolve in this order: an explicitly configured absolute,
executable `binary`; the runtime's standard installation path; then its name
on `PATH`. Environment variables such as `OPENCODE_BIN`, `CLAUDE_BIN`, and
`CODEX_BIN` are not consulted.

## Installation

Install as a Python CLI with your preferred isolated package tool:

```bash
uv tool install .
```

This installs the `cross-ai` command into UV's executable directory (usually
`~/.local/bin` on Linux; run `uv tool dir --bin` to see the exact location).

For development, run the single source script directly:

```bash
./cross_ai.py
```

## Usage

Cross-AI has two modes:

- **Review** (`--mode review`) asks each selected reviewer to challenge the
  supplied context for correctness, design, security, operational risk, and
  ship readiness.
- **Fusion planning** (`--mode plan`) asks each selected reviewer for an
  implementation-ready plan so their assumptions, sequencing, and tradeoffs can
  be compared before implementation. Plans remain separate Markdown reports;
  the command does not silently collapse them into manufactured consensus.

Pass the exact workspace and one or more context files. Review is the default:

```bash
cross-ai \
  --mode review \
  --repo-root /path/to/project \
  docs/plan.md
```

Run fusion planning against requirements or a design brief:

```bash
cross-ai \
  --mode plan \
  --repo-root /path/to/project \
  --goal "Design the smallest safe implementation" \
  docs/requirements.md
```

The default profile runs `default_profile`. `--profile NAME` selects another
configured profile, `--premium` selects the profile named `premium`, `--all`
runs every configured reviewer (including orphans), and repeated
`--reviewer SLUG` options select specific reviewers. Run `cross-ai` without
arguments for the concise usage protocol and CLI help. No-args help does not
load the active configuration or display its reviewer catalog.

Reports are written under `.adversarial-reviews/` inside the selected workspace
unless `--output-dir` specifies another directory within that workspace.

## Agent usage protocol

The agent-owned review loop is deliberately explicit: run the cheap/default
profile, fix findings, and repeat until the cheap reviewers converge; run the
explicit premium profile; if premium finds blockers, return to the cheap loop
until it converges again, then run premium once more. Use a named profile when
the task needs a different team, `--reviewer` for diagnosis or a retry, and
`--all` only when the full configured catalog is wanted.

Cross-AI itself is one pass with no persisted state: it never iterates, tracks
convergence, or runs premium automatically. Its exit status is technical only
(successful process/output validation), not a semantic GO decision; the agent
must inspect the Markdown reports and decide what to do next.

## Temporary-file containment

Every invocation creates a private directory under `/tmp`, passes it to
OpenCode through `TMPDIR` and `BUN_TMPDIR`, and monitors it with a 2 GiB limit.
The directory is removed after success, failure, timeout, `SIGINT`, `SIGTERM`,
`SIGHUP`, or `SIGQUIT`, after reviewer process groups have been stopped.
Markdown reports remain in the output directory. As with any userspace cleanup,
`SIGKILL` and host failure cannot run the cleanup handler.

## Development

Install the locked development environment and run the focused lifecycle suite:

```bash
uv sync
just unit
```

UV uses Python 3.14 for development, while the dependency-free CLI remains
compatible with system Python 3.12 and newer.

The suite covers the permission contract, temporary-directory limit and
cleanup, successful report preservation, signal-driven process-group teardown,
and partial shared-server startup.

Run `just check` for formatting, lint, type, dead-code, lock, compile, and
packaging checks. Run `just crap-check` for the per-function CRAP threshold of
30, or `just verify` for the complete local quality contract. `just coverage`
is a diagnostic report and does not enforce a standalone coverage target.

This public repository currently has no open-source license. Public visibility
does not grant permission to copy, modify, or redistribute the code.
