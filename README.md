# Cross-AI Review

Cross-AI Review runs bounded planning and adversarial-review passes through a
configured panel of AI command-line tools. It keeps the reviewers focused on
repository evidence while preserving one Markdown report per model.

The permission model assumes a trusted operator, workspace, repository, and
reviewer. It is a behavioral and resource boundary, not a hostile-code security
sandbox. Reviewers may freely read and search the selected repository and use
read-only `git diff`, `git status`, `git show`, and `git log`. Editing, tests,
builds, general shell commands, web access, and nested delegation are disabled.

## Requirements

- Python 3.12 or newer
- [OpenCode](https://opencode.ai/) for the standard reviewers
- Claude Code and Codex only when their optional reviewer profiles are enabled

Provider authentication and model access are managed by the corresponding AI
CLI. No credentials are stored in this repository.

## Installation

Install as a Python CLI with your preferred isolated package tool:

```bash
uv tool install .
```

For development, run the repository launcher directly:

```bash
./cross-ai
```

### Planned package distribution

Cross-AI Review is planned to be published as a standalone package on PyPI so
it can be installed on another machine by package name alone:

```bash
uv tool install cross-ai-review
```

Once published, `uv tool upgrade cross-ai-review` will update the isolated
installation. PyPI publishing and the release workflow are not configured yet;
the commands above describe the intended distribution model rather than the
current installation path.

## Usage

Pass the exact workspace and one or more context files:

```bash
cross-ai \
  --mode review \
  --repo-root /path/to/project \
  docs/plan.md
```

The default profile runs the standard OpenCode reviewers. `--premium` runs the
configured premium gate, `--all` runs every enabled reviewer, and repeated
`--reviewer SLUG` options select specific reviewers. Run `cross-ai` without
arguments for the full operational workflow and current registry.

Reports are written under `.adversarial-reviews/` inside the selected workspace
unless `--output-dir` specifies another directory within that workspace.

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
