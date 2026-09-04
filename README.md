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

Run the focused lifecycle suite from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

The suite covers the permission contract, temporary-directory limit and
cleanup, successful report preservation, signal-driven process-group teardown,
and partial shared-server startup.

This public repository currently has no open-source license. Public visibility
does not grant permission to copy, modify, or redistribute the code.
