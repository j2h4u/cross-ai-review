# External Configuration Implementation Checklist

- [x] Add a single active XDG config at `$XDG_CONFIG_HOME/cross-ai/config.toml`, with `~/.config` as the XDG default and no alternate lookup locations.
- [x] Ship a TOML template and add `cross-ai --init-config` that creates the active config without overwriting an existing file.
- [x] Model configuration hierarchically with `[runtimes.<name>]`, `[profiles.<name>]`, and `[reviewers.<slug>]`; omit schema versioning and reviewer `enabled` flags.
- [x] Add a required top-level `default_timeout_seconds` and optional per-reviewer `timeout_seconds` overrides.
- [x] Strictly validate the complete config before starting processes or creating run artifacts.
- [x] Resolve binaries explicitly: configured absolute path, then the runtime's standard path, then `PATH`; remove runtime binary environment-variable overrides.
- [x] Keep command construction, permissions, output parsing, process lifecycle, and cleanup inside the closed Python runtime adapters.
- [x] Preserve default, `--premium`, `--all`, and explicit `--reviewer` selection while adding named `--profile` selection.
- [x] Add `cross-ai doctor` for non-fatal reviewer/profile connectivity diagnostics and availability checks across the full configured catalog.
- [x] Present a concise agent usage protocol without adding iteration, convergence tracking, automatic premium execution, or semantic GO evaluation to Cross-AI.
- [x] Update focused unit and lifecycle coverage, including invalid config fail-fast behavior and effective timeout inheritance.
- [x] Update README and packaging metadata, and verify the installed wheel can initialize and load its XDG config on Python 3.12.
- [x] Run the complete local quality gate and an independent final audit; resolve every confirmed issue.
