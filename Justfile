set shell := ["bash", "-uc"]
export UV_LINK_MODE := "hardlink"

# Show available repository commands.
default:
    @just --list

# Compile Python sources for syntax errors.
compile:
    uv run python -m compileall -q cross_ai.py scripts tests

# Verify uv.lock is synchronized with pyproject.toml.
lock-check:
    uv lock --check

# Lint all Python code.
lint:
    uv run ruff check --preview cross_ai.py scripts tests

# Check formatting without writing files.
fmt-check:
    uv run ruff format --no-preview --check cross_ai.py scripts tests

# Type-check production and QA helper code.
typecheck:
    uv run basedpyright cross_ai.py scripts

# Scan for dead code.
dead-code:
    uv run vulture

# Build both package artifacts.
package-check:
    uv build

# Run the static quality gate.
check: fmt-check lint lock-check typecheck compile dead-code package-check

# Run focused tests.
unit:
    uv run pytest -q

# Print a non-blocking diagnostic coverage report.
coverage:
    uv run pytest --cov=cross_ai --cov-report=term-missing

# Print the highest CRAP scores for diagnosis.
crap:
    uv run pytest --cov=cross_ai --cov-report=term-missing --crap --crap-threshold=30 --crap-top-n=30

# Fail if any function exceeds CRAP 30.
crap-check:
    coverage_file="$(mktemp /tmp/cross-ai-crap-coverage.XXXXXX.json)"; \
    trap 'rm -f "$coverage_file"' EXIT; \
    uv run pytest --cov=cross_ai --cov-report=json:"$coverage_file"; \
    uv run python -m scripts.crap_gate --coverage "$coverage_file" --src cross_ai.py --threshold 30

# Audit the locked dependency set for known vulnerabilities.
deps-audit:
    #!/usr/bin/env bash
    set -euo pipefail
    requirements="$(mktemp /tmp/cross-ai-audit.XXXXXX.txt)"
    trap 'rm -f "$requirements"' EXIT
    uv export --locked --all-groups --no-emit-project --no-header --no-annotate > "$requirements"
    uv run pip-audit -r "$requirements" --strict --no-deps --disable-pip

# Full local quality contract.
verify: check crap-check unit deps-audit
