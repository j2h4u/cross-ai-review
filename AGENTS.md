# Agent guidance

`cross_ai.py` in this repository is the source of truth for the installed
Cross-AI command.

Treat configured repositories and reviewer models as trusted. OpenCode
permissions are intentionally a behavioral and resource boundary: allow broad
read/search access inside `--repo-root` and read-only Git inspection, while
denying edits, tests, builds, general shell use, web access, and delegation. Do
not reinterpret this policy as a hostile-code sandbox without an explicit
change to the product's trust model.

Keep temporary files scoped to a per-run directory under `/tmp`. Cleanup must
remain ownership-checked and cover normal completion, errors, timeouts, handled
termination signals, and partial process startup. Preserve Markdown reports in
the selected output directory.

Use focused lifecycle tests while developing. Run broader external-AI review
only after a coherent code-complete change.
