"""Enforce a per-function CRAP score limit from coverage.py JSON and Radon."""

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, TypeGuard, cast

from radon.complexity import cc_visit

DEFAULT_THRESHOLD = 30.0
DEFAULT_SOURCE = Path("cross_ai.py")


class _FunctionSummary(TypedDict):
    covered_lines: int
    num_statements: int


class _CoverageFunctionEntry(TypedDict):
    summary: _FunctionSummary


class _CoverageFileEntry(TypedDict):
    functions: dict[str, _CoverageFunctionEntry]


class _CoverageReport(TypedDict):
    files: dict[str, _CoverageFileEntry]


class _RadonBlock(Protocol):
    name: str
    lineno: int
    complexity: int


class _RadonFunctionBlock(_RadonBlock, Protocol):
    closures: Sequence["_RadonFunctionBlock"]


class _RadonClassBlock(_RadonBlock, Protocol):
    inner_classes: Sequence["_RadonClassBlock"]
    methods: Sequence[_RadonFunctionBlock]


class _GateArgs(Protocol):
    coverage: Path
    src: Path
    threshold: float


@dataclass(frozen=True, slots=True)
class FunctionMetric:
    key: str
    start_line: int
    complexity: int
    coverage_fraction: float
    crap: float


RadonVisit = Callable[[str], Sequence[_RadonBlock]]
_CC_VISIT = cast(RadonVisit, cc_visit)


def _is_class_block(block: _RadonBlock) -> TypeGuard[_RadonClassBlock]:
    return hasattr(block, "methods") and hasattr(block, "inner_classes")


def _qualified_blocks(block: _RadonBlock, prefix: tuple[str, ...] = ()) -> list[tuple[str, _RadonBlock]]:
    if _is_class_block(block):
        class_prefix = (*prefix, block.name)
        return [
            item
            for member in (*block.inner_classes, *block.methods)
            for item in _qualified_blocks(member, class_prefix)
        ]

    function = cast(_RadonFunctionBlock, block)
    name = ".".join((*prefix, function.name))
    return [
        (name, function),
        *[item for child in function.closures for item in _qualified_blocks(child, (*prefix, function.name))],
    ]


def _load_report(path: Path) -> _CoverageReport:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("files"), dict):
        raise ValueError("coverage report does not contain an object at 'files'")
    return cast(_CoverageReport, raw)


def _metrics(report: _CoverageReport, source: Path) -> list[FunctionMetric]:
    source = source.resolve()
    file_entry = next((data for name, data in report["files"].items() if Path(name).resolve() == source), None)
    if file_entry is None:
        raise ValueError(f"coverage report does not contain {source}")

    functions = file_entry["functions"]
    metrics: list[FunctionMetric] = []
    for qualname, block in [
        item for root in _CC_VISIT(source.read_text(encoding="utf-8")) for item in _qualified_blocks(root)
    ]:
        if qualname not in functions:
            continue
        summary = functions[qualname]["summary"]
        statements = summary["num_statements"]
        coverage = 1.0 if statements <= 0 else summary["covered_lines"] / statements
        score = block.complexity**2 * (1 - coverage) ** 3 + block.complexity
        metrics.append(FunctionMetric(qualname, block.lineno, block.complexity, coverage, score))
    return metrics


def _parse_args(argv: list[str] | None) -> _GateArgs:
    parser = argparse.ArgumentParser(description="Fail when a function exceeds the CRAP threshold.")
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--src", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return cast(_GateArgs, parser.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    metrics = _metrics(_load_report(args.coverage), args.src)
    offenders = sorted((metric for metric in metrics if metric.crap > args.threshold), key=lambda item: -item.crap)
    if offenders:
        print(f"CRAP gate failed: {len(offenders)} function(s) exceed {args.threshold:.2f}")
        for metric in offenders[:20]:
            print(
                f"  {args.src}::{metric.key}:{metric.start_line} CRAP {metric.crap:.2f}, "
                f"complexity {metric.complexity}, coverage {metric.coverage_fraction:.1%}"
            )
        return 1
    print(f"CRAP gate passed: {len(metrics)} function(s), threshold {args.threshold:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
