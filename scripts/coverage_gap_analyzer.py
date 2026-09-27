"""Analyze a coverage.py JSON report and map uncovered lines to functions.

Usage:
    python scripts/coverage_gap_analyzer.py coverage.json [--root .]

Prints, per source file:
  * fully-uncovered functions (no executed statement at all)
  * partially-covered functions with their uncovered line numbers
and a final summary of the biggest functional gaps.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import defaultdict


def _qualname(node: ast.AST, parents: list[str]) -> str:
    name = getattr(node, "name", "<lambda>")
    return ".".join(parents + [name])


class FuncCollector(ast.NodeVisitor):
    """Collect (def_line, body_start, end_line, qualname) for every function.

    ``body_start`` is the first line of the function body (excluding the
    ``def`` header and decorators), because the header line counts as executed
    as soon as the module is imported -- which would otherwise mask a function
    that is never actually called.
    """

    def __init__(self) -> None:
        self.funcs: list[tuple[int, int, int, str]] = []
        self._stack: list[str] = []

    def _visit_func(self, node) -> None:
        qn = _qualname(node, self._stack)
        end = max(node.lineno, node.end_lineno or node.lineno)
        body_start = min(
            (child.lineno for child in node.body),
            default=node.lineno,
        )
        self.funcs.append((node.lineno, body_start, end, qn))
        self._stack.append(getattr(node, "name", "<lambda>"))
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_ClassDef(self, node) -> None:  # noqa: N802
        # Treat classes as containers, not reportable functions, but keep the
        # name in the qualified path.
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()


def executed_lines(coverage_file: dict) -> set[int]:
    """Lines that ran at least once (statement lines only)."""
    return set(coverage_file.get("executed_lines", []))


def analyze(coverage_json: str, root: str) -> dict:
    with open(coverage_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    results: dict[str, dict] = {}

    for path, info in data["files"].items():
        abs_path = os.path.join(root, path)
        if not os.path.exists(abs_path):
            continue
        with open(abs_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        collector = FuncCollector()
        collector.visit(tree)

        ran = executed_lines(info)
        missing = set(info.get("missing_lines", []))

        full: list[tuple[str, int, int]] = []
        partial: list[tuple[str, int, int, list[int]]] = []

        for def_line, body_start, end, qn in collector.funcs:
            body_lines = set(range(body_start, end + 1))
            executed_here = body_lines & ran
            missing_here = sorted(body_lines & missing)
            if not missing_here:
                continue
            if not executed_here:
                full.append((qn, def_line, end))
            else:
                partial.append((qn, def_line, end, missing_here))

        # Sort: biggest fully-uncovered function first
        full.sort(key=lambda t: t[1])
        partial.sort(key=lambda t: t[1])

        results[path] = {
            "summary": info["summary"],
            "fully_uncovered": full,
            "partial": partial,
            "missing_lines": sorted(missing),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--only-full", action="store_true")
    args = parser.parse_args()

    results = analyze(args.coverage_json, args.root)

    ordered = sorted(
        results.items(),
        key=lambda kv: kv[1]["summary"]["percent_covered"],
    )

    total_full = 0
    for path, info in ordered:
        pct = info["summary"]["percent_covered"]
        full = info["fully_uncovered"]
        partial = info["partial"]
        total_full += len(full)
        if not full and not partial:
            continue
        print("=" * 78)
        print(f"{path}  [{pct:.1f}%]  完全未覆盖函数 {len(full)} 个 / 部分覆盖 {len(partial)} 个")
        if full:
            print("  -- 完全未执行 --")
            for qn, start, end in full:
                print(f"     L{start:<5}-{end:<5} {qn}")
        if partial and not args.only_full:
            print("  -- 部分执行（缺失行） --")
            for qn, start, end, lines in partial[:12]:
                shown = ", ".join(str(x) for x in lines[:14])
                more = " ..." if len(lines) > 14 else ""
                print(f"     L{start:<5}-{end:<5} {qn}  缺失: {shown}{more}")
            if len(partial) > 12:
                print(f"     ... 另有 {len(partial) - 12} 个部分覆盖函数")

    print("=" * 78)
    print(f"完全未覆盖函数总数: {total_full}")


if __name__ == "__main__":
    main()
