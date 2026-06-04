#!/usr/bin/env python3
"""Refresh the repository function ontology after code changes."""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path


ONTOLOGY = "ontology.md"
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".ipynb_checkpoints",
    "data",
    "flat_with_kv_mm_filenames_cropped",
}
SKIP_PREFIXES = {
    ("scripts", "hooks"),
}


@dataclass(frozen=True)
class FunctionEntry:
    path: str
    line: int
    qualname: str
    signature: str
    purpose: str


def repo_root() -> Path:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return Path(output.strip())
    except subprocess.CalledProcessError:
        return Path.cwd()


def iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        if any(rel_parts[: len(prefix)] == prefix for prefix in SKIP_PREFIXES):
            continue
        files.append(path)
    return sorted(files)


def annotation_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    return ast.unparse(node)


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args: list[str] = []
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults = [""] * (len(positional) - len(node.args.defaults)) + [
        ast.unparse(default) for default in node.args.defaults
    ]

    for arg, default in zip(positional, defaults):
        value = arg.arg
        annotation = annotation_text(arg.annotation)
        if annotation:
            value += f": {annotation}"
        if default:
            value += f" = {default}"
        args.append(value)

    if node.args.vararg:
        value = f"*{node.args.vararg.arg}"
        annotation = annotation_text(node.args.vararg.annotation)
        if annotation:
            value += f": {annotation}"
        args.append(value)
    elif node.args.kwonlyargs:
        args.append("*")

    for arg, default_node in zip(node.args.kwonlyargs, node.args.kw_defaults):
        value = arg.arg
        annotation = annotation_text(arg.annotation)
        if annotation:
            value += f": {annotation}"
        if default_node is not None:
            value += f" = {ast.unparse(default_node)}"
        args.append(value)

    if node.args.kwarg:
        value = f"**{node.args.kwarg.arg}"
        annotation = annotation_text(node.args.kwarg.annotation)
        if annotation:
            value += f": {annotation}"
        args.append(value)

    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    result = f"{prefix}{node.name}({', '.join(args)})"
    returns = annotation_text(node.returns)
    if returns:
        result += f" -> {returns}"
    return result


def doc_summary(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    doc = ast.get_docstring(node)
    if not doc:
        return "No docstring yet; inspect implementation before reuse."
    return " ".join(doc.strip().split())[:220]


class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.scope: list[str] = []
        self.entries: list[FunctionEntry] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join([*self.scope, node.name])
        self.entries.append(
            FunctionEntry(
                path=self.rel_path,
                line=node.lineno,
                qualname=qualname,
                signature=function_signature(node),
                purpose=doc_summary(node),
            )
        )


def parse_functions(root: Path) -> list[FunctionEntry]:
    entries: list[FunctionEntry] = []
    for path in iter_python_files(root):
        rel_path = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)
        except SyntaxError:
            continue
        visitor = FunctionVisitor(rel_path)
        visitor.visit(tree)
        entries.extend(visitor.entries)
    return sorted(entries, key=lambda item: (item.path, item.line, item.qualname))


def render_index(entries: list[FunctionEntry]) -> str:
    if not entries:
        return (
            "## Function Index\n\n"
            "No Python functions are currently indexed. Add Python modules under "
            "`src/`, `tests/`, or another repository code directory, then run the "
            "post-hook to populate this section.\n"
        )

    lines = ["## Function Index", ""]
    current_path = ""
    for entry in entries:
        if entry.path != current_path:
            current_path = entry.path
            lines.extend([f"### `{current_path}`", ""])
        lines.append(f"- `{entry.qualname}` at line {entry.line}")
        lines.append(f"  - Signature: `{entry.signature}`")
        lines.append(f"  - Purpose: {entry.purpose}")
    return "\n".join(lines).rstrip() + "\n"


def update_ontology(root: Path, entries: list[FunctionEntry]) -> None:
    ontology_path = root / ONTOLOGY
    if ontology_path.exists():
        text = ontology_path.read_text(encoding="utf-8")
    else:
        text = "# Function Ontology\n\n"

    marker = "## Function Index"
    generated = render_index(entries)
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n\n" + generated
    else:
        text = text.rstrip() + "\n\n" + generated

    ontology_path.write_text(text, encoding="utf-8")


def main() -> int:
    root = repo_root()
    entries = parse_functions(root)
    update_ontology(root, entries)
    print(f"Updated {ONTOLOGY} with {len(entries)} function(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
