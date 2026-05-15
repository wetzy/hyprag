"""
hyprag.chunker
~~~~~~~~~~~~~~
Parse Python source files into a hierarchy of Chunk objects.

Hierarchy levels
----------------
    depth 0 — module          one per file; text = docstring or header
    depth 1 — class           full body of a class definition
               top-level fn   functions defined at module scope
    depth 2 — method          functions defined inside a class
               nested fn      functions defined inside a top-level fn

Each Chunk carries a ``node_path`` (e.g. ``"mymod.Parser.chunk_file"``) and a
``depth`` tag. The retriever uses ``node_path``/``parent_path`` for subtree
expansion after the initial FAISS lookup; ``depth`` is a convenience for
downstream filtering and reporting.

Chunk text format (retrieval-optimised)
----------------------------------------
Every chunk is emitted as:

    {node_path}
    {signature}

    {docstring}

This puts the fully-qualified name and natural-language description at the
top of the text where encoders weight it highest, and avoids drowning the
semantic signal in noisy function bodies.  For nodes with no docstring the
first few lines of the body are included as fallback.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Chunk", "HierarchicalChunker"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single node extracted from the AST hierarchy of a Python codebase."""

    id: int
    text: str
    depth: int
    node_path: str
    source_file: str
    start_line: int
    end_line: int

    @property
    def parent_path(self) -> str:
        if "." not in self.node_path:
            return ""
        return self.node_path.rsplit(".", 1)[0]

    @property
    def name(self) -> str:
        return self.node_path.rsplit(".", 1)[-1]

    def __repr__(self) -> str:
        return (
            f"Chunk(depth={self.depth}, path={self.node_path!r}, "
            f"lines={self.start_line}-{self.end_line})"
        )


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class HierarchicalChunker:
    """
    Parse Python source into hierarchical Chunk objects.

    Parameters
    ----------
    include_source : bool
        When *True* (default) each chunk's text leads with the node_path +
        signature + docstring, followed by a short excerpt of the source body.
        When *False* only the node_path + signature + docstring are included —
        useful when the encoder is optimised for short natural-language text.
    max_depth : int
        Maximum hierarchy depth to emit (default 2).
    """

    def __init__(
        self,
        *,
        include_source: bool = True,
        max_depth: int = 2,
    ) -> None:
        self.include_source = include_source
        self.max_depth = max_depth

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_file(self, path: str | Path) -> list[Chunk]:
        """Return all Chunks produced from a single .py file."""
        path = Path(path).resolve()
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return [_error_chunk(str(path), f"OSError: {exc}")]
        return self._parse(source, module_name=path.stem, file_path=str(path))

    @staticmethod
    def _qualified_name(path: Path, root: Path) -> str:
        """
        Compute a fully-qualified dotted module name from a file path.

            json/decoder.py          ->  json.decoder
            json/__init__.py         ->  json
            xml/etree/ElementTree.py ->  xml.etree.ElementTree
        """
        rel = path.resolve().relative_to(root.resolve())
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts) if parts else path.stem

    def chunk_directory(self, root: str | Path) -> list[Chunk]:
        """
        Recursively chunk every .py file found under *root*.

        Module names are fully-qualified relative to *root*:
            json/decoder.py  ->  json.decoder
            json/__init__.py ->  json
        """
        root = Path(root).resolve()
        all_chunks: list[Chunk] = []
        id_offset = 0
        for py_file in sorted(root.rglob("*.py")):
            module_name = self._qualified_name(py_file, root)
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                file_chunks = [_error_chunk(str(py_file), f"OSError: {exc}", module_name)]
            else:
                file_chunks = self._parse(source, module_name=module_name,
                                          file_path=str(py_file))
            for c in file_chunks:
                c.id += id_offset
            id_offset += len(file_chunks)
            all_chunks.extend(file_chunks)
        return all_chunks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse(self, source: str, module_name: str, file_path: str) -> list[Chunk]:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return [_error_chunk(file_path, f"SyntaxError: {exc}", module_name)]

        lines = source.splitlines()
        visitor = _HierarchyVisitor(
            lines=lines,
            module_name=module_name,
            file_path=file_path,
            include_source=self.include_source,
            max_depth=self.max_depth,
        )
        visitor.visit(tree)
        return visitor.chunks


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------

class _HierarchyVisitor(ast.NodeVisitor):
    """Depth-first AST walker that emits one Chunk per significant node."""

    def __init__(
        self,
        lines: list[str],
        module_name: str,
        file_path: str,
        include_source: bool,
        max_depth: int,
    ) -> None:
        self._lines = lines
        self._file_path = file_path
        self._include_source = include_source
        self._max_depth = max_depth

        self._path_stack: list[str] = [module_name]
        self._depth_stack: list[int] = [0]

        self.chunks: list[Chunk] = []
        self._emit_module_chunk()

    # -- Stack helpers ---------------------------------------------------

    @property
    def _current_path(self) -> str:
        return ".".join(self._path_stack)

    @property
    def _current_depth(self) -> int:
        return self._depth_stack[-1]

    def _next_id(self) -> int:
        return len(self.chunks)

    # -- Text extraction -------------------------------------------------

    def _signature(self, node: ast.AST) -> str:
        """
        Extract the def/class signature line(s) up to the colon.
        Handles multi-line signatures (long parameter lists).
        """
        if not hasattr(node, "lineno"):
            return ""
        sig_lines: list[str] = []
        for i in range(node.lineno - 1,
                       min(node.lineno + 8, node.end_lineno)):
            line = self._lines[i].rstrip()
            sig_lines.append(line)
            stripped = line.strip()
            if stripped.endswith(":") or stripped.endswith("->"):
                # Arrow without colon — keep going one more line
                if stripped.endswith("->"):
                    continue
                break
        return textwrap.dedent("\n".join(sig_lines)).strip()

    def _rich_text(self, node: ast.AST, node_path: str) -> str:
        """
        Build retrieval-optimised chunk text.

        Layout
        ------
            {node_path}
            {signature}

            {docstring}

            [{first few body lines, if include_source and no docstring}]

        Putting the node_path first ensures that queries containing module
        or class names ("asyncio", "DictWriter") get lexical signal even
        when the semantic encoder struggles with code.  The docstring
        provides the natural-language bridge.  The body is included only
        as a fallback when no docstring exists, and capped at 300 chars to
        prevent noise from dominating the embedding.
        """
        parts: list[str] = []

        # 1. Fully-qualified name  (lexical anchor)
        parts.append(node_path)

        # 2. Signature  (parameter names carry semantic meaning)
        sig = self._signature(node)
        if sig:
            parts.append(sig)

        # 3. Docstring  (highest natural-language signal)
        doc = ast.get_docstring(node) or ""
        if doc:
            # Truncate very long docstrings; first 400 chars carry the meaning
            parts.append(doc[:400].strip())

        # 4. Body excerpt  (only when no docstring — avoids noise)
        if not doc and self._include_source:
            # Skip the def/class line and any docstring line
            body_start = node.lineno  # 0-indexed = lineno - 1 + 1
            body_lines = self._lines[body_start:
                                     min(body_start + 8, node.end_lineno)]
            body = textwrap.dedent("\n".join(body_lines)).strip()
            # Strip any string literal that might be an undiscovered docstring
            if body and not body.startswith(('"""', "'''", '"', "'")):
                parts.append(body[:300])

        return "\n\n".join(p for p in parts if p.strip())

    # -- Module chunk ----------------------------------------------------

    def _emit_module_chunk(self) -> None:
        module_path = self._path_stack[0]
        # Find tree node for the module to get its docstring
        doc_lines: list[str] = []
        for line in self._lines[:5]:
            stripped = line.strip()
            if stripped.startswith(('"""', "'''", "#")):
                doc_lines.append(line)
            elif stripped:
                break

        # Build: module_path + docstring / header
        header_text = "\n".join(doc_lines).strip()
        if not header_text:
            # Fallback: first meaningful non-import line
            collected: list[str] = []
            for line in self._lines[:30]:
                collected.append(line)
                s = line.strip()
                if s and not s.startswith(
                    ("#", "import ", "from ", '"""', "'''", '"', "'")
                ):
                    break
            header_text = "\n".join(collected).strip()

        text = f"{module_path}\n\n{header_text}" if header_text else module_path

        self.chunks.append(Chunk(
            id=self._next_id(),
            text=text,
            depth=0,
            node_path=module_path,
            source_file=self._file_path,
            start_line=1,
            end_line=len(self._lines),
        ))

    # -- Visitors --------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        child_depth = self._current_depth + 1
        child_path = f"{self._current_path}.{node.name}"

        if child_depth <= self._max_depth:
            self.chunks.append(Chunk(
                id=self._next_id(),
                text=self._rich_text(node, child_path),
                depth=child_depth,
                node_path=child_path,
                source_file=self._file_path,
                start_line=node.lineno,
                end_line=node.end_lineno,
            ))

        self._path_stack.append(node.name)
        self._depth_stack.append(child_depth)
        self.generic_visit(node)
        self._path_stack.pop()
        self._depth_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        child_depth = self._current_depth + 1
        child_path = f"{self._current_path}.{node.name}"

        if child_depth <= self._max_depth:
            self.chunks.append(Chunk(
                id=self._next_id(),
                text=self._rich_text(node, child_path),
                depth=child_depth,
                node_path=child_path,
                source_file=self._file_path,
                start_line=node.lineno,
                end_line=node.end_lineno,
            ))

    visit_AsyncFunctionDef = visit_FunctionDef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_chunk(file_path: str, message: str, module_name: str = "") -> Chunk:
    stem = module_name or Path(file_path).stem
    return Chunk(
        id=0,
        text=f"[{message}]",
        depth=0,
        node_path=stem,
        source_file=file_path,
        start_line=1,
        end_line=1,
    )
