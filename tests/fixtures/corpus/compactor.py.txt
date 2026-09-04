"""
Traceback and code compaction for ContextOS.

Two independent compactors:

  * `TracebackCompactor` strips framework frames from Python, Node.js, Rust and
    generic error output while preserving the user call sites and the root cause.
  * `CodeOutlineCompactor` folds a source file down to its structural skeleton,
    optionally leaving one symbol fully expanded.

Both are conservative: if compaction does not actually shrink the input, the
original text is returned unchanged and the reported saving is zero. Neither
compactor is ever allowed to report a saving it did not achieve.
"""

from __future__ import annotations

import ast
import re
from typing import List, Optional, Tuple

from contextos.core.tokens import estimate_tokens

# A Python traceback header, or an exception line like "module.Klass: message".
_PY_HEADER = re.compile(r"^Traceback \(most recent call last\):", re.M)
_PY_FRAME = re.compile(r"""^\s*File ["'][^"']+["'], line \d+""", re.M)
_PY_EXCEPTION = re.compile(
    r"^(?:[A-Za-z_][\w.]*\.)?[A-Z]\w*(?:Error|Exception|Warning|Interrupt|Exit)\b\s*:",
    re.M,
)

# Node/V8: "    at fn (/path/file.js:12:5)" or "    at /path/file.js:12:5"
_NODE_FRAME = re.compile(r"^\s*at\s+(?:.+\s+)?\(?[^\s()]+:\d+:\d+\)?\s*$", re.M)

# Rust: "thread 'main' panicked at src/x.rs:1:1:"
_RUST_PANIC = re.compile(r"^thread '.*' panicked at ", re.M)

# Go: "panic: ..." followed by "goroutine N [running]:"
_GO_PANIC = re.compile(r"^panic:.*$\n(?:.*\n)*?goroutine \d+ \[", re.M)

# Test-runner failure summaries.
_TEST_FAILURE = re.compile(
    r"^(?:FAILED |FAIL\b|E\s{3}|\s*✕|\s*✗|AssertionError\b|=+ FAILURES =+)", re.M
)

_LIBRARY_MARKERS = (
    "site-packages",
    "dist-packages",
    "/lib/python",
    "\\lib\\python",
    "node_modules",
    "internal/",
    "node:internal",
    "/usr/lib/",
    "runtime/",
)


class TracebackCompactor:
    """Compresses error output down to user frames plus root cause."""

    @staticmethod
    def is_traceback(text: str) -> bool:
        """
        True only when `text` genuinely looks like an error trace.

        Deliberately strict. An earlier version matched the bare substrings
        "Error:" and "at ", which classified ordinary prose and chat messages as
        tracebacks and then silently discarded most of their content. Detection
        now requires a structural marker: a traceback header, a stack frame in a
        known format, a panic banner, or a test-runner failure line.
        """
        if not text or len(text) < 16:
            return False

        return bool(
            _PY_HEADER.search(text)
            or _PY_FRAME.search(text)
            or _NODE_FRAME.search(text)
            or _RUST_PANIC.search(text)
            or _GO_PANIC.search(text)
            or _TEST_FAILURE.search(text)
            or (_PY_EXCEPTION.search(text) and "\n" in text)
        )

    @classmethod
    def detect_language(cls, text: str) -> str:
        """Identify which trace dialect `text` is, for reporting and routing."""
        if _PY_HEADER.search(text) or _PY_FRAME.search(text):
            return "python"
        if _RUST_PANIC.search(text):
            return "rust"
        if _GO_PANIC.search(text):
            return "go"
        if _NODE_FRAME.search(text):
            return "node"
        if _TEST_FAILURE.search(text):
            return "test"
        return "generic"

    @classmethod
    def compact(cls, text: str) -> Tuple[str, int, int]:
        """
        Compact an error trace.

        Returns `(compacted_text, original_tokens, compacted_tokens)`. If the
        compacted form is not smaller, the original text is returned with equal
        token counts so callers can never book a phantom saving.
        """
        orig_tokens = estimate_tokens(text)
        if not text.strip():
            return text, orig_tokens, orig_tokens

        lines = text.splitlines()
        language = cls.detect_language(text)

        if language == "python":
            compacted = cls._compact_python_traceback(lines)
        elif language == "node":
            compacted = cls._compact_node_traceback(lines)
        elif language == "rust":
            compacted = cls._compact_rust_panic(lines)
        elif language == "go":
            compacted = cls._compact_go_panic(lines)
        else:
            compacted = cls._compact_generic_error(lines)

        comp_tokens = estimate_tokens(compacted)
        if comp_tokens >= orig_tokens:
            return text, orig_tokens, orig_tokens
        return compacted, orig_tokens, comp_tokens

    # -- per-language strategies ------------------------------------------------

    @staticmethod
    def _is_library_frame(line: str) -> bool:
        lowered = line.replace("\\", "/").lower()
        return any(marker.replace("\\", "/").lower() in lowered for marker in _LIBRARY_MARKERS)

    @classmethod
    def _compact_python_traceback(cls, lines: List[str]) -> str:
        """Keep the last user frames and the exception line; drop library frames."""
        result = ["[ContextOS] Python traceback reduced to user frames + root cause"]

        user_frames: List[str] = []
        library_frame_count = 0
        root_cause = ""
        chained = False

        index = 0
        while index < len(lines):
            line = lines[index]
            stripped = line.strip()

            if stripped.startswith("During handling of") or stripped.startswith(
                "The above exception was the direct cause"
            ):
                chained = True
                index += 1
                continue

            if _PY_FRAME.match(line):
                frame = [line]
                # The source line belonging to this frame, if present.
                if index + 1 < len(lines):
                    following = lines[index + 1].strip()
                    if following and not following.startswith("File "):
                        frame.append(lines[index + 1])
                        index += 1

                if cls._is_library_frame(line):
                    library_frame_count += 1
                else:
                    user_frames.append("\n".join(frame))
            elif _PY_EXCEPTION.match(line) or (
                stripped and re.match(r"^\w+(?:\.\w+)*\s*:", stripped) and index == len(lines) - 1
            ):
                root_cause = stripped
            index += 1

        if user_frames:
            result.append(f"User frames ({len(user_frames)} kept, {library_frame_count} library frames dropped):")
            result.extend(user_frames[-3:])
        elif library_frame_count:
            result.append(f"({library_frame_count} library frames dropped; no user frames in trace)")

        if not root_cause:
            for line in reversed(lines):
                if line.strip():
                    root_cause = line.strip()
                    break

        if chained:
            result.append("Note: exception chain collapsed to the final cause.")
        result.append(f"Root cause: {root_cause}")
        return "\n".join(result)

    @classmethod
    def _compact_node_traceback(cls, lines: List[str]) -> str:
        """Drop node_modules and node-internal frames from a V8 stack."""
        message_lines = []
        for line in lines:
            if _NODE_FRAME.match(line):
                break
            if line.strip():
                message_lines.append(line.strip())

        user_frames = []
        dropped = 0
        for line in lines:
            if not _NODE_FRAME.match(line):
                continue
            if cls._is_library_frame(line):
                dropped += 1
            else:
                user_frames.append(line.strip())

        result = ["[ContextOS] Node.js stack reduced to application frames"]
        message = " ".join(message_lines) if message_lines else "unknown error"
        result.append(message if ":" in message else f"Error: {message}")
        if user_frames:
            result.append(f"Application frames ({len(user_frames)} kept, {dropped} runtime frames dropped):")
            result.extend(user_frames[:3])
        elif dropped:
            result.append(f"({dropped} runtime frames dropped; no application frames in stack)")
        return "\n".join(result)

    @classmethod
    def _compact_rust_panic(cls, lines: List[str]) -> str:
        """Keep the panic site and message; drop the std/core backtrace."""
        panic_site = ""
        message: List[str] = []
        user_frames: List[str] = []
        in_backtrace = False

        for line in lines:
            stripped = line.strip()
            if _RUST_PANIC.match(line):
                panic_site = stripped
                continue
            if stripped.startswith("stack backtrace:") or stripped.startswith("note: run with"):
                in_backtrace = True
                continue
            if in_backtrace:
                frame = re.sub(r"^\s*\d+:\s*", "", stripped)
                if frame and not frame.startswith(("std::", "core::", "alloc::", "__", "rust_begin")):
                    user_frames.append(frame)
            elif stripped:
                message.append(stripped)

        result = ["[ContextOS] Rust panic reduced to panic site + crate frames"]
        if panic_site:
            result.append(panic_site)
        if message:
            result.append("Message: " + " | ".join(message[:3]))
        if user_frames:
            result.append("Crate frames: " + ", ".join(user_frames[:3]))
        return "\n".join(result)

    @classmethod
    def _compact_go_panic(cls, lines: List[str]) -> str:
        """Keep the panic message and the first non-runtime goroutine frames."""
        message = next((l.strip() for l in lines if l.startswith("panic:")), "panic")
        user_frames = [
            l.strip()
            for l in lines
            if "/" in l and not cls._is_library_frame(l) and not l.strip().startswith("goroutine")
        ]
        result = ["[ContextOS] Go panic reduced to message + application frames", message]
        if user_frames:
            result.append("Application frames: " + ", ".join(user_frames[:3]))
        return "\n".join(result)

    @classmethod
    def _compact_generic_error(cls, lines: List[str]) -> str:
        """
        Last-resort compaction for output with no recognised stack format.

        Keeps lines that carry an error signal. If none are found, keeps a head
        and tail window rather than silently discarding the tail.
        """
        signal = re.compile(r"(error|fail|assert|panic|fatal|exception|traceback)", re.I)
        error_lines = [l.strip() for l in lines if l.strip() and signal.search(l)]

        if error_lines:
            kept = error_lines[:8]
            omitted = len(error_lines) - len(kept)
            result = ["[ContextOS] Key error signals extracted"]
            result.extend(kept)
            if omitted > 0:
                result.append(f"... {omitted} further error lines omitted")
            return "\n".join(result)

        head, tail = 4, 2
        if len(lines) <= head + tail:
            return "\n".join(lines)
        omitted = len(lines) - head - tail
        return "\n".join(
            lines[:head] + [f"... [ContextOS] {omitted} lines omitted ..."] + lines[-tail:]
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Kept for backwards compatibility; delegates to `core.tokens`."""
        return estimate_tokens(text)


class CodeOutlineCompactor:
    """
    Folds a source file into a structural skeleton.

    Python is parsed with the `ast` module, so signatures, decorators and
    docstrings are identified exactly rather than guessed. Other languages fall
    back to an indentation-aware line scanner. Both paths support `focus_symbol`,
    which leaves one function or class fully expanded while everything else is
    folded to a signature plus a line count.
    """

    @classmethod
    def compact_code(
        cls, filename: str, code: str, focus_symbol: str = ""
    ) -> Tuple[str, int, int]:
        """
        Returns `(outline, original_tokens, outline_tokens)`.

        If folding does not shrink the file (very small files, or a focus symbol
        that spans nearly the whole file) the original source is returned with
        equal token counts.
        """
        orig_tokens = estimate_tokens(code)
        if not code.strip():
            return code, orig_tokens, orig_tokens

        outline: Optional[str] = None
        if filename.endswith(".py") or filename.endswith(".pyi"):
            outline = cls._outline_python_ast(filename, code, focus_symbol)
        if outline is None:
            outline = cls._outline_by_indent(filename, code, focus_symbol)

        comp_tokens = estimate_tokens(outline)
        if comp_tokens >= orig_tokens:
            return code, orig_tokens, orig_tokens
        return outline, orig_tokens, comp_tokens

    # -- Python, via ast --------------------------------------------------------

    @classmethod
    def _outline_python_ast(
        cls, filename: str, code: str, focus_symbol: str
    ) -> Optional[str]:
        """Build an outline using the real parse tree. None if the file won't parse."""
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError):
            return None

        lines = code.splitlines()
        out: List[str] = [f"# [ContextOS outline] {filename} ({len(lines)} lines)"]

        module_doc = ast.get_docstring(tree)
        if module_doc:
            out.append(f'"""{module_doc.strip().splitlines()[0]}"""')

        def render(node: ast.AST, depth: int) -> None:
            indent = "    " * depth

            if isinstance(node, (ast.Import, ast.ImportFrom)) and depth == 0:
                segment = ast.get_source_segment(code, node)
                if segment:
                    out.append(segment)
                return

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                is_focus = bool(focus_symbol) and node.name == focus_symbol

                if is_focus:
                    segment = ast.get_source_segment(code, node)
                    if segment:
                        out.append(f"{indent}# --- focus: {node.name} (expanded) ---")
                        out.append(segment)
                        return

                for decorator in node.decorator_list:
                    decorator_src = ast.get_source_segment(code, decorator)
                    if decorator_src:
                        out.append(f"{indent}@{decorator_src}")

                out.append(f"{indent}{cls._signature(node, code)}")

                doc = ast.get_docstring(node)
                if doc:
                    first = doc.strip().splitlines()[0]
                    out.append(f'{indent}    """{first}"""')

                if isinstance(node, ast.ClassDef):
                    children = [
                        child
                        for child in node.body
                        if isinstance(
                            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                        )
                    ]
                    if children:
                        for child in children:
                            render(child, depth + 1)
                    else:
                        out.append(f"{indent}    ...")
                else:
                    body_lines = cls._node_line_span(node)
                    plural = "line" if body_lines == 1 else "lines"
                    out.append(f"{indent}    ...  # {body_lines} {plural} folded")
                return

        for node in tree.body:
            render(node, 0)

        return "\n".join(out)

    @staticmethod
    def _node_line_span(node: ast.AST) -> int:
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start) or start
        return max(0, end - start)

    @classmethod
    def _signature(cls, node: ast.AST, code: str) -> str:
        """Render a def/class header line without its body."""
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                segment = ast.get_source_segment(code, base)
                if segment:
                    bases.append(segment)
            for keyword in node.keywords:
                segment = ast.get_source_segment(code, keyword)
                if segment:
                    bases.append(segment)
            suffix = f"({', '.join(bases)})" if bases else ""
            return f"class {node.name}{suffix}:"

        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        try:
            args = ast.unparse(node.args)
        except Exception:
            args = "..."
        returns = ""
        if getattr(node, "returns", None) is not None:
            try:
                returns = f" -> {ast.unparse(node.returns)}"
            except Exception:
                returns = ""
        return f"{prefix} {node.name}({args}){returns}:"

    # -- Everything else, by indentation ---------------------------------------

    @classmethod
    def _outline_by_indent(cls, filename: str, code: str, focus_symbol: str) -> str:
        """
        Indentation-aware fallback for non-Python sources and unparseable Python.

        A focus block is the definition line plus every following line that is
        blank or indented deeper than the definition. The previous implementation
        used `not line.startswith(" ")` as the exit test, which never fired for a
        sibling method inside a class, so focus mode never ended and nothing was
        folded at all.
        """
        lines = code.splitlines()
        out: List[str] = [f"# [ContextOS outline] {filename} ({len(lines)} lines)"]

        definition = re.compile(
            r"^\s*(?:export\s+)?(?:public\s+|private\s+|protected\s+|static\s+|final\s+)*"
            r"(?:async\s+)?(?:def|class|function|fn|func|impl|interface|struct|enum|type|const\s+\w+\s*=\s*\()"
            r"\b"
        )
        keep = re.compile(r"^\s*(?:import |from |#include|package |use |require\()")

        in_focus = False
        focus_indent = 0
        folded_since_def = 0
        last_def_index: Optional[int] = None

        def flush_fold() -> None:
            nonlocal folded_since_def, last_def_index
            if last_def_index is not None and folded_since_def > 0:
                plural = "line" if folded_since_def == 1 else "lines"
                out[last_def_index] += f"  # ... {folded_since_def} {plural} folded"
            folded_since_def = 0
            last_def_index = None

        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            is_definition = bool(stripped) and bool(definition.match(line))

            if in_focus:
                # Focus ends at the first non-blank line that is not indented
                # deeper than the definition that opened it.
                if stripped and indent <= focus_indent:
                    in_focus = False
                else:
                    out.append(line)
                    continue

            if focus_symbol and is_definition and re.search(
                r"\b" + re.escape(focus_symbol) + r"\b", stripped
            ):
                flush_fold()
                in_focus = True
                focus_indent = indent
                out.append(f"{' ' * indent}# --- focus: {focus_symbol} (expanded) ---")
                out.append(line)
                continue

            if is_definition:
                flush_fold()
                out.append(line.rstrip())
                last_def_index = len(out) - 1
            elif keep.match(line):
                flush_fold()
                out.append(line.rstrip())
            elif stripped.startswith("@") and len(stripped) < 80:
                out.append(line.rstrip())
            elif stripped:
                folded_since_def += 1

        flush_fold()
        return "\n".join(out)
