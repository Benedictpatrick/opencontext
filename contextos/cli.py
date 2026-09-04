"""
Command-line interface.

    contextos              interactive terminal UI (same as `tui`)
    contextos top          live read-only monitor
    contextos status       one-shot status summary
    contextos scan         index a project into context memory
    contextos chat         ask a question about the codebase
    contextos compact      compact a traceback from a file, argument or stdin
    contextos serve        run the dashboard and OpenAI-compatible proxy
    contextos mcp          run the MCP stdio server
    contextos bench        run the benchmark and print measured results
    contextos doctor       check the installation and configuration

Interface modules are imported inside the command that needs them. Importing them
at module scope meant `contextos mcp` — a stdio server needing neither — loaded
FastAPI, uvicorn and Textual before serving a byte, and an import error in any one
interface broke all of them.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from contextos import __version__

console = Console()


def build_kernel(token_budget: int = 16000, directory: Optional[str] = None, scan: bool = True):
    """Create a kernel and, unless told otherwise, index the target directory."""
    from contextos.core.kernel import ContextKernel
    from contextos.core.workspace import WorkspaceScanner

    kernel = ContextKernel(token_budget=token_budget)
    if scan:
        WorkspaceScanner(kernel, root_dir=directory).scan_and_ingest()
    return kernel


# -- commands -------------------------------------------------------------------


def cmd_tui(args: argparse.Namespace) -> int:
    """Launch the interactive Textual UI."""
    from contextos.core.session import DEFAULT_SESSION_PATH
    from contextos.interfaces.interactive_tui import run_interactive_tui

    kernel = build_kernel(args.budget, args.directory)
    # The UI saves pins, tiers and the budget on exit and restores them on start,
    # so an arrangement survives closing the app. --fresh ignores what was saved.
    session_path = None if getattr(args, "fresh", False) else DEFAULT_SESSION_PATH
    run_interactive_tui(kernel, root_dir=args.directory, session_path=session_path)
    return 0


def cmd_top(args: argparse.Namespace) -> int:
    """Launch the live read-only monitor."""
    from contextos.interfaces.tui import run_top

    kernel = build_kernel(args.budget, args.directory)
    run_top(kernel, refresh_rate=args.interval, once=args.once)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print a one-shot summary of context memory."""
    kernel = build_kernel(args.budget, args.directory)
    metrics = kernel.get_metrics()

    summary = Table(expand=True, box=None)
    summary.add_column("", style="#8b949e")
    summary.add_column("", style="#e6edf3")

    utilization = f"{metrics.working_tokens:,} / {metrics.token_budget:,} tokens ({metrics.budget_utilization_pct}%)"
    if metrics.over_budget:
        utilization += "  [bold #f85149]OVER BUDGET[/bold #f85149]"
    summary.add_row("Context in use", utilization)
    summary.add_row(
        "Swapped to disk", f"{metrics.swapped_tokens:,} tokens across {metrics.l3_pages} pages"
    )
    summary.add_row("Kept out of context", f"{metrics.total_tokens_saved:,} tokens")
    faults = f"{metrics.total_page_faults}"
    if metrics.total_page_faults:
        faults += f" ({metrics.avg_page_fault_ms:.2f} ms average)"
    summary.add_row("Page faults", faults)
    if metrics.swap_disk_rows != metrics.l3_pages:
        summary.add_row(
            "Swap database",
            f"{metrics.swap_disk_rows} rows, {metrics.swap_disk_tokens:,} tokens "
            f"[dim](includes earlier runs — `contextos scan --prune-swap` to clean)[/dim]",
        )

    console.print(Panel(summary, title=f"[bold #f59e0b]ContextOS {__version__}[/bold #f59e0b]", border_style="#30363d"))

    from contextos.core.types import PageStatus, PageTier

    pages = Table(expand=True, border_style="#30363d")
    pages.add_column("State", justify="center", width=9)
    pages.add_column("Tier", justify="center", width=5)
    pages.add_column("Page", style="#e6edf3", no_wrap=True)
    pages.add_column("Tokens", justify="right", style="bold")

    ordered = sorted(
        kernel.pages.values(),
        key=lambda p: (0 if p.tier == PageTier.L0_PINNED else 1, -p.token_count),
    )
    for page in ordered[: args.limit]:
        resident = page.status == PageStatus.ACTIVE
        pages.add_row(
            "[#3fb950]resident[/#3fb950]" if resident else "[#d29922]swapped[/#d29922]",
            {"L0_PINNED": "L0", "L1_WORKING": "L1", "L2_EPISODIC": "L2"}.get(page.tier.value, "??"),
            page.title,
            f"{page.token_count:,}",
        )

    console.print(Panel(pages, title=f"[bold]Pages ({len(kernel.pages)} total)[/bold]", border_style="#30363d"))
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Index a project directory into context memory."""
    from contextos.core.kernel import ContextKernel
    from contextos.core.workspace import WorkspaceScanner

    target = os.path.abspath(args.directory or os.getcwd())
    if not os.path.isdir(target):
        console.print(f"[#f85149]Not a directory:[/#f85149] {target}")
        return 1

    console.print(f"[dim]Scanning[/dim] [#f59e0b]{target}[/#f59e0b]")
    kernel = ContextKernel(token_budget=args.budget)

    if args.prune_swap:
        removed = kernel.swap.clear()
        console.print(f"[dim]Cleared {removed} rows from swap database[/dim]")

    result = WorkspaceScanner(kernel, root_dir=target).scan_and_ingest()

    console.print(f"[#3fb950]Indexed[/#3fb950]      {result['files_scanned']} files, {result['total_tokens']:,} tokens")
    console.print(f"[#3fb950]In context[/#3fb950]   {result['working_tokens']:,} tokens")
    console.print(f"[#d29922]Swapped[/#d29922]      {result['swapped_tokens']:,} tokens")

    skipped = result["files_skipped_binary"] + result["files_skipped_large"] + result["files_skipped_error"]
    if skipped:
        console.print(
            f"[dim]Skipped {skipped} files "
            f"({result['files_skipped_binary']} binary, {result['files_skipped_large']} oversized, "
            f"{result['files_skipped_error']} unreadable)[/dim]"
        )
    if result["files_pinned"]:
        console.print(f"[dim]{result['files_pinned']} root files pinned and protected from eviction[/dim]")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Ask a question about the codebase, with context assembled by the kernel."""
    from contextos.core.pager import ContextPager
    from contextos.llm import LLMClient, LLMUnavailable

    question = " ".join(args.message).strip()
    if not question:
        console.print("[#f85149]Provide a question.[/#f85149]")
        return 1

    kernel = build_kernel(args.budget, args.directory)
    pager = ContextPager(kernel, root_dir=args.directory)
    resolved = pager.process_incoming_prompt_verbose(question)

    if resolved["rehydrated"]:
        console.print(
            f"[#f59e0b]Paged in[/#f59e0b] {', '.join(resolved['rehydrated'])}"
        )
    console.print(
        f"[dim]Context assembled: {resolved['context_tokens']:,} tokens "
        f"of a {kernel.token_budget:,} budget[/dim]\n"
    )

    client = LLMClient()
    system_prompt = (
        "You are a software engineering assistant answering questions about a "
        "codebase. The context below was assembled by ContextOS from the user's "
        "workspace. Pages marked as swapped are on disk and not shown; say so if "
        "you need one.\n\n" + resolved["context"]
    )

    try:
        if args.no_stream:
            console.print(client.complete(system_prompt, question))
        else:
            for delta in client.stream(system_prompt, question):
                console.print(delta, end="")
            console.print()
    except LLMUnavailable as error:
        # No answer is invented here. A fabricated response is worse than none,
        # because the user cannot tell it from a real one.
        console.print(f"[#f85149]No answer: {error}[/#f85149]")
        return 2

    return 0


def cmd_compact(args: argparse.Namespace) -> int:
    """Compact a traceback and report the measured reduction."""
    from contextos.core.compactor import TracebackCompactor

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError as error:
            console.print(f"[#f85149]Cannot read {args.file}: {error}[/#f85149]")
            return 1
    else:
        text = args.text_flag or args.text or (sys.stdin.read() if not sys.stdin.isatty() else "")

    if not text.strip():
        console.print("[#f85149]No input. Pass a file with -f, text with -t, or pipe to stdin.[/#f85149]")
        return 1

    compacted, before, after = TracebackCompactor.compact(text)
    saved = before - after

    if saved == 0:
        console.print(
            f"[dim]Not compacted ({before} tokens). "
            f"Detected as: {TracebackCompactor.detect_language(text)}. "
            "Output would not have been smaller, so the original is unchanged.[/dim]\n"
        )
    else:
        console.print(
            f"[#38bdf8]{before:,}[/#38bdf8] -> [#3fb950]{after:,} tokens[/#3fb950] "
            f"([#3fb950]-{round(saved / before * 100, 1)}%[/#3fb950], "
            f"{TracebackCompactor.detect_language(text)})\n"
        )
    console.print(compacted)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the dashboard and OpenAI-compatible proxy."""
    from contextos.interfaces.proxy import run_proxy_server
    from contextos.llm import LLMConfig

    upstream = LLMConfig.from_env()

    console.print(f"\n[bold #f59e0b]ContextOS {__version__}[/bold #f59e0b]\n")
    console.print(f"  Dashboard   http://{args.host}:{args.port}/")
    console.print(f"  Proxy       http://{args.host}:{args.port}/v1")
    console.print(f"  Upstream    {upstream.describe()}")
    console.print(f"  Budget      {args.budget:,} tokens")

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        if os.environ.get("CONTEXTOS_API_KEY"):
            console.print("  Auth        [#3fb950]API key required[/#3fb950]")
        else:
            console.print(
                "\n[bold #f85149]  Warning:[/bold #f85149] binding to "
                f"{args.host} with no CONTEXTOS_API_KEY set.\n"
                "  The dashboard and page APIs expose your indexed source tree to the network.\n"
                "  Set CONTEXTOS_API_KEY, or bind to 127.0.0.1."
            )

    console.print("\n[dim]Ctrl+C to stop.[/dim]\n")

    kernel = build_kernel(args.budget, args.directory)
    try:
        run_proxy_server(kernel=kernel, host=args.host, port=args.port, auto_open=args.open)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    """Run the MCP stdio server."""
    from contextos.interfaces.mcp_server import run_mcp_server

    run_mcp_server(root_dir=args.directory)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Run the benchmark and print measured results."""
    from contextos.benchmark import run_benchmark

    machine_readable = args.json or args.markdown
    if machine_readable:
        # Status goes to stderr so `contextos bench --json | jq` stays parseable.
        print("Running benchmark over committed fixtures...", file=sys.stderr)
    else:
        console.print("[dim]Running benchmark over committed fixtures...[/dim]\n")

    report = run_benchmark(include_latency=not args.no_latency)

    if args.json:
        import json

        print(json.dumps(report.as_dict(), indent=2))
        return 0

    if args.markdown:
        print(report.to_markdown())
        return 0

    table = Table(expand=True, border_style="#30363d")
    table.add_column("Workload", style="#e6edf3")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Reduction", justify="right", style="bold #3fb950")

    for result in report.results:
        table.add_row(
            f"{result.name}\n[dim]{result.detail}[/dim]",
            f"{result.before_tokens:,}",
            f"{result.after_tokens:,}",
            f"{result.reduction_pct}%",
        )
    console.print(Panel(table, title="[bold #f59e0b]Token reduction (measured)[/bold #f59e0b]", border_style="#30363d"))

    if report.latency:
        latency = Table(expand=True, box=None)
        latency.add_column("", style="#8b949e")
        latency.add_column("", style="#e6edf3")
        latency.add_row("Page fault, median", f"{report.latency['page_fault_median_ms']:.3f} ms")
        latency.add_row("Page fault, mean", f"{report.latency['page_fault_mean_ms']:.3f} ms")
        latency.add_row("Page fault, p95", f"{report.latency['page_fault_p95_ms']:.3f} ms")
        latency.add_row("Swap out, per page", f"{report.latency['swap_out_per_page_ms']:.3f} ms")
        console.print(Panel(latency, title="[bold]Latency[/bold]", border_style="#30363d"))

    environment = report.environment
    console.print(
        f"[dim]ContextOS {environment['contextos_version']} - Python {environment['python']} "
        f"on {environment['platform']} - token counting: {environment['tokenizer']}[/dim]"
    )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what is installed, configured and reachable."""
    from contextos.llm import LLMClient, LLMConfig

    table = Table(expand=True, box=None)
    table.add_column("", width=3)
    table.add_column("", style="#8b949e", width=22)
    table.add_column("", style="#e6edf3")

    ok, warn, bad = "[#3fb950]OK[/#3fb950]", "[#d29922]--[/#d29922]", "[#f85149]!![/#f85149]"
    failures = 0

    table.add_row(ok, "ContextOS", f"{__version__} at {os.path.dirname(os.path.abspath(__file__))}")
    table.add_row(ok, "Python", f"{sys.version.split()[0]} ({sys.platform})")

    for label, module in (("Dashboard/proxy", "fastapi"), ("Interactive UI", "textual"), ("Terminal output", "rich")):
        try:
            __import__(module)
            table.add_row(ok, label, f"{module} available")
        except ImportError:
            table.add_row(bad, label, f"{module} is not installed")
            failures += 1

    try:
        from contextos.core.tokens import _get_encoder

        table.add_row(
            ok,
            "Token counting",
            "tiktoken (exact)" if _get_encoder() else "built-in heuristic (set CONTEXTOS_TOKENIZER=tiktoken for exact)",
        )
    except Exception as error:
        table.add_row(warn, "Token counting", str(error))

    config = LLMConfig.from_env()
    client = LLMClient(config)
    if client.is_reachable():
        models = client.available_models()
        detail = config.describe()
        if models and config.model not in models:
            detail += f"\n[#d29922]CONTEXTOS_MODEL='{config.model}' is not offered. Available: {', '.join(models[:5])}[/#d29922]"
        table.add_row(ok, "Model server", detail)
    else:
        table.add_row(
            warn,
            "Model server",
            f"Not reachable at {config.base_url}.\n"
            "`contextos chat` needs one; the kernel, TUI, scan and bench do not.",
        )

    swap_path = os.path.join(os.getcwd(), ".contextos", "swap.db")
    if os.path.exists(swap_path):
        from contextos.storage.swap import SwapStorage

        stats = SwapStorage(swap_path).get_disk_stats()
        table.add_row(
            ok if stats["rows"] < 500 else warn,
            "Swap database",
            f"{stats['rows']} rows, {stats['tokens']:,} tokens at {swap_path}",
        )
    else:
        table.add_row(ok, "Swap database", "none yet (created on first swap)")

    console.print(Panel(table, title="[bold #f59e0b]contextos doctor[/bold #f59e0b]", border_style="#30363d"))
    return 1 if failures else 0


# -- argument parsing ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextos",
        description="A virtual memory kernel for LLM context windows.",
    )
    parser.add_argument("--version", action="version", version=f"contextos {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--budget", "-b", type=int, default=16000, help="Working token budget")
        subparser.add_argument("--directory", "-d", default=None, help="Project directory (default: cwd)")

    tui = subparsers.add_parser("tui", help="Interactive terminal UI")
    add_common(tui)
    tui.add_argument(
        "--fresh", action="store_true", help="Ignore any saved session and start clean"
    )

    top = subparsers.add_parser("top", help="Live read-only monitor")
    add_common(top)
    top.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds")
    top.add_argument("--once", action="store_true", help="Print one frame and exit")

    status = subparsers.add_parser("status", help="One-shot status summary")
    add_common(status)
    status.add_argument("--limit", type=int, default=15, help="Pages to list")

    scan = subparsers.add_parser("scan", help="Index a project into context memory")
    scan.add_argument("directory", nargs="?", default=None, help="Directory to scan")
    scan.add_argument("--budget", "-b", type=int, default=16000, help="Working token budget")
    scan.add_argument("--prune-swap", action="store_true", help="Clear the swap database first")

    chat = subparsers.add_parser("chat", help="Ask a question about the codebase")
    chat.add_argument("message", nargs="+", help="Your question")
    add_common(chat)
    chat.add_argument("--no-stream", action="store_true", help="Wait for the full reply")

    compact = subparsers.add_parser("compact", help="Compact a traceback")
    compact.add_argument("text", nargs="?", default="", help="Error text")
    compact.add_argument("--file", "-f", help="Read the trace from a file")
    compact.add_argument("--text", "-t", dest="text_flag", help="Error text")

    serve = subparsers.add_parser("serve", help="Run the dashboard and proxy")
    add_common(serve)
    serve.add_argument("--host", default="127.0.0.1", help="Bind address")
    serve.add_argument("--port", "-p", type=int, default=9090, help="Port")
    serve.add_argument("--open", "-o", action="store_true", help="Open the dashboard in a browser")

    mcp = subparsers.add_parser("mcp", help="Run the MCP stdio server")
    mcp.add_argument("--directory", "-d", default=None, help="Project directory")

    bench = subparsers.add_parser("bench", help="Run the benchmark")
    bench.add_argument("--json", action="store_true", help="Emit JSON")
    bench.add_argument("--markdown", action="store_true", help="Emit a markdown table")
    bench.add_argument("--no-latency", action="store_true", help="Skip latency measurement")

    subparsers.add_parser("doctor", help="Check installation and configuration")

    return parser


# Command name -> handler attribute on this module. Resolved by name at call time
# rather than bound at import, so the dispatch always reflects the current module
# state (which is also what makes the commands patchable in tests).
COMMANDS = {
    "tui": "cmd_tui",
    "top": "cmd_top",
    "status": "cmd_status",
    "scan": "cmd_scan",
    "chat": "cmd_chat",
    "compact": "cmd_compact",
    "serve": "cmd_serve",
    "mcp": "cmd_mcp",
    "bench": "cmd_bench",
    "doctor": "cmd_doctor",
}


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        args = parser.parse_args(["tui"])

    handler_name = COMMANDS.get(args.command)
    if handler_name is None:
        parser.print_help()
        return 1

    handler = globals()[handler_name]
    try:
        return handler(args)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
