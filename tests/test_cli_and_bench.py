"""
Tests for the CLI and the benchmark.

`cli.py` had no tests in 0.1.0, which is how `opencontext top` came to launch the
wrong interface and `demo.py` came to import a function that did not exist.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from opencontext import __version__, cli
from opencontext.benchmark import BenchmarkResult, run_benchmark


# -- argument parsing ---------------------------------------------------------------


def test_every_advertised_command_is_wired():
    """A command in the parser but missing from the dispatch table would fall through."""
    parser = cli.build_parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    for name in subparsers.choices:
        assert name in cli.COMMANDS, f"'{name}' is parsed but not dispatched"


def test_top_and_tui_are_distinct_commands():
    """
    0.1.0 mapped both `top` and `tui` to the Textual app, so the monitor the README
    advertised was unreachable and its 214 lines were dead.
    """
    assert cli.COMMANDS["top"] != cli.COMMANDS["tui"]
    assert cli.COMMANDS["top"] == "cmd_top"


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_defaults_to_the_tui(monkeypatch):
    launched = {}

    def fake_tui(args):
        launched["args"] = args
        return 0

    monkeypatch.setattr(cli, "cmd_tui", fake_tui)
    assert cli.main([]) == 0
    assert "args" in launched, "bare `opencontext` should launch the interactive UI"


# -- commands ------------------------------------------------------------------------


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("def run():\n    return 1\n" * 5, encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_scan_reports_what_it_indexed(project, capsys):
    assert cli.main(["scan", str(project)]) == 0
    output = capsys.readouterr().out
    assert "Indexed" in output
    assert "2 files" in output


def test_scan_rejects_a_missing_directory(project, capsys):
    assert cli.main(["scan", str(project / "nope")]) == 1
    assert "Not a directory" in capsys.readouterr().out


def test_status_prints_a_summary(project, capsys):
    assert cli.main(["status", "-d", str(project)]) == 0
    output = capsys.readouterr().out
    assert "Context in use" in output
    assert "Page faults" in output


def test_top_once_renders_a_single_frame(project, capsys):
    assert cli.main(["top", "--once", "-d", str(project)]) == 0
    output = capsys.readouterr().out
    assert "OpenContext" in output
    assert "context" in output


def test_compact_from_a_file(tmp_path, capsys):
    trace_file = tmp_path / "trace.txt"
    trace_file.write_text(
        "Traceback (most recent call last):\n"
        + '  File "/venv/lib/python3.11/site-packages/x.py", line 1, in f\n    g()\n' * 10
        + '  File "/app/main.py", line 3, in run\n    boom()\n'
        + "ValueError: boom",
        encoding="utf-8",
    )
    assert cli.main(["compact", "-f", str(trace_file)]) == 0

    output = capsys.readouterr().out
    assert "python" in output
    assert "site-packages" not in output
    assert "ValueError: boom" in output


def test_compact_from_a_text_argument(capsys):
    assert cli.main(["compact", "-t", "Traceback (most recent call last):\n  File \"a.py\", line 1\nValueError: x"]) == 0


def test_compact_says_so_when_it_cannot_shrink(capsys):
    cli.main(["compact", "-t", "ValueError: boom\nat handler"])
    assert "Not compacted" in capsys.readouterr().out


def test_compact_without_input_is_an_error(capsys, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert cli.main(["compact"]) == 1
    assert "No input" in capsys.readouterr().out


def test_compact_reports_an_unreadable_file(capsys):
    assert cli.main(["compact", "-f", "does_not_exist.txt"]) == 1
    assert "Cannot read" in capsys.readouterr().out


def test_doctor_reports_the_environment(project, capsys):
    exit_code = cli.main(["doctor"])
    output = capsys.readouterr().out

    assert exit_code in (0, 1)
    assert "OpenContext" in output
    assert "Model server" in output
    assert "Token counting" in output


def test_chat_reports_when_no_model_is_available(project, capsys, monkeypatch):
    """The CLI must not invent an answer when there is nothing to answer with."""
    from opencontext import llm

    def unavailable(self, system_prompt, user_prompt, max_tokens=None):
        raise llm.LLMUnavailable("No model server reachable at http://localhost:11434/v1")

    monkeypatch.setattr(llm.LLMClient, "complete", unavailable)
    monkeypatch.setattr(llm.LLMClient, "stream", unavailable)

    assert cli.main(["chat", "how", "does", "this", "work", "-d", str(project)]) == 2
    assert "No answer" in capsys.readouterr().out


def test_chat_without_a_question_is_rejected(project, capsys):
    assert cli.main(["chat", "  "]) == 1
    assert "Provide a question" in capsys.readouterr().out


def test_bench_emits_json(capsys):
    assert cli.main(["bench", "--json", "--no-latency"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["results"]
    assert report["environment"]["opencontext_version"] == __version__


def test_bench_emits_markdown(capsys):
    assert cli.main(["bench", "--markdown", "--no-latency"]) == 0
    output = capsys.readouterr().out
    assert output.startswith("| Workload |")
    assert "%" in output


# -- benchmark ---------------------------------------------------------------------------


def test_benchmark_result_arithmetic():
    result = BenchmarkResult("x", before_tokens=1000, after_tokens=250)
    assert result.saved_tokens == 750
    assert result.reduction_pct == 75.0


def test_benchmark_result_handles_zero_input():
    assert BenchmarkResult("x", 0, 0).reduction_pct == 0.0


def test_benchmark_produces_real_reductions():
    """
    Every published figure comes from this function running over committed fixtures.

    0.1.0's README carried a benchmark table that no code produced.
    """
    report = run_benchmark(include_latency=False)

    assert len(report.results) == 5
    for result in report.results:
        assert result.before_tokens > 0
        assert result.after_tokens > 0
        assert result.after_tokens < result.before_tokens, f"{result.name} achieved no reduction"
        assert 0 < result.reduction_pct < 100


def test_benchmark_latency_is_measured():
    report = run_benchmark(include_latency=True)
    latency = report.latency

    assert latency["samples"] > 0
    assert latency["page_fault_mean_ms"] > 0
    assert latency["page_fault_p95_ms"] >= latency["page_fault_median_ms"]


def test_benchmark_markdown_matches_the_measured_results():
    report = run_benchmark(include_latency=False)
    markdown = report.to_markdown()

    for result in report.results:
        assert result.name in markdown
        assert f"{result.reduction_pct}%" in markdown


def test_benchmark_fixtures_are_committed():
    from opencontext.benchmark import FIXTURE_DIR

    for name in ("python_traceback.txt", "node_stacktrace.txt"):
        assert os.path.exists(os.path.join(FIXTURE_DIR, name)), (
            "benchmark fixtures must be committed so results are reproducible"
        )


# -- serve ---------------------------------------------------------------------------------


@pytest.fixture
def stub_server(monkeypatch):
    """Capture the arguments `cmd_serve` would launch a server with, without binding a port."""
    captured = {}

    def fake_run(kernel=None, host="127.0.0.1", port=9090, auto_open=False, upstream_url=None):
        captured.update(host=host, port=port, auto_open=auto_open, pages=len(kernel.pages))

    monkeypatch.setattr("opencontext.interfaces.proxy.run_proxy_server", fake_run)
    return captured


def test_serve_reports_its_endpoints(project, stub_server, capsys):
    assert cli.main(["serve", "-d", str(project), "--port", "9099"]) == 0

    output = capsys.readouterr().out
    assert "9099" in output
    assert "/v1" in output
    assert "Upstream" in output
    assert stub_server["port"] == 9099
    assert stub_server["pages"] > 0, "the workspace should be indexed before serving"


def test_binding_publicly_without_a_key_warns(project, stub_server, capsys, monkeypatch):
    """
    The dashboard and page APIs expose the indexed tree. Binding off localhost with
    no key must say so — this is the one notice a silent regression could remove.
    """
    monkeypatch.delenv("OPENCONTEXT_API_KEY", raising=False)
    cli.main(["serve", "-d", str(project), "--host", "0.0.0.0"])

    output = capsys.readouterr().out
    assert "Warning" in output
    assert "OPENCONTEXT_API_KEY" in output
    assert "indexed source tree" in output


def test_binding_publicly_with_a_key_does_not_warn(project, stub_server, capsys, monkeypatch):
    monkeypatch.setenv("OPENCONTEXT_API_KEY", "secret")
    cli.main(["serve", "-d", str(project), "--host", "0.0.0.0"])

    output = capsys.readouterr().out
    assert "Warning" not in output
    assert "API key required" in output


def test_binding_to_localhost_does_not_warn(project, stub_server, capsys, monkeypatch):
    monkeypatch.delenv("OPENCONTEXT_API_KEY", raising=False)
    cli.main(["serve", "-d", str(project)])
    assert "Warning" not in capsys.readouterr().out
