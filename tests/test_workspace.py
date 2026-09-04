"""Tests for the workspace scanner."""

from __future__ import annotations

import os

import pytest

from contextos.core.kernel import ContextKernel
from contextos.core.types import PageTier
from contextos.core.workspace import WorkspaceScanner
from contextos.storage.swap import SwapStorage


@pytest.fixture
def project(tmp_path):
    """A small project tree with the usual noise around it."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def run():\n    return 'ok'\n" * 10, encoding="utf-8")
    (tmp_path / "src" / "utils.py").write_text("def helper():\n    pass\n" * 20, encoding="utf-8")
    (tmp_path / "README.md").write_text("# Project\n\nDocs.\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("ignore me", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("module.exports = {}", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.py").write_text("generated", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    return tmp_path


def build(tmp_path, root, budget=20000):
    kernel = ContextKernel(token_budget=budget, swap_storage=SwapStorage(str(tmp_path / "swap.db")))
    return kernel, WorkspaceScanner(kernel, root_dir=str(root))


def test_scan_indexes_source_and_skips_noise(project, tmp_path):
    kernel, scanner = build(tmp_path, project)
    result = scanner.scan_and_ingest()

    titles = {page.title for page in kernel.pages.values()}
    assert "src/app.py" in titles
    assert "src/utils.py" in titles
    assert "README.md" in titles

    assert not any("node_modules" in title for title in titles)
    assert not any(title.startswith(".git") for title in titles)
    assert not any("build/" in title for title in titles)
    assert not any(title.endswith(".png") for title in titles)
    assert result["files_scanned"] == 4


def test_page_ids_are_relative_and_slash_separated(project, tmp_path):
    kernel, scanner = build(tmp_path, project)
    scanner.scan_and_ingest()

    assert "file:src/app.py" in kernel.pages
    assert not any("\\" in page_id for page_id in kernel.pages)


def test_root_config_files_are_pinned(project, tmp_path):
    kernel, scanner = build(tmp_path, project)
    result = scanner.scan_and_ingest()

    assert kernel.pages["file:README.md"].tier == PageTier.L0_PINNED
    assert kernel.pages["file:pyproject.toml"].tier == PageTier.L0_PINNED
    assert kernel.pages["file:src/app.py"].tier == PageTier.L1_WORKING_RAM
    assert result["files_pinned"] == 2


def test_nested_config_files_are_not_pinned(project, tmp_path):
    """
    Pinning matched on bare filename at any depth in 0.1.0, so a monorepo pinned
    every nested package.json into a tier that cannot be evicted.
    """
    packages = project / "packages"
    for name in ("web", "api", "worker"):
        (packages / name).mkdir(parents=True)
        (packages / name / "package.json").write_text('{"name": "%s"}' % name, encoding="utf-8")

    kernel, scanner = build(tmp_path, project)
    scanner.scan_and_ingest()

    nested = [page for page in kernel.pages.values() if "packages/" in page.title]
    assert len(nested) == 3
    assert all(page.tier == PageTier.L1_WORKING_RAM for page in nested), (
        "only root-level config files should be pinned"
    )


def test_pinning_is_capped_by_budget_share(tmp_path):
    """Pinned content may not swallow the whole budget."""
    root = tmp_path / "big"
    root.mkdir()
    (root / "README.md").write_text("padding line\n" * 5000, encoding="utf-8")
    (root / "pyproject.toml").write_text("padding = 1\n" * 5000, encoding="utf-8")

    kernel, scanner = build(tmp_path, root, budget=2000)
    scanner.scan_and_ingest()

    pinned = sum(
        page.token_count for page in kernel.pages.values() if page.tier == PageTier.L0_PINNED
    )
    assert pinned <= kernel.token_budget, "pinned content must not exceed the budget outright"


def test_binary_files_without_an_extension_are_skipped(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "data").write_bytes(b"\x00\x01\x02binary\x00content")
    (root / "real.py").write_text("x = 1\n", encoding="utf-8")

    kernel, scanner = build(tmp_path, root)
    result = scanner.scan_and_ingest()

    assert result["files_scanned"] == 1
    assert result["files_skipped_binary"] == 1


def test_oversized_files_are_skipped(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "huge.py").write_text("x = 1\n" * 100_000, encoding="utf-8")
    (root / "small.py").write_text("y = 2\n", encoding="utf-8")

    kernel, scanner = build(tmp_path, root)
    result = scanner.scan_and_ingest(max_file_size_kb=10)

    assert result["files_skipped_large"] == 1
    assert result["files_scanned"] == 1


def test_max_files_caps_the_walk(tmp_path):
    root = tmp_path / "many"
    root.mkdir()
    for index in range(40):
        (root / f"mod_{index}.py").write_text(f"value = {index}\n", encoding="utf-8")

    kernel, scanner = build(tmp_path, root)
    result = scanner.scan_and_ingest(max_files=10)
    assert result["files_scanned"] == 10


def test_rescanning_does_not_duplicate_pages(project, tmp_path):
    kernel, scanner = build(tmp_path, project)
    scanner.scan_and_ingest()
    first = len(kernel.pages)

    scanner.scan_and_ingest()
    assert len(kernel.pages) == first


def test_scan_reports_working_and_swapped_totals(project, tmp_path):
    kernel, scanner = build(tmp_path, project, budget=300)
    result = scanner.scan_and_ingest()

    assert result["working_tokens"] <= kernel.token_budget or kernel.get_metrics().over_budget
    assert result["total_tokens"] > 0


def test_list_tracked_files(project, tmp_path):
    kernel, scanner = build(tmp_path, project)
    scanner.scan_and_ingest()
    tracked = scanner.list_tracked_files()

    assert "file:src/app.py" in tracked
    assert tracked == sorted(tracked)
