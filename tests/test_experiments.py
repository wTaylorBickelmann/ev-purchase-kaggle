"""EXPERIMENTS.md append-only helper."""

from __future__ import annotations

from ev_s6e9.experiments import HEADER, append_chunk, format_chunk, parse_chunks


def test_format_chunk_shape():
    text = format_chunk("lgbm raw", "0.80000 ± 0.00100", lb="—", takeaway="ok", day="2026-09-09")
    assert text.startswith("### 2026-09-09 — lgbm raw")
    assert "- CV: 0.80000 ± 0.00100" in text
    assert "- LB: —" in text
    assert "- Takeaway: ok" in text


def test_parse_chunks_ignores_template_fence():
    text = "```\n### YYYY-MM-DD — x\n```\n\n### 2026-09-09 — real\n- CV: 1\n"
    ch = parse_chunks(text)
    assert len(ch) == 1
    assert "real" in ch[0]


def test_append_does_not_rewrite(tmp_path):
    path = tmp_path / "EXPERIMENTS.md"
    path.write_text(HEADER + "### 2026-01-01 — old\n- CV: 0.1\n- LB: —\n- Takeaway: keep me\n\n")
    first = path.read_text()
    append_chunk(format_chunk("new", "0.2", takeaway="second", day="2026-09-09"), path=path)
    later = path.read_text()
    assert later.startswith(first.rstrip() + "\n") or first in later
    assert later.index("### 2026-01-01") < later.index("### 2026-09-09")
    assert later.count("### 2026-01-01 — old") == 1
    assert "keep me" in later
