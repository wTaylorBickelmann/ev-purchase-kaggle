"""Static gallery builder."""

from __future__ import annotations

from ev_s6e9.site import build_site, is_full_train


def test_is_full_train_rejects_sample():
    from ev_s6e9.paths import TRAIN_SAMPLE_CSV

    assert is_full_train(TRAIN_SAMPLE_CSV) is False


def test_build_site_writes_gallery(tmp_path):
    out = build_site(out=tmp_path / "site", sample=True)
    for name in ("index.html", "eda.html", "features.html", "training.html", ".nojekyll"):
        assert (out / name).exists()
    html = (out / "training.html").read_text(encoding="utf-8")
    assert "EXPERIMENTS.md" in html
    assert "0.94150" in html or "Will_Buy_EV" in html
    assert (out / "reports" / "target_rate.png").exists()
    assert "html-wasm" not in (out / "index.html").read_text(encoding="utf-8")
    assert "read-only" in (out / "index.html").read_text(encoding="utf-8").lower()
