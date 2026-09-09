"""Thin CLI: python -m ev_s6e9 <download|train|predict|submit|eda>."""

from __future__ import annotations

import argparse
import sys

from ev_s6e9.paths import DATA_RAW, EXPERIMENTS_MD, SUB_CSV, TRAIN_CSV
from ev_s6e9.schema import COMPETITION


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ev_s6e9")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="kaggle competitions download into data/raw")
    d.add_argument("--synth", action="store_true", help="write schema-accurate fake CSVs instead")

    t = sub.add_parser("train", help="stratified CV + save OOF/models + append EXPERIMENTS.md")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--n-estimators", type=int, default=None)
    t.add_argument("--note", default="", help="one-line takeaway for EXPERIMENTS.md")
    t.add_argument("--no-log", action="store_true", help="do not append EXPERIMENTS.md")
    t.add_argument("--synth", action="store_true", help="train on tiny synthetic data (no kaggle files)")

    sub.add_parser("predict", help="average fold models → outputs/submission.csv")

    s = sub.add_parser("submit", help="kaggle competitions submit")
    s.add_argument("-m", "--message", default="lgbm baseline")
    s.add_argument("-f", "--file", default=str(SUB_CSV))

    e = sub.add_parser("eda", help="write plots under reports/")
    e.add_argument("--synth", action="store_true")
    return p


def _need_train_csv(synth: bool):
    from ev_s6e9.data import load_train, synth as make_synth

    if synth:
        return make_synth(800, seed=0)
    if not TRAIN_CSV.exists():
        sys.exit(
            f"missing {TRAIN_CSV}\n"
            "  1. put kaggle.json in ~/.kaggle/ (chmod 600)\n"
            "  2. python -m ev_s6e9 download\n"
            "or: python -m ev_s6e9 train --synth"
        )
    return load_train()


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)

    if args.cmd == "download":
        from ev_s6e9.data import download, write_synth_raw

        if args.synth:
            write_synth_raw()
            print(f"wrote synthetic CSVs under {DATA_RAW}")
        else:
            download()
            print(f"downloaded {COMPETITION} → {DATA_RAW}")
        return

    if args.cmd == "train":
        from ev_s6e9.train import train
        from ev_s6e9.viz import eda

        df = _need_train_csv(args.synth)
        ov = {}
        if args.n_estimators is not None:
            ov["n_estimators"] = args.n_estimators
        if args.synth:
            ov.setdefault("n_estimators", 40)
        train(
            df,
            folds=args.folds,
            seed=args.seed,
            log=not args.no_log and not args.synth,
            note=args.note,
            experiments_path=EXPERIMENTS_MD,
            model_overrides=ov or None,
        )
        eda(df)
        return

    if args.cmd == "predict":
        from ev_s6e9.data import load_test
        from ev_s6e9.predict import predict
        from ev_s6e9.paths import TEST_CSV

        if not TEST_CSV.exists():
            sys.exit(f"missing {TEST_CSV}; run download (or download --synth)")
        predict(load_test())
        return

    if args.cmd == "submit":
        from pathlib import Path

        from ev_s6e9.submit import submit

        submit(path=Path(args.file), message=args.message)
        return

    if args.cmd == "eda":
        from ev_s6e9.viz import eda

        df = _need_train_csv(args.synth)
        eda(df)


if __name__ == "__main__":
    main()
