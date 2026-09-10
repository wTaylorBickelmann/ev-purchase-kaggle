# Cursor / agent instructions

Conventions for this repo (and for any agent working in it).

## Code style

- Prefer **human-readable** code over clever code.
- Prefer **small modular functions** and **as few LOC as possible** while staying clear and modular.
- No god-files, no notebook-only spaghetti. Notebooks and CLIs call into the library; they do not own the logic.
- Type hints on public functions. Short names, clear structure. Docstrings only when they add something.

## Architecture

- Keep a thin package layout (`src/ev_s6e9/` or equivalent) with focused modules.
- **Feature engineering** and **modeling** each get their **own classes** (not only free functions dumped in a script). Put reusable transforms / model wrappers behind those class APIs so train, predict, notebooks, and site builds share one path.
- Other concerns (data I/O, CV, submit, experiments log, site build) stay in small modules that call those classes.

## Experiments

- Append-only `EXPERIMENTS.md`: one terse chunk per attempt.
- After a Kaggle submit scores, the chunk for that attempt must include the public **LB** score (poll submissions if needed). Use `LB: —` only while scoring is still pending.

## Do not

- Commit secrets, Kaggle credentials, or full competition CSVs.
- Rewrite old experiment log entries.
- Sacrifice readability for micro-optimizations or one-off notebook hacks.
