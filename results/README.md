# Results archive

Each run of `src.main` writes JSON to `log/run/` (gitignored — scratch).
**Runs worth keeping** are copied into a timestamped directory here, with a rendered summary so reviewers (and future-you) can read the numbers without running anything.

## Protocol for a new recorded run

```bash
# 1. Run
uv run python -m src.main

# 2. Archive
STAMP=$(date +%Y-%m-%d)_<short-name>
mkdir -p results/$STAMP
cp log/run/results.json results/$STAMP/
cp config/config.yaml results/$STAMP/config_snapshot.yaml

# 3. Render PNG + markdown summary
uv run python scripts/render_results.py results/$STAMP

# 4. Commit
git add results/$STAMP && git commit -m "Record run: $STAMP"
```

## Runs

| Date | Directory | Notes |
|---|---|---|
| 2026-04-17 | [`2026-04-17_baseline/`](2026-04-17_baseline/summary.md) | First full reproduction; patience=5; identifies NMF anomaly. See `PLAN.md` Decisions log. |
