# ai4sci-molecule snapshot

`kg_agent/demo_study.py` parses a real research repository into a world model. This directory is
an **unmodified copy** of the small files it reads, so the demo and its tests run in a fresh clone
and on Colab. Nothing here was edited, reformatted or hand-written; that is the point of the demo.

- source: `https://github.com/Jake-Song/ai4sci-molecule` (local checkout `/home/jake/ai4sci-molecule`)
- commit: `ee6a13dc662f23f36bebcbb49bb0e702df235ee6`
- copied: 2026-09-03

## What was copied

| path | why the loader reads it |
| --- | --- |
| `pyproject.toml` | the project name and Python floor |
| `README.md` | the `uv run python -m ai4sci_molecule.weekN` commands, and the `data/MoleculeNet/` path |
| `.gitignore` | which `results/` outputs a fresh clone has to regenerate |
| `results/week*/​*_config.json` | split types, seeds, architectures, strategies, the headline choice |
| `results/week3/summary.json` | ESOL's scaffold statistics |
| `results/week5/accuracy.csv` | the per-regime ensemble ranking |
| `results/week6/{summary.json,budget_efficiency.csv}` | the per-regime strategy ranking and the study's caveats |
| `results/week3/metrics.csv`, `results/week4/model_agreement.csv`, `results/week5/ranking.csv`, `results/week6/{cold_start,strategy_comparison}.csv` | committed artifacts of each study; listed in the graph, not parsed further |

The `.gitignore` above is a copy of that repository's own, and applies to this directory as well.
It ignores nothing that is stored here.

## What was left out

`results/week4/summary.json` (342 KB), every `splits.json` (75–125 KB each), all figures, and the
150 per-trajectory shards under `results/week6/`. Point the demo at a real checkout to include
them, along with the study order parsed from `src/ai4sci_molecule/week*.py` imports:

```bash
uv run python -m kg_agent.demo_study --repo /path/to/ai4sci-molecule
```

## Refreshing it

Re-copy the files listed above from a newer checkout and update the commit hash here. The loader
treats every file as optional, so a partial refresh degrades to a smaller graph rather than an
error.
