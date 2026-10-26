# Entropiq

Entropiq ranks the source files of a Git repository by how much maintenance pain each is likely to cause in the
next 12 months, explains why, and turns the ranking into a fix plan that fits a time budget.

Internally the project is called TDP (AI Technical Debt Prioritizer); Python packages use the `tdp_` prefix.

## Quick start

```bash
uv sync
uv run pytest
```

## Layout

| Path | What |
|---|---|
| `packages/core` | `tdp_core`: mining, parsing, features, labels, model runtime, planning |
| `packages/platform` | `tdp_platform`: database models, settings, observability |
| `services/api`, `services/worker` | FastAPI service and Celery workers |
| `cli` | `tdp` command |
| `ml` | `tdp-ml`: dataset, training, evaluation |
| `handoff/` | one note per finished implementation step |

Build work follows [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Design documents are in `docs/`.
