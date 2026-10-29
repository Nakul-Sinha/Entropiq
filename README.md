# Entropiq

Entropiq ranks the Python source files of a Git repository by how much maintenance pain each is likely to cause
in the next 12 months, explains why, and turns the ranking into a fix plan that fits a time budget.

For every file it reports:

- a **priority** from 0 to 100 (a percentile within the repository) and a **band** (Critical, High, Medium, Low);
- a calibrated **risk** that the file carries a latent defect found in the next 12 months, with a range;
- the **reasons** behind the score (the feature families that pushed it up), its **debt types** and an estimated
  **fix cost** in hours;
- a **fix plan** that picks the files with the best return within a budget, a **what-if** simulation of fixes, and
  optional **AI advice** for one file.

Internally the project is called TDP (AI Technical Debt Prioritizer); Python packages use the `tdp_` prefix.

## Install

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and git.

```bash
git clone https://github.com/Nakul-Sinha/Entropiq.git
cd Entropiq
uv sync
uv run entropiq --version
```

Inside the checkout `uv run entropiq` finds the production model through `models/production.json`.

### A global command

To run `entropiq` (or its short name `tdp`) from any folder, install the command line as a uv tool:

```bash
uv tool install ./cli --with anthropic
```

If uv says its tool folder is not on your `PATH`, run `uv tool update-shell` once and open a new terminal. After
pulling new code, run `uv tool install --reinstall ./cli --with anthropic`. (`--with anthropic` adds the client that
`/advise` uses.)

An installed command lives outside the checkout, so give it the production model once: copy
`models/production.json` and the bundle folder it names into the Entropiq home folder.

```text
~/.entropiq/models/production.json          (%USERPROFILE%\.entropiq\models on Windows)
~/.entropiq/models/ranker-2026.10.1/        (the bundle: ranker.json, calibrator.json, metadata.json, ...)
```

`entropiq model` then reports `production model` and names that pointer. Other ways to choose a model:
`--model-dir DIR`, the `TDP_MODEL_DIR` variable, or `--dev-model` for a small synthetic demo model. Without any of
them Entropiq falls back to the demo model and says so on every screen.

### Shell completion

`entropiq --install-completion` detects bash, zsh, fish or PowerShell and adds completion to that shell's profile
(restart the shell afterwards). `entropiq --show-completion` prints the script instead, so you can install it by
hand. Inside the interactive shell, Tab completes commands and their arguments without any setup.

## Quick start

```bash
uv run entropiq            # or just: entropiq
```

```text
entropiq ❯ /analyze .                      # a local repository, or a GitHub URL
entropiq ❯ /files                          # the ranked files, 20 per page
entropiq ❯ /file 1                         # everything about the top file
entropiq ❯ /plan --budget 10               # what to fix with 10 hours
entropiq ❯ /quit
```

Every view also runs once as a subcommand, which is handy for scripts and CI:

```bash
entropiq analyze . --top 10
entropiq report . --format sarif -o entropiq.sarif
```

Analyses are saved in `~/.entropiq/analyses`, so reopening one (`/open`, `/history`) is instant and `/trend` can
compare them over time.

## Commands

### Interactive shell

| Group | Command | Aliases | What it does |
|---|---|---|---|
| Analysis | `/analyze <path or URL>` | `/analyse` | Analyse a local repository or a GitHub URL |
| Analysis | `/history` | | List saved analyses |
| Analysis | `/open <number or sha>` | `/load` | Reopen a saved analysis without running it again |
| Analysis | `/trend` | | Show how Critical files and hours changed across saved analyses |
| Explore | `/file <path or rank>` | | Show the full detail of one file |
| Explore | `/files` | | List ranked files with filters, sorting and paging |
| Explore | `/graph [file]` | | Show central files, import cycles and hidden couplings, or one file's dependencies |
| Explore | `/tree [folder] [--files]` | | Show the folder tree with lines of code and the highest band |
| Explore | `/whatif <file> [--all]` | `/what-if` | Simulate fixes on one file and compare its priority and risk |
| Plan | `/advise <file>` | `/advice` | Get AI fix advice for one file (needs `ANTHROPIC_API_KEY`) |
| Plan | `/plan [--budget HOURS]` | | Show the fix plan that fits a time budget |
| Report | `/diff --base REF` | | Rank the files changed since a base commit |
| Report | `/export md\|json\|sarif [-o FILE]` | `/report` | Write the analysis to a Markdown, JSON or SARIF file |
| Session | `/model` | | Show the model in use and its training metrics |
| Session | `/settings [key value \| reset]` | | Show or change settings |
| Session | `/status` | | Show the loaded repository, model and settings |
| General | `/clear` | `/cls` | Clear the screen |
| General | `/help [command]` | `/h` | List commands or explain one |
| General | `/quit` | `/exit`, `/q` | Leave Entropiq (Ctrl+D works too) |

A file argument can be its path, a unique ending such as `core.py`, or its rank (`1` or `#1`).

### Subcommands

The commands that read a repository take an optional path or GitHub URL (default `.`), plus `--ref`, `--fresh`,
`--model-dir` and `--dev-model`. Run `entropiq <command> --help` for the full list of options.

| Command | What it does |
|---|---|
| `entropiq analyze [PATH]` | Analyse a local repository or a GitHub URL (`--top N`, `--format table\|json\|md`) |
| `entropiq history` | List saved analyses |
| `entropiq trend [PATH]` | Show how Critical files and hours changed across saved analyses |
| `entropiq files [PATH]` | List ranked files with filters and sorting (for example `--band critical --format json`) |
| `entropiq file FILE [PATH]` | Show the full detail of one file |
| `entropiq graph [PATH]` | Show central files, import cycles and hidden couplings, or one file's dependencies |
| `entropiq tree [PATH]` | Show the folder tree with lines of code and the highest band |
| `entropiq whatif FILE [PATH]` | Simulate fixes on one file (`--add-tests`, `--split`, `--dedupe`, `--resolve-satd`, `--break-cycle`, `--all`) |
| `entropiq advise FILE [PATH]` | Get AI fix advice for one file (needs `ANTHROPIC_API_KEY`) |
| `entropiq plan [PATH]` | Show the fix plan that fits a time budget (`--budget HOURS`) |
| `entropiq diff [PATH] --base REF` | Rank the files changed since a base commit |
| `entropiq report [PATH]` | Write the analysis to a Markdown, JSON or SARIF file (`--format`, `-o FILE`) |
| `entropiq model` | Show the model in use and its training metrics |

Root options: `--version`, `--model-dir DIR`, `--dev-model`, `--install-completion`, `--show-completion`.

Exit codes: 0 success, 1 runtime failure (git, network, disk, or advice without an API key), 2 bad input
(paths, URLs, options, a folder that is not a model bundle), 130 cancelled with Ctrl+C.

## In pull requests

The GitHub Action in `action/` runs `tdp diff` on every pull request and keeps one comment with the changed files'
band, priority, risk and top reason, plus the same report in the job summary. It can fail the job when a changed
file is High or Critical and can upload SARIF to code scanning. It downloads the production model from the
`model-ranker-2026.10.1` release of this repository and checks its sha256. Add it with
`uses: Nakul-Sinha/Entropiq/action@main` and `base: ${{ github.event.pull_request.base.sha }}`; the inputs,
permissions and a full example workflow are in [action/README.md](action/README.md).

## How the model was built

All numbers below come from the recorded runs (the handoff notes in `handoff/` and the result files they list).

**Data.** 23 open-source Python repositories were evaluated and 20 kept. pallets/click (no bug label),
marshmallow (19 source files, no bug label) and Kludex/starlette (26 bug issues, under the 30 needed) were dropped
from training; click and starlette are still used as golden test repositories. Each kept repository contributes 8
snapshots between 2021-04-02 and 2025-10-11, which gives 160 queries and 27,208 file rows. The horizon is 12 months:
11.5% of rows have a latent defect. A leakage check rebuilt 50 of 50 sampled queries from clean clones and matched
every feature within 1e-6.

**Labels and inputs.** Pain is the number of latent defects traced to the file by SZZ plus
0.25 log1p(churn / median churn), turned into a grade from 0 to 4 within each snapshot (grade counts 17,913,
4,599, 2,777, 1,390 and 529). The model reads 46 features in 7 families (static structure, evolution, ownership,
defect history, impact, tests and self-admitted debt) plus 43 within-repository percentile copies: 89 inputs.

**Baselines.** The test set is the last snapshot of each repository (20 queries). NDCG@10 with 95% intervals from
a bootstrap over repositories:

| Ranking | NDCG@10 |
|---|---|
| Hotspot (recent churn times complexity) | 0.718 [0.632, 0.801] |
| Total complexity | 0.708 [0.619, 0.793] |
| Lines of code | 0.701 [0.624, 0.776] |
| Pointwise XGBoost | 0.700 (leave one repository out), 0.685 (temporal) |
| Random forest | 0.695 (leave one repository out), 0.651 (temporal) |
| Prior bugs | 0.659 |
| Random | 0.189 (temporal), 0.185 (leave one repository out) |

Hotspot is the strongest baseline: precision@10 0.515, recall at 20% of lines 0.242, Popt 0.575.

**Training.** An XGBoost LambdaMART ranker (`rank:ndcg`) was tuned with 50 Optuna trials (41 s on a GPU). The
chosen model is small: depth 3, learning rate 0.02, 145 rounds. The shipped version is monotone in recent commits,
recent churn, prior latent bugs and transitive dependents (and their percentiles), so more of any of these never
lowers a score.

| Protocol | NDCG@10 | Precision@10 | Recall@20% lines | Popt | vs hotspot, one-sided p |
|---|---|---|---|---|---|
| Leave one repository out | 0.725 [0.648, 0.801] | 0.525 | 0.204 | 0.538 | 0.337 |
| Temporal (last snapshot) | 0.732 [0.649, 0.811] | 0.520 | 0.239 | 0.548 | 0.184 |

**Calibration.** Risk comes from isotonic regression fitted on out-of-fold scores (leave one repository out). ECE
is 0.0105 [0.010, 0.066] against a 0.05 target; Brier 0.0729 against 0.1009 for the base rate (skill 0.277);
AUC-PR 0.501. The shipped calibrator is refitted on all 27,208 rows (96 thresholds, mean risk range width 0.084).

**Registry.** The production bundle is `ranker-2026.10.1` (tarball 48,254 bytes, sha256 `06144154eb1c...`). It
passed the real-model, integrity, incumbent, calibration, leakage (50 queries) and golden (3 of 3 repositories)
gates. The hotspot significance gate failed (p 0.337) and was waived with a recorded reason. Golden top files:
click `src/click/core.py` Critical 97, starlette `starlette/responses.py` Critical 98, requests
`src/requests/models.py` Critical 95.

**Speed.** On a 12-thread laptop with the production model, a fresh analysis of pallets/click (32 files, 584
commits) took 3.9 to 5.8 s and of psf/requests (22 files) 3.2 s, about 13 s from typing the command to the
result. Reopening a saved analysis took 4 to 5 s end to end. The first run after an install is slower (about a
minute in rehearsal) while Python compiles and Windows Defender scans the new files.

## Limitations

- The ranker does not significantly beat the hotspot heuristic (p 0.337 and 0.184), and its Popt is lower.
- Only 20 test queries, so every interval is wide.
- Labels undercount bugs: keyword detection of bug fixes has a recall of about 0.43, and 5 repositories have
  keyword labels only.
- Risk is calibrated on all repositories pooled, so the most extreme files can read 100% (click's `core.py`).
- Priority is relative to the repository: a Critical file in a clean repository may be healthier than a Medium
  file elsewhere.
- Python only, and history is limited to the last 3 years and 20,000 commits.
- Example files of fastapi and pylint make up 45% of the rows.
- Rename detection differs between git versions, so a rebuilt feature can differ slightly on another machine.
- The model bundle is a release asset of this private repository, so the action in another repository needs
  read access to its releases, or it falls back to the demo model and says so.
- AI advice needs `ANTHROPIC_API_KEY` and sends the file's scores, reasons and up to 300 lines of its code to
  Anthropic.

## After V1

- A web dashboard (Next.js) for browsing analyses.
- The API and worker services with Postgres and Redis, Docker Compose and deployment.
- Experiments: feature family ablation, horizon and parameter sensitivity, figures and a full results report.
- Browser end-to-end tests and performance checks.

## Repository layout

| Path | What |
|---|---|
| `packages/core` | `tdp_core`: mining, parsing, features, labels, model runtime, planning |
| `packages/platform` | `tdp_platform`: database models, settings, observability |
| `services/api`, `services/worker` | FastAPI service and Celery workers |
| `cli` | the `entropiq` and `tdp` commands |
| `ml` | `tdp-ml`: dataset, training, evaluation, registry |
| `models/production.json` | pointer to the production model bundle |
| `handoff/` | one note per finished implementation step |

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy packages/core
```

Build work follows [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). A five-minute demo script is in
[docs/DEMO.md](docs/DEMO.md).
