# Entropiq five-minute demo

A rehearsed script for showing the command line on the demo laptop. Every command below was run on this laptop
with the installed `entropiq` command and the production model `ranker-2026.10.1`; the excerpts are real output,
shortened.

## Before the audience arrives

1. Open Windows Terminal with PowerShell and maximise it. Keep it at 140 columns or wider: narrower windows wrap
   the `/diff` and `/trend` tables into tall, thin columns.
2. Go to the folder of prefetched repositories and check the setup:

   ```powershell
   cd C:\Users\nakul\entropiq-data\data\mirrors
   entropiq --version          # entropiq 0.1.0
   entropiq model              # Kind: production model, Version: ranker-2026.10.1
   entropiq history            # 3 saved analyses: requests 3c7baaa, click 2247b35, click 1a994ad
   ```

   `entropiq model` must say `production model` and `Chosen by C:\Users\nakul\.entropiq\models\production.json`.
   If it says `demo model`, the pointer or the bundle folder in `C:\Users\nakul\.entropiq\models` is missing.
3. Fallbacks if `entropiq` is not found: the full path
   `C:\Users\nakul\AppData\Roaming\SPB_Data\.local\bin\entropiq.exe`, or `uv run entropiq` from a checkout of the
   repository (then pass the mirror's full path to `/analyze`).
4. Run `entropiq analyze pallets__click.git --top 3` once, so Windows has scanned the program files and the first
   live run is fast.

## The script

| Time | Type | Point out |
|---|---|---|
| 0:00 | `entropiq` | The logo line names the model: `model ranker-2026.10.1 · production model`. |
| 0:20 | `/analyze pallets__click.git --fresh` | The stages and the summary card (below). |
| 1:00 | `/files`, then `n`, then `q` | Priority is a percentile within the repository; risk is a calibrated 12-month chance of a latent defect. |
| 1:30 | `/file 1` | Reasons, contribution by feature family, most complex functions, connections, commit activity, cost breakdown. |
| 2:15 | `/graph`, then `/tree` | Central files and the 11-file import cycle around `core.py`; the folder tree with the highest band per folder. |
| 2:45 | `/plan --budget 10` | Picks files by return on effort, not just by rank. |
| 3:15 | `/whatif 1 --all` | Simulated fixes and how the risk changes. |
| 3:40 | `/open 3c7baaa`, then `/diff --base HEAD~30` | A saved analysis opens instantly; `/diff` ranks only the files a change touched. |
| 4:10 | `/open 2247b35`, then `/trend` | Two analyses of click, five months apart. |
| 4:30 | `/model`, `/export md -o - --top 5`, `/quit` | Training metrics; a Markdown report ready for a pull request. |

Use the commit prefixes with `/open` (not the `/history` numbers), so the steps work even if the order of saved
analyses changes. If the live `/analyze ... --fresh` feels slow, run `/analyze pallets__click.git` without
`--fresh`: it reuses the saved analysis at once.

Optional extras if time allows: `tdp files pallets__click.git --band critical --format json` (machine output),
`/advise 1` (without `ANTHROPIC_API_KEY` it shows a panel explaining how to turn advice on), and
`/analyze https://github.com/pallets/click` (needs the network; it makes a lean mirror first).

## What the audience sees

### 0:20 `/analyze pallets__click.git --fresh`

```text
Analysing pallets__click @ 2247b35
[ 25%] Mined 584 commits and found 68 bug fixes (0.9 s)
[ 50%] Parsed the source files (2.0 s)
[ 70%] Computed the file features (1.6 s)
[ 90%] Explained 32 files (1.3 s)
[ 95%] Planned 5 fixes within 40 hours (0.0 s)
Analysis finished in 5.8 s
┌─ pallets__click @ 2247b35 ─────────────────────────────────────────┐
│ 32 files analysed in 5.8 s                                         │
│ ● Critical 3   ● High 5   ● Medium 8   ● Low 16                    │
│ Estimated Critical debt: 65.7 h (52.6-78.9 h)                      │
│ Model ranker-2026.10.1 · trained                                   │
└────────────────────────────────────────────────────────────────────┘
1   src/click/core.py           ● Critical 97   100%   Very complex file: total cyclomatic complexity of 662 (top 1% in repo)
2   src/click/_termui_impl.py   ● Critical 94    55%   Complex file: total cyclomatic complexity of 180 (top 6% in repo)
3   src/click/types.py          ● Critical 91    55%   Very large file: 1,148 lines of code (top 3% in repo)
```

Say: the whole analysis (mining, parsing, features, scoring, explanations, plan) takes about 4 to 6 seconds on this
laptop. The 100% risk on `core.py` is the calibration ceiling for the most extreme file, not a certainty.

### 1:30 `/file 1`

```text
● Critical 97  ·  rank 1 of 32  ·  risk 100% (85-100%) in the next 12 months
Estimated fix 37.2-55.8 h (46.5 h)  ·  not in the fix plan (40.0 h budget)

Why it ranks here
↑ Very complex file: total cyclomatic complexity of 662 (top 1% in repo)
↑ Very busy lately: 20.1 recency-weighted commits (top 1% in repo)
↑ Many past bugs: 18 past bugs traced back to its code (top 1% in repo)

Contribution by feature family
Static structure                │████████████  +1.84
Evolution                       │██            +0.36
Defect history                  │█             +0.19
```

Say: the reasons come from SHAP values of the ranker, so they explain this model's score, not a fixed rule.

### 2:45 `/plan --budget 10`

```text
Fix plan for pallets__click @ 2247b35, budget 10.0 h
#   File and reason                                   Group       Hours   Return   Benefit so far
1   src/click/parser.py                                   1   1.9-2.9 h    0.675   █░░░░░░░░░   5%
    best return on effort among the remaining files
2   src/click/_termui_impl.py                             2   5.2-7.7 h    0.477   ██░░░░░░░░  16%
2 files, 8.9 h of the 10.0 h budget (estimate 7.1-10.6 h), benefit 4.70, 16% of the benefit of fixing all
16 files with priority 50 or more.
```

Say: `core.py` is the top file but costs about 46 hours, so a 10-hour budget goes to cheaper files with a better
return. The plan also orders files so that a dependency is fixed before the files that import it.

### 3:15 `/whatif 1 --all`

```text
                   Before          After           Change
Priority           ● Critical 97   ● Critical 97   no change
Risk (12 months)   100%            85%             ↓ 15 pts
cc_max             highest cyclomatic complexity                35    17.5
scc_size           files in its import cycle                    11       1
has_test_file      has a matching test file (1 yes, 0 no)        0       1
```

### 3:40 `/open 3c7baaa` and `/diff --base HEAD~30`

```text
✓ Opened psf__requests @ 3c7baaa
● Critical 2   ● High 3   ● Medium 6   ● Low 11
1   src/requests/models.py     ● Critical 95    64%   Very large file: 892 lines of code (top 1% in repo)
...
10 of 23 changed files are ranked: 2 Critical, 3 High, 4 Medium, 1 Low.
```

### 4:10 `/open 2247b35` and `/trend`

```text
#   Commit           Committed    Analysed           Critical   High   Files   Critical hours   Model
1   1a994ad          2026-05-28   2026-10-29 10:57          3      5      32           75.0 h   ranker-2026.10.1
2   2247b35 loaded   2026-10-05   2026-10-29 11:01          3      5      32           65.7 h   ranker-2026.10.1
Since the previous analysis: Critical files 3 to 3 (no change), Critical hours 75.0 h to 65.7 h (↓ 9.3 h)
```

### 4:30 `/model` and `/export md -o - --top 5`

```text
Version        ranker-2026.10.1
Kind           production model
Training rows  27,208
Features       89 inputs (extractor 1)
Metrics        ...
               lopo.hotspot_ndcg_10 0.7176
               lopo.ndcg_10 0.7246
```

```text
## Technical debt report
32 files ranked: 3 Critical, 5 High, 8 Medium, 16 Low. Fixing 3 Critical files is estimated at 65.7 hours.
| File | Priority | Risk (12 mo) | Top reason |
| `src/click/core.py` | 97 Critical | 100% | Very complex file: total cyclomatic complexity of 662 (top 1% in repo) |
```

Close with the honest result: the ranker matches the hotspot heuristic (NDCG@10 0.725 against 0.718, not a
significant gain), but it adds calibrated risk, reasons, costs and a budgeted plan that a heuristic does not give.

## Measured on this laptop

| Run | Analysis | From typing to result |
|---|---|---|
| pallets/click, fresh (32 files, 584 commits) | 3.9 to 5.8 s | about 13 s |
| psf/requests, fresh (22 files) | 3.2 s | about 13 s |
| Either one, saved analysis reused | | 4 to 5 s |

The end-to-end time includes starting Python and loading the model. The first run after installing the command is
slower (about a minute) while Windows scans the new files, which is why step 4 of the setup runs one analysis
early.

## Resetting after a rehearsal

Saved analyses live in `C:\Users\nakul\.entropiq\analyses`; leave them in place. A rehearsal that changes settings
writes `C:\Users\nakul\.entropiq\settings.json`; run `/settings reset` or delete that file. The shell keeps its
prompt history in `C:\Users\nakul\.entropiq\history`, which can be deleted for a clean start.
