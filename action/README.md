# Entropiq GitHub Action

The action ranks the files a pull request changes by how much maintenance pain each is likely to cause in the
next year, using the Entropiq CLI (`tdp diff`). It keeps one comment on the pull request with a short table
(band, priority, risk and top reason per changed file) and a one-line summary, writes the same report to the
job summary, can fail the job when a changed file is High or Critical, and can upload a SARIF report to code
scanning.

## Requirements

- A Linux or macOS runner. The action stops with an error on Windows runners.
- `actions/checkout` with `fetch-depth: 0`, so the history and the base commit are present. A shallow
  checkout gives a warning and a poor analysis.
- Permissions: `contents: read` and `pull-requests: write` (plus `issues: write` on some setups) for the
  comment; `security-events: write` and `actions: read` as well when `sarif` is set.
- This repository is private. Another repository that uses the action needs Actions access to this
  repository (Settings, Actions, General, Access), and a `github-token` that can read its releases;
  without that it gets the demo model and the comment says so.

## Inputs

| Input | Default | Meaning |
| --- | --- | --- |
| `base` | required | Branch, tag or commit to compare with. The report covers the files changed in `BASE...HEAD`. On pull requests pass `${{ github.event.pull_request.base.sha }}`. |
| `comment` | `true` | Post or update one pull request comment (`true` or `false`). |
| `sarif` | empty | Path of a SARIF file to write and upload to code scanning. Empty skips SARIF. |
| `fail-on` | `none` | `none`, `high` (fail when a changed file is High or Critical) or `critical`. |
| `model-version` | empty | Model release such as `ranker-2026.10.1`. Empty uses `models/production.json` at the action ref. |
| `github-token` | `${{ github.token }}` | Token that downloads the model release and writes the comment. |

## Outputs

| Output | Meaning |
| --- | --- |
| `model-source` | `release` when the published model was used, `dev` when the action fell back to the demo model. |
| `model-version` | Version of the model that ranked the files. |
| `highest-band` | Highest band among the changed files: `Critical`, `High`, `Medium`, `Low` or `none`. |
| `comment-id` | Id of the pull request comment, empty when no comment was written. |
| `report-json` | Path of the `tdp diff --format json` report on the runner. |

## Example workflow

```yaml
name: entropiq

on:
  pull_request:

permissions:
  contents: read
  pull-requests: write
  security-events: write
  actions: read

concurrency:
  group: entropiq-${{ github.event.pull_request.number }}
  cancel-in-progress: true

jobs:
  report:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
        with:
          fetch-depth: 0
      - uses: Nakul-Sinha/Entropiq/action@main
        with:
          base: ${{ github.event.pull_request.base.sha }}
          fail-on: critical
          sarif: entropiq.sarif
```

## How the model is chosen

1. The action reads `models/production.json` at the action ref (or uses `model-version` when it is set).
2. It downloads the asset `<version>.tar.gz` of the release `model-<version>` of this repository with `gh`
   and `github-token`. A release asset URL in the `uri` field of `production.json` names the release
   directly.
3. It checks the archive's sha256 against `production.json`. For any other `model-version` it uses the
   release's `<version>.tar.gz.sha256` file instead.
4. The verified archive is cached with `actions/cache` under the key `tdp-model-<version>` and unpacked fresh
   on every run; a cached file is checked again before it is used.
5. When the release cannot be downloaded, verified or unpacked, the action falls back to the demo model and the
   comment says why. `fail-on` is not applied to demo rankings, because they are not real risk.

## The comment

There is one comment per pull request. The action finds it through the hidden marker `<!-- tdp-report -->`
at the start of the body together with a bot author (or the token's own user) and updates it in place, so a
person who quotes the marker never has their comment edited. When the comment cannot be written (a fork, or a
missing permission) the action only warns. The job summary always gets the report.

## SARIF

With `sarif` set, `tdp report --format sarif` writes results for the whole repository at High and above, and
`github/codeql-action/upload-sarif` uploads them with the category `entropiq`. Code scanning shows which alerts
are new in the pull request. The upload never fails the job, because private repositories need GitHub code
security for code scanning.

## Publishing a model release

Bundles are built and promoted on a workstation (`tdp-ml promote`), never in CI. After a promotion, publish
the bundle tarball and its checksum as a release of this repository:

```bash
V=ranker-2026.10.1
T="$TDP_MODELS_DIR/$V.tar.gz"
H=$(sha256sum "$T" | cut -d' ' -f1)
printf '%s  %s\n' "$H" "$V.tar.gz" > "$V.tar.gz.sha256"
gh release create "model-$V" "$T" "$V.tar.gz.sha256" --repo Nakul-Sinha/Entropiq --target main \
  --title "Entropiq model $V" --notes "Production ranking model $V. The archive's sha256 is $H."
gh release view "model-$V" --repo Nakul-Sinha/Entropiq --json assets --jq '.assets[].url'
```

Then set `uri` in `models/production.json` to the asset URL
(`https://github.com/Nakul-Sinha/Entropiq/releases/download/model-<version>/<version>.tar.gz`) through a pull
request. `tdp-ml promote` writes a local path there; the action still finds `model-<version>` by convention,
but a test of the tracked pointer fails until the URL is set back.

## Using the released model locally

```bash
V=ranker-2026.10.1
gh release download "model-$V" --repo Nakul-Sinha/Entropiq --pattern "$V.tar.gz" --dir /tmp/entropiq-model
mkdir -p ~/.entropiq/models
tar -xzf "/tmp/entropiq-model/$V.tar.gz" -C ~/.entropiq/models
tdp diff . --base=main --model-dir ~/.entropiq/models/$V
```

The CLI also finds bundles by version under `~/.entropiq/models`, or under `TDP_MODELS_DIR` when that is set.

## Runtime

On this repository a run takes about 15 seconds on a GitHub-hosted runner: about 5 seconds to install the
CLI and about 10 seconds for the analysis, which grows with the size of the repository. The SARIF upload can
add up to a minute. A second run in the same job reuses the environment, the cached model and the saved
analysis.
