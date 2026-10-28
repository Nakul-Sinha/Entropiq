"""The offline pipeline that turns the training repositories into ``data/dataset.parquet``.

Stages (each one a ``tdp-ml`` command; outputs that exist are skipped unless ``--force``):

1. ``manifest``: pin each repository of ``ml/configs/repos.yaml`` to the sha of its mirror's default branch,
   and store the label settings of ``ml/configs/labels.yaml`` (every later stage reads them from the
   manifest, so one dataset build has one source of settings).
2. ``mine``: the whole history at the manifest sha, with that sha's ``.mailmap`` (``commits`` and
   ``file_changes`` tables).
3. ``issues``: GitHub issues of the crawled repositories (cached once per repository; repositories marked
   ``issues: false`` are keyword-only).
4. ``bugfix``: two bug-fix tables, issue-aware (labels) and keyword-only (features, as at inference).
5. ``snapshots``: the leakage-safe snapshot schedule (section 15.5); the newest ``max_per_repo`` are kept
   with their original indexes, and older ones are dropped until the fixes SZZ must trace fit the budget.
   The plan file lists each kept snapshot's window end and the fixes to trace.
6. ``szz``: SZZ for the issue-aware fixes inside a kept label window (labels) and for the newest prior
   keyword fixes of every kept snapshot (the defect-history features).
7. ``features``: ``build_features`` per kept snapshot, from a history mined with the snapshot's own
   ``.mailmap``; the path lineage is stored next to the feature table.
8. ``labels``: ``make_labels`` per kept snapshot with both reachable sets.
9. ``dataset``: the inclusion criteria from exact counts (failing repositories are reported, not deleted),
   then features and labels joined per query with a stable ``qid``, validated and written.

``stats`` reports counts, grade distributions, runtimes and disk usage, and ``check-leakage`` rebuilds
sampled queries in clones truncated at the snapshot and compares them with the stored rows.

Data layout under ``tdp_core.paths.data_dir()`` (see :class:`tdp_ml.pipeline.layout.DataLayout`)::

    manifest.json
    mirrors/<slug>.git
    mined/<slug>/commits.parquet, file_changes.parquet
    issues/<slug>.parquet
    bugfix/<slug>.parquet, <slug>.keyword.parquet
    snapshots/<slug>.json, <slug>.plan.json
    szz/<slug>.parquet, <slug>.keyword.parquet
    features/<slug>/<idx>.parquet, <idx>.lineage.parquet, <idx>.json
    labels/<slug>/<idx>.parquet
    dataset.parquet, dataset.json, stats.json
    cache/static, cache/szz/<slug>, cache/tmp
    runs/<slug>/<stage>.json, runs/check-leakage.json
"""
