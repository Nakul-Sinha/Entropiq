## Technical debt report

Commit `0123456789ab`, model `dev-20260115T093005Z`, analysed 2026-01-15 09:30 UTC.

5 files ranked: 1 Critical, 2 High, 1 Medium, 1 Low. Fixing 1 Critical file is estimated at 12.5 hours.

| File | Priority | Risk (12 mo) | Top reason |
|---|---|---|---|
| `src/payments/processor.py` | 95 Critical | 64% | Very high recent churn: 831 recency-weighted lines added or deleted (top 1% in repo) |
| `src/billing/invoice.py` | 81 High | 41% | Central module in the import graph (top 9% in repo by PageRank) |
| `src/legacy/old utils.py` | 81 High | 35% | Heavy duplication: 32% of its code is duplicated (top 3% in repo) |
| `src/pkg/__init__.py` | 60 Medium | 18% | Admitted debt: 3 TODO, FIXME or HACK style comments (top 12% in repo) |
| `setup.py` | 12 Low | 3% | - |
