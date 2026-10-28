### Fix plan

3 files chosen within the budget of 40.0 hours, 27.5 hours in total.

| Step | File | Hours | Reason |
|---|---|---|---|
| 1 | `src/billing/invoice.py` | 7.5 | fix before src/payments/processor.py, which depends on it |
| 2 | `src/payments/processor.py` | 12.5 | best return on effort among the remaining files |
| 3 | `src/legacy/old utils.py` | 7.5 | fix before src/pkg/\_\_init\_\_.py, which depends on it |
