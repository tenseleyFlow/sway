# sway compare — 3 runs

## Runs

| label | finished_at |
|---|---|
| v1 | 2026-01-01T12:00:30+00:00 |
| v2 | 2026-01-01T12:30:30+00:00 |
| v3 | 2026-01-01T13:00:30+00:00 |

## Scores

| probe | v1 | v2 | v3 | Δ v1→v2 | Δ v2→v3 |
|---|---|---|---|---|---|
| delta_kl | 0.80 | 0.82 | 0.65 | +0.02 | -0.17 |
| leakage | — | 0.90 | 0.88 | — | -0.02 |
| section_internalization | 0.70 | 0.72 | — | +0.02 | — |
| **composite** | 0.75 | 0.81 | 0.72 | +0.06 | -0.09 |

## Regressions (≥0.10 drop vs previous run)

- **delta_kl** — `-0.170`
