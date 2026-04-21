# sway report

**Overall:** 0.65 (`healthy`)  
**Base:** `HuggingFaceTB/SmolLM2-135M-Instruct`  
**Adapter:** `runs/adapter/v0003`  
**Wall:** 2.50s  
**Determinism:** `best_effort` (seed=0)  

## Components

| category | score | weight | |
|---|---:|---:|---|
| adherence | 0.87 | 0.30 |  |
| attribution | 0.30 | 0.35 |  |
| calibration | 0.50 | 0.20 |  |
| ablation | 0.00 | 0.15 |  |
| baseline | 1.00 | 0.00 | (informational, weight=0) |

## Probes

| name | kind | verdict | score | raw | z | duration | note |
|---|---|---|---:|---:|---:|---:|---|
| dk | `delta_kl` | pass | 0.87 | 0.456 | +5.12σ | 0.12s | mean js=0.4560, z=+5.12σ vs null |
| sis | `section_internalization` | fail | 0.30 | 0.012 | +0.50σ | 0.46s | 1/4 sections cleared effective_sis≥0.05 |
| lk | `leakage` | skip | — | — | — | 0.00s | no PROSE sections to test for leakage |
| ablation | `adapter_ablation` | error | — | — | — | 0.00s | backend does not implement ScalableDifferentialBackend |

## Top findings

- sis (section_internalization) failed: 1/4 sections cleared effective_sis≥0.05
- ablation score is 0.00 — below the noise threshold
