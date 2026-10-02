# Sanity Check Report: Cosmos-Reason

**Run name:** `cosmos_reason_20260427_192914`  
**Run dir:** `/datadrive/datasets/cdb_outputs_20260429/cosmos_reason_20260427_192914`  
**Generated:** 2026-04-29T19:40:08.261415+00:00  

## Dataset: `causal_nuscenes`

### Benchmark Reference

| Metric | Value |
|--------|-------|
| Total samples | 815 |
| Samples with QA | ? |
| Total VQA questions | ? |

### VQA Output Status

| Metric | Value |
|--------|-------|
| Samples with prompts | 0 |
| Samples with outputs | 0 |
| **Complete (✓)** | **0** (?) |
| Incomplete (partial) | 0 |
| Missing outputs | 0 |
| Questions expected | 0 |
| Questions written | 0 (?) |
| Error outputs | 0 |

### Trajectory Output Status

| Metric | Value |
|--------|-------|
| Samples with traj prompts | 0 |
| **Complete (✓)** | **0** |
| Incomplete | 0 |
| Missing | 0 |
| Traj questions written | 0 |
| Traj error outputs | 0 |

### Issues

_No issues found._

## Evaluation Metrics (from run report)

**Overall accuracy:** 35.6%  (2545/7158 correct)

### Accuracy by QA Type (Source)

| QA Type | Accuracy | Correct | N |
|---------|----------|---------|---|
| Ladder | 59.0% | 1372 | 2326 |
| Dormant | 37.9% | 551 | 1452 |
| Distractor | 18.4% | 622 | 3380 |

### Accuracy by Rung (PCH Layer)

| Rung | Accuracy | Correct | N |
|------|----------|---------|---|
| R0 | 53.1% | 647 | 1218 |
| R1 | 18.7% | 346 | 1846 |
| R2 | 23.9% | 410 | 1718 |
| R3 | 48.1% | 1142 | 2376 |

### Accuracy by Question Type

| Question Type | Accuracy | Correct | N |
|--------------|----------|---------|---|
| AB | 51.0% | 243 | 476 |
| AR | 100.0% | 4 | 4 |
| BAS | 80.0% | 12 | 15 |
| CA | 89.7% | 35 | 39 |
| CB | 78.8% | 108 | 137 |
| CDE | 42.5% | 20 | 47 |
| CH0_CA | 75.0% | 3 | 4 |
| CH0_CA_ | 0.0% | 0 | 1 |
| CH0_NIE | 100.0% | 1 | 1 |
| CH1_CDE | 100.0% | 1 | 1 |
| CH1_CDE_ | 0.0% | 0 | 1 |
| CH1_NDE | 0.0% | 0 | 1 |
| CH2_NIE | 0.0% | 0 | 1 |
| CI | 3.8% | 48 | 1256 |
| CL0_CB | 100.0% | 2 | 2 |
| CL0_NC | 50.0% | 1 | 2 |
| CL1_CB | 100.0% | 1 | 1 |
| CL2_NC | 100.0% | 1 | 1 |
| CO0_BAS | 100.0% | 1 | 1 |
| CO0_CDE | 100.0% | 6 | 6 |
| CO0_CR | 55.6% | 5 | 9 |
| CO0_CaI | 100.0% | 1 | 1 |
| CO0_CoI | 100.0% | 2 | 2 |
| CO0_NDE | 0.0% | 0 | 1 |
| CO0_NIE | 50.0% | 1 | 2 |
| CO1_BAS | 100.0% | 1 | 1 |
| CO1_NDE | 0.0% | 0 | 1 |
| CO2_CoI | 100.0% | 1 | 1 |
| CR | 30.9% | 195 | 632 |
| CU0_BAS | 100.0% | 2 | 2 |
| CU0_CR | 50.0% | 2 | 4 |
| CU0_CR_ | 100.0% | 1 | 1 |
| CU0_CoI | 100.0% | 9 | 9 |
| CU0_CoI_ | 66.7% | 2 | 3 |
| CU1_CR | 100.0% | 2 | 2 |
| CU2_BAS | 100.0% | 2 | 2 |
| CU3_CoI | 100.0% | 1 | 1 |
| CaI | 72.0% | 379 | 526 |
| CoI | 91.9% | 34 | 37 |
| D0_CR | 57.1% | 4 | 7 |
| D0_CR_ | 33.3% | 1 | 3 |
| D0_CaI | 100.0% | 9 | 9 |
| D0_CaI_ | 100.0% | 1 | 1 |
| D0_SC | 100.0% | 5 | 5 |
| D0_SC_ | 100.0% | 1 | 1 |
| D1_CR | 37.5% | 3 | 8 |
| D1_CR_ | 0.0% | 0 | 2 |
| D1_CaI | 100.0% | 3 | 3 |
| D1_CaI_ | 0.0% | 0 | 1 |
| D1_SC | 66.7% | 4 | 6 |
| D1_SC_ | 100.0% | 2 | 2 |
| D2_CR | 50.0% | 2 | 4 |
| D2_CaI | 100.0% | 3 | 3 |
| D2_SC | 50.0% | 3 | 6 |
| D3_CR | 100.0% | 1 | 1 |
| D4_CR | 100.0% | 1 | 1 |
| D6_CaI | 100.0% | 1 | 1 |
| DQ | 23.2% | 122 | 527 |
| DR-E | 100.0% | 2 | 2 |
| DR-Empty | 86.5% | 77 | 89 |
| EL | 41.4% | 186 | 449 |
| NC | 30.6% | 59 | 193 |
| NDE | 29.3% | 12 | 41 |
| NI | 4.8% | 51 | 1064 |
| NI1_contrast | 100.0% | 1 | 1 |
| NI1a | 0.0% | 0 | 1 |
| NI1b | 100.0% | 1 | 1 |
| NIE | 59.1% | 26 | 44 |
| Q | 69.6% | 16 | 23 |
| SC | 82.6% | 379 | 459 |
| WC | 45.8% | 442 | 966 |

### Answer Distribution

| Answer | GT Count | GT % | Pred Count | Pred % | Accuracy |
|--------|----------|------|------------|--------|----------|
|  | 0 | 0.0% | 22 | 0.3% | 0.0% |
| A | 660 | 9.2% | 529 | 7.4% | 72.4% |
| B | 236 | 3.3% | 224 | 3.1% | 67.8% |
| C | 123 | 1.7% | 178 | 2.5% | 82.1% |
| D | 164 | 2.3% | 233 | 3.3% | 81.7% |
| NO | 4649 | 64.9% | 506 | 7.1% | 9.2% |
| YES | 1326 | 18.5% | 5466 | 76.4% | 94.0% |

**Answer bias:** 59.7%  
**Answer classes:** 6  

### Binary Classification Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 28.0% |
| Precision | 22.8% |
| Recall | 94.0% |
| F1 | 36.7% |
| TP/FP/FN/TN | 1246/4220/80/429 |
