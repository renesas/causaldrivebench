# Sanity Check Report: RecogDrive

**Run name:** `recogdrive_20260428_162305`  
**Run dir:** `/datadrive/datasets/cdb_outputs_20260429/recogdrive_20260428_162305`  
**Generated:** 2026-04-29T19:40:08.265417+00:00  

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
| Samples with prompts | 815 |
| Samples with outputs | 815 |
| **Complete (✓)** | **815** (?) |
| Incomplete (partial) | 0 |
| Missing outputs | 0 |
| Questions expected | 7342 |
| Questions written | 7342 (?) |
| Error outputs | 0 |

### Trajectory Output Status

| Metric | Value |
|--------|-------|
| Samples with traj prompts | 815 |
| **Complete (✓)** | **815** |
| Incomplete | 0 |
| Missing | 0 |
| Traj questions written | 1247 |
| Traj error outputs | 0 |

### Issues

_No issues found._

## Evaluation Metrics (from run report)

**Overall accuracy:** 54.2%  (3883/7158 correct)

### Accuracy by QA Type (Source)

| QA Type | Accuracy | Correct | N |
|---------|----------|---------|---|
| Ladder | 69.9% | 1626 | 2326 |
| Dormant | 48.6% | 705 | 1452 |
| Distractor | 45.9% | 1552 | 3380 |

### Accuracy by Rung (PCH Layer)

| Rung | Accuracy | Correct | N |
|------|----------|---------|---|
| R0 | 57.7% | 703 | 1218 |
| R1 | 38.2% | 705 | 1846 |
| R2 | 59.2% | 1018 | 1718 |
| R3 | 61.3% | 1457 | 2376 |

### Accuracy by Question Type

| Question Type | Accuracy | Correct | N |
|--------------|----------|---------|---|
| AB | 63.2% | 301 | 476 |
| AR | 100.0% | 4 | 4 |
| BAS | 93.3% | 14 | 15 |
| CA | 100.0% | 39 | 39 |
| CB | 43.8% | 60 | 137 |
| CDE | 59.6% | 28 | 47 |
| CH0_CA | 100.0% | 4 | 4 |
| CH0_CA_ | 100.0% | 1 | 1 |
| CH0_NIE | 100.0% | 1 | 1 |
| CH1_CDE | 100.0% | 1 | 1 |
| CH1_CDE_ | 0.0% | 0 | 1 |
| CH1_NDE | 0.0% | 0 | 1 |
| CH2_NIE | 100.0% | 1 | 1 |
| CI | 29.1% | 366 | 1256 |
| CL0_CB | 50.0% | 1 | 2 |
| CL0_NC | 50.0% | 1 | 2 |
| CL1_CB | 0.0% | 0 | 1 |
| CL2_NC | 0.0% | 0 | 1 |
| CO0_BAS | 0.0% | 0 | 1 |
| CO0_CDE | 100.0% | 6 | 6 |
| CO0_CR | 66.7% | 6 | 9 |
| CO0_CaI | 100.0% | 1 | 1 |
| CO0_CoI | 50.0% | 1 | 2 |
| CO0_NDE | 0.0% | 0 | 1 |
| CO0_NIE | 100.0% | 2 | 2 |
| CO1_BAS | 100.0% | 1 | 1 |
| CO1_NDE | 100.0% | 1 | 1 |
| CO2_CoI | 100.0% | 1 | 1 |
| CR | 54.6% | 345 | 632 |
| CU0_BAS | 100.0% | 2 | 2 |
| CU0_CR | 0.0% | 0 | 4 |
| CU0_CR_ | 0.0% | 0 | 1 |
| CU0_CoI | 100.0% | 9 | 9 |
| CU0_CoI_ | 66.7% | 2 | 3 |
| CU1_CR | 100.0% | 2 | 2 |
| CU2_BAS | 100.0% | 2 | 2 |
| CU3_CoI | 100.0% | 1 | 1 |
| CaI | 87.3% | 459 | 526 |
| CoI | 89.2% | 33 | 37 |
| D0_CR | 42.9% | 3 | 7 |
| D0_CR_ | 66.7% | 2 | 3 |
| D0_CaI | 100.0% | 9 | 9 |
| D0_CaI_ | 100.0% | 1 | 1 |
| D0_SC | 100.0% | 5 | 5 |
| D0_SC_ | 100.0% | 1 | 1 |
| D1_CR | 75.0% | 6 | 8 |
| D1_CR_ | 100.0% | 2 | 2 |
| D1_CaI | 100.0% | 3 | 3 |
| D1_CaI_ | 100.0% | 1 | 1 |
| D1_SC | 83.3% | 5 | 6 |
| D1_SC_ | 100.0% | 2 | 2 |
| D2_CR | 100.0% | 4 | 4 |
| D2_CaI | 100.0% | 3 | 3 |
| D2_SC | 66.7% | 4 | 6 |
| D3_CR | 100.0% | 1 | 1 |
| D4_CR | 100.0% | 1 | 1 |
| D6_CaI | 100.0% | 1 | 1 |
| DQ | 24.1% | 127 | 527 |
| DR-E | 50.0% | 1 | 2 |
| DR-Empty | 48.3% | 43 | 89 |
| EL | 61.7% | 277 | 449 |
| NC | 58.6% | 113 | 193 |
| NDE | 68.3% | 28 | 41 |
| NI | 54.7% | 582 | 1064 |
| NI1_contrast | 100.0% | 1 | 1 |
| NI1a | 0.0% | 0 | 1 |
| NI1b | 100.0% | 1 | 1 |
| NIE | 63.6% | 28 | 44 |
| Q | 73.9% | 17 | 23 |
| SC | 77.8% | 357 | 459 |
| WC | 57.8% | 558 | 966 |

### Answer Distribution

| Answer | GT Count | GT % | Pred Count | Pred % | Accuracy |
|--------|----------|------|------------|--------|----------|
|  | 0 | 0.0% | 21 | 0.3% | 0.0% |
| A | 660 | 9.2% | 739 | 10.3% | 90.8% |
| B | 236 | 3.3% | 273 | 3.8% | 82.6% |
| C | 123 | 1.7% | 103 | 1.4% | 73.2% |
| D | 164 | 2.3% | 55 | 0.8% | 32.3% |
| NO | 4649 | 64.9% | 2382 | 33.3% | 43.0% |
| YES | 1326 | 18.5% | 3585 | 50.1% | 71.3% |

**Answer bias:** 33.42%  
**Answer classes:** 6  

### Binary Classification Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 49.4% |
| Precision | 26.4% |
| Recall | 71.3% |
| F1 | 38.5% |
| TP/FP/FN/TN | 945/2640/381/2009 |
