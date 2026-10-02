# CausalDriveBench Evaluation — Combined Statistics

_Generated: 2026-04-29T19:40:08.266413+00:00_

## Overview Table

| Model | Run Name | VQA Complete | VQA% | Traj Complete | Overall Acc | Active (Ladder) | Dormant | Distractor | R0 | R1 | R2 | R3 | Issues | T1 Reports | T3 Report |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AlpaMayo-1.5 | `alpamayo_1_5_20260428_081120` | 568/? | ? | 1759 | 37.1% | 47.0% | 31.2% | 33.2% | 36.2% | 20.9% | 39.9% | 48.0% | 2 | ✓ | ✓ |
| Cosmos-Reason | `cosmos_reason_20260427_192914` | 0/? | ? | 0 | 35.6% | 59.0% | 37.9% | 18.4% | 53.1% | 18.7% | 23.9% | 48.1% | 0 | ✓ | ✓ |
| Cosmos-Reason | `cosmos_reason_20260429_054346` | 815/? | ? | 815 | 35.6% | 59.0% | 37.9% | 18.4% | 53.1% | 18.7% | 23.9% | 48.1% | 0 | ✓ | ✓ |
| InternVL-3.5-8B | `intern_vl3_5_8b_20260428_174406` | 815/? | ? | 815 | 62.9% | 64.7% | 56.7% | 64.3% | 54.2% | 51.7% | 78.3% | 64.9% | 0 | ✓ | ✓ |
| OmniQ (OmniDrive) | `omni_q_20260427_210155` | 0/? | ? | 0 | 24.1% | 30.9% | 20.6% | 20.9% | 6.8% | 25.9% | 16.5% | 37.0% | 0 | ✓ | ✓ |
| OmniQ (OmniDrive) | `omni_q_20260428_165719` | 815/? | ? | 0 | 24.1% | 30.9% | 20.6% | 20.9% | 6.8% | 25.9% | 16.5% | 37.0% | 0 | ✓ | ✓ |
| OpenREAD | `openread_20260428_120025` | 815/? | ? | 815 | 63.8% | 69.8% | 60.6% | 60.8% | 70.0% | 49.0% | 79.0% | 61.0% | 0 | ✓ | ✓ |
| Qwen2.5-VL | `qwen25vl_20260425_072054` | 815/? | ? | 815 | 73.1% | 60.2% | 72.8% | 82.3% | 71.3% | 81.0% | 83.9% | 60.2% | 0 | ✓ | ✓ |
| Qwen2.5-VL | `qwen25vl_20260428_163639` | 0/? | ? | 810 | — | — | — | — | — | — | — | — | 11 | ✗ | ✗ |
| Qwen3-VL-8B | `qwen3_vl_8b_20260428_174418` | 815/? | ? | 815 | 66.0% | 56.7% | 64.1% | 73.4% | 53.2% | 65.0% | 82.7% | 61.4% | 0 | ✓ | ✓ |
| RecogDrive | `recogdrive_20260427_195115` | 0/? | ? | 0 | 54.2% | 69.9% | 48.6% | 45.9% | 57.7% | 38.2% | 59.2% | 61.3% | 0 | ✓ | ✓ |
| RecogDrive | `recogdrive_20260428_162305` | 815/? | ? | 815 | 54.2% | 69.9% | 48.6% | 45.9% | 57.7% | 38.2% | 59.2% | 61.3% | 0 | ✓ | ✓ |
| SafeAuto | `safeauto_20260424_193636` | 815/? | ? | 0 | 49.4% | 56.6% | 41.8% | 47.6% | 51.5% | 43.2% | 56.7% | 47.9% | 0 | ✓ | ✓ |
| WiseAD | `wisead_8img_20260429_112651` | 0/? | ? | 815 | — | — | — | — | — | — | — | — | 0 | ✗ | ✗ |

## VQA Sanity Details (per model, causal_nuscenes)

| Model | Run | Expected Q | Written Q | Errors | Partial | Missing |
|-------|-----|-----------|----------|--------|---------|---------|
| AlpaMayo-1.5 | `alpamayo_1_5_20260428_081120` | 4887 | 4870 | 0 | 0 | 2 |
| Cosmos-Reason | `cosmos_reason_20260427_192914` | 0 | 0 | 0 | 0 | 0 |
| Cosmos-Reason | `cosmos_reason_20260429_054346` | 7342 | 7342 | 0 | 0 | 0 |
| InternVL-3.5-8B | `intern_vl3_5_8b_20260428_174406` | 7285 | 7285 | 0 | 0 | 0 |
| OmniQ (OmniDrive) | `omni_q_20260427_210155` | 0 | 0 | 0 | 0 | 0 |
| OmniQ (OmniDrive) | `omni_q_20260428_165719` | 7290 | 7290 | 0 | 0 | 0 |
| OpenREAD | `openread_20260428_120025` | 7285 | 7285 | 0 | 0 | 0 |
| Qwen2.5-VL | `qwen25vl_20260425_072054` | 7285 | 7285 | 0 | 0 | 0 |
| Qwen2.5-VL | `qwen25vl_20260428_163639` | 0 | 0 | 0 | 0 | 0 |
| Qwen3-VL-8B | `qwen3_vl_8b_20260428_174418` | 7285 | 7285 | 0 | 0 | 0 |
| RecogDrive | `recogdrive_20260427_195115` | 0 | 0 | 0 | 0 | 0 |
| RecogDrive | `recogdrive_20260428_162305` | 7342 | 7342 | 0 | 0 | 0 |
| SafeAuto | `safeauto_20260424_193636` | 7285 | 7285 | 0 | 0 | 0 |
| WiseAD | `wisead_8img_20260429_112651` | 0 | 0 | 0 | 0 | 0 |

## Trajectory Sanity Details

| Model | Run | Traj Prompts | Traj Outputs | Traj Errors | Complete |
|-------|-----|-------------|-------------|-------------|---------|
| AlpaMayo-1.5 | `alpamayo_1_5_20260428_081120` | 1776 | 1776 | 0 | 1759 |
| Cosmos-Reason | `cosmos_reason_20260429_054346` | 1247 | 1247 | 0 | 815 |
| InternVL-3.5-8B | `intern_vl3_5_8b_20260428_174406` | 1992 | 1992 | 0 | 815 |
| OpenREAD | `openread_20260428_120025` | 1992 | 1992 | 0 | 815 |
| Qwen2.5-VL | `qwen25vl_20260425_072054` | 835 | 835 | 0 | 815 |
| Qwen2.5-VL | `qwen25vl_20260428_163639` | 815 | 820 | 11 | 810 |
| Qwen3-VL-8B | `qwen3_vl_8b_20260428_174418` | 1992 | 1992 | 0 | 815 |
| RecogDrive | `recogdrive_20260428_162305` | 1247 | 1247 | 0 | 815 |
| WiseAD | `wisead_8img_20260429_112651` | 1247 | 1247 | 0 | 815 |

## HTML Reports

Root index: `/datadrive/datasets/cdb_reports_20260429/index.html`

| Model | HTML Report |
|-------|------------|
| AlpaMayo-1.5 | `/datadrive/datasets/cdb_reports_20260429/alpamayo_1_5_20260428_081120/index.html` |
| Cosmos-Reason | `/datadrive/datasets/cdb_reports_20260429/cosmos_reason_20260427_192914/index.html` |
| Cosmos-Reason | `/datadrive/datasets/cdb_reports_20260429/cosmos_reason_20260429_054346/index.html` |
| InternVL-3.5-8B | `/datadrive/datasets/cdb_reports_20260429/intern_vl3_5_8b_20260428_174406/index.html` |
| OmniQ (OmniDrive) | `/datadrive/datasets/cdb_reports_20260429/omni_q_20260427_210155/index.html` |
| OmniQ (OmniDrive) | `/datadrive/datasets/cdb_reports_20260429/omni_q_20260428_165719/index.html` |
| OpenREAD | `/datadrive/datasets/cdb_reports_20260429/openread_20260428_120025/index.html` |
| Qwen2.5-VL | `/datadrive/datasets/cdb_reports_20260429/qwen25vl_20260425_072054/index.html` |
| Qwen2.5-VL | _not generated_ |
| Qwen3-VL-8B | `/datadrive/datasets/cdb_reports_20260429/qwen3_vl_8b_20260428_174418/index.html` |
| RecogDrive | `/datadrive/datasets/cdb_reports_20260429/recogdrive_20260427_195115/index.html` |
| RecogDrive | `/datadrive/datasets/cdb_reports_20260429/recogdrive_20260428_162305/index.html` |
| SafeAuto | `/datadrive/datasets/cdb_reports_20260429/safeauto_20260424_193636/index.html` |
| WiseAD | _not generated_ |