# Paper Tables — Extracted Values

_Generated: 2026-04-29T19:40:09.898623+00:00_

## Table 1: Causal QA Accuracy (%)

| Model | Category | R0 | R1 | R2 | R3 | Active | Dormant | Distractor | Overall | N Questions | Run |
|-------|----------|----|----|----|----|--------|---------|------------|---------|------------|-----|
| OmniQ (OmniDrive) | driving_vla | 6.8% | 25.9% | 16.5% | 37.0% | 30.9% | 20.6% | 20.9% | 24.1% | 7212 | `omni_q_20260428_165719` |
| WiseAD | driving_vla | — | — | — | — | — | — | — | — | — | `—` |
| WiseAD | driving_vla | — | — | — | — | — | — | — | — | 0 | `wisead_8img_20260429_112651` |
| SafeAuto | driving_vla | 51.5% | 43.2% | 56.7% | 47.9% | 56.6% | 41.8% | 47.6% | 49.4% | 7285 | `safeauto_20260424_193636` |
| SimLingo | driving_vla | — | — | — | — | — | — | — | — | — | `—` |
| Orion | driving_vla | — | — | — | — | — | — | — | — | — | `—` |
| DriveMoE | driving_vla | — | — | — | — | — | — | — | — | — | `—` |
| ImpromptuVLA | driving_vla | — | — | — | — | — | — | — | — | — | `—` |
| RecogDrive | driving_vla | 57.7% | 38.2% | 59.2% | 61.3% | 69.9% | 48.6% | 45.9% | 54.2% | 7158 | `recogdrive_20260428_162305` |
| OpenREAD | driving_vla | — | — | — | — | — | — | — | — | — | `—` |
| OpenREAD | driving_vla | 70.0% | 49.0% | 79.0% | 61.0% | 69.8% | 60.6% | 60.8% | 63.8% | 7274 | `openread_20260428_120025` |
| AlpaMayo-1.5 | driving_vla | 36.2% | 20.9% | 39.9% | 48.0% | 47.0% | 31.2% | 33.2% | 37.1% | 4887 | `alpamayo_1_5_20260428_081120` |
| Cosmos-Reason | general_vlm | 53.1% | 18.7% | 23.9% | 48.1% | 59.0% | 37.9% | 18.4% | 35.6% | 7158 | `cosmos_reason_20260429_054346` |
| Qwen2.5-VL | general_vlm | 71.3% | 81.0% | 83.9% | 60.2% | 60.2% | 72.8% | 82.3% | 73.1% | 7285 | `qwen25vl_20260425_072054` |
| InternVL-3.5-8B | general_vlm | 54.2% | 51.7% | 78.3% | 64.9% | 64.7% | 56.7% | 64.3% | 62.9% | 7285 | `intern_vl3_5_8b_20260428_174406` |
| Qwen3-VL-8B | general_vlm | 53.2% | 65.0% | 82.7% | 61.4% | 56.7% | 64.1% | 73.4% | 66.0% | 7285 | `qwen3_vl_8b_20260428_174418` |
| Gemma-VL (E4B) | general_vlm | — | — | — | — | — | — | — | — | — | `—` |
| Gemma-VL (31B) | general_vlm | — | — | — | — | — | — | — | — | — | `—` |
| Gemma-VL (E2B) | general_vlm | — | — | — | — | — | — | — | — | — | `—` |
| Gemma-VL (26B-A4B) | general_vlm | — | — | — | — | — | — | — | — | — | `—` |
| Gemma-VL | general_vlm | — | — | — | — | — | — | — | — | — | `—` |
| Doe-1 | general_vlm | — | — | — | — | — | — | — | — | — | `—` |
| Qwen3-VL (fine-tuned) | finetuned | — | — | — | — | — | — | — | — | — | `—` |

## Table 2: Trajectory Evaluation (ADE in metres)

| Model | BTE-ADE ↑ | CTE-ADE-R2 ↓ | CTE-ADE-R3 ↓ | Run |
|-------|-----------|-------------|-------------|-----|
| OmniQ (OmniDrive) | — | — | — | `omni_q_20260428_165719` |
| WiseAD | — | — | — | `—` |
| WiseAD | 2.90 m | 3.50 m | 1.76 m | `wisead_8img_20260429_112651` |
| SafeAuto | — | — | — | `safeauto_20260424_193636` |
| SimLingo | — | — | — | `—` |
| Orion | — | — | — | `—` |
| DriveMoE | — | — | — | `—` |
| ImpromptuVLA | — | — | — | `—` |
| RecogDrive | 2.01 m | 3.69 m | 1.94 m | `recogdrive_20260428_162305` |
| OpenREAD | — | — | — | `—` |
| OpenREAD | 3.97 m | — | — | `openread_20260428_120025` |
| AlpaMayo-1.5 | 1.33 m | 1.81 m | 1.47 m | `alpamayo_1_5_20260428_081120` |
| Qwen3-VL (fine-tuned) | — | — | — | `—` |

## Model → Run Mapping

| Model ID (code) | Display Name | Category | Run Name |
|----------------|-------------|----------|---------|
| `alpamayo_1_5` | AlpaMayo-1.5 | driving_vla | `alpamayo_1_5_20260428_081120` |
| `cosmos_reason` | Cosmos-Reason | general_vlm | `cosmos_reason_20260429_054346` |
| `intern_vl3_5_8b` | InternVL-3.5-8B | general_vlm | `intern_vl3_5_8b_20260428_174406` |
| `omni_q` | OmniQ (OmniDrive) | driving_vla | `omni_q_20260428_165719` |
| `openread` | OpenREAD | driving_vla | `openread_20260428_120025` |
| `qwen25vl` | Qwen2.5-VL | general_vlm | `qwen25vl_20260425_072054` |
| `qwen3_vl_8b` | Qwen3-VL-8B | general_vlm | `qwen3_vl_8b_20260428_174418` |
| `recogdrive` | RecogDrive | driving_vla | `recogdrive_20260428_162305` |
| `safeauto` | SafeAuto | driving_vla | `safeauto_20260424_193636` |
| `wisead_8img` | WiseAD | driving_vla | `wisead_8img_20260429_112651` |