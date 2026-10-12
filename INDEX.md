# CausalDriveBench: Code & Document Index

A map of every document and source file in this repository: what it is, where it sits in the pipeline, and whether it is current or legacy.

[← Back to README](README.md) · 📄 [Paper (arXiv:2609.32157)](https://arxiv.org/abs/2609.32157) · 🤗 [Dataset](https://huggingface.co/datasets/causaldrivebench/CausalDriveBench)

**Legend:** ⭐ main entry point · 🟢 current · 🟡 helper / one-off utility · ⚪ legacy (kept for reference, not used by current entry points)

> Links into `prompts/` point to GitHub, because that folder is not published on the project site (its double-brace placeholder templates conflict with Jekyll's Liquid engine).

---

## Contents

- [Quick navigation: "I want to…"](#quick-navigation-i-want-to)
- [Documents](#documents)
- [Root files](#root-files)
- [`scripts/`: benchmark generation pipeline](#scripts-benchmark-generation-pipeline)
- [`prompts/`: LLM prompt templates](#prompts-llm-prompt-templates)
- [`driving_rules/`: driving-rule knowledge base](#driving_rules-driving-rule-knowledge-base)
- [`evaluation/`: model evaluation framework](#evaluation-model-evaluation-framework)

---

## Quick navigation: "I want to…"

| Goal | Start here |
|---|---|
| Understand the project at a glance | [README.md](README.md), [docs/project_overview.md](docs/project_overview.md) |
| Run the evaluation on bundled sample data (no GPU) | [evaluation/README.md](evaluation/README.md), [evaluation/generate_sample_outputs.py](evaluation/generate_sample_outputs.py) |
| Evaluate an existing model | [evaluation/models/](#evaluationmodels-model-integrations) → `<model>/commands.md`, [evaluation/docs/docker_operations.md](evaluation/docs/docker_operations.md) |
| Add a new model | [evaluation/docs/adding_a_model.md](evaluation/docs/adding_a_model.md), [evaluation/models/_template/](#evaluationmodels-model-integrations) |
| Understand the evaluation design | [evaluation/docs/architecture.md](evaluation/docs/architecture.md), [evaluation/docs/evaluation_pipeline.md](evaluation/docs/evaluation_pipeline.md) |
| Understand the data / file formats | [evaluation/docs/data_formats.md](evaluation/docs/data_formats.md), [README → Data formats](README.md#data-formats) |
| Understand trajectory (BTE / CTE) scoring | [evaluation/docs/trajectory_eval_details.md](evaluation/docs/trajectory_eval_details.md), [evaluation/common/trajectory_eval.py](evaluation/common/trajectory_eval.py) |
| Understand the LLM reasoning judge | [evaluation/docs/llm_judge.md](evaluation/docs/llm_judge.md), [evaluation/common/llm_judge.py](evaluation/common/llm_judge.py) |
| Regenerate causal scene graphs / QA from raw data | [scripts/batch_processing/run_batch_pipeline.sh](scripts/batch_processing/run_batch_pipeline.sh), [README → Regenerating the benchmark](README.md#regenerating-the-benchmark) |
| Build records from nuScenes / OpenScene / AV2 | [scripts/build_records/](#scriptsbuild_records-raw-data--unified-records) |
| Read the exact prompts used to build the benchmark | [prompts/causal_scene_graph/v5](https://github.com/renesas/causaldrivebench/tree/main/prompts/causal_scene_graph/v5), [prompts/causal_qas/v5](https://github.com/renesas/causaldrivebench/tree/main/prompts/causal_qas/v5) |
| Understand graph clustering / diversity selection | [scripts/graph_filtering/ALGORITHM.md](scripts/graph_filtering/ALGORITHM.md) |
| Browse an example HTML evaluation report | [evaluation/sample_reports/](https://github.com/renesas/causaldrivebench/tree/main/evaluation/sample_reports) (open `index.html` locally) |

---

## Documents

### Project-level

| Document | Contents |
|---|---|
| [README.md](README.md) | Project landing page: paper, dataset, install, quick start, pipeline reference, citation. |
| [INDEX.md](INDEX.md) | This file: code and document index. |
| [LICENSE](LICENSE) | BSD 3-Clause license for the code (© 2026 Renesas Electronics). |
| [docs/project_overview.md](docs/project_overview.md) | Non-technical overview of dataset creation and model evaluation, with mermaid diagrams and a file map. |
| [docs/agent_trajectory_intervention_spec.md](docs/agent_trajectory_intervention_spec.md) | Spec for agent → ego interventions (remove / reposition / speed change) and nuPlan re-simulation. |
| [driving_rules/driving_rules_reference.md](driving_rules/driving_rules_reference.md) | Human-readable reference for the driving-rule priority system (0–8) and rule summaries. |

### Generation pipeline

| Document | Contents |
|---|---|
| [scripts/batch_processing/README.md](scripts/batch_processing/README.md) | Notes on the batch-API pipeline. ⚪ Partly outdated: describes the earlier OpenAI-only, Stage-2 flow; see the README for the current stages. |
| [scripts/rendering/README.md](scripts/rendering/README.md) | Flask causal-graph editor: users, serving, edit mode, bookmarks. |
| [scripts/graph_filtering/ALGORITHM.md](scripts/graph_filtering/ALGORITHM.md) | 3-level hierarchical multi-modal clustering used to select a diverse subset of scene graphs. |

### Evaluation framework

| Document | Contents |
|---|---|
| [evaluation/README.md](evaluation/README.md) | Framework overview: Docker quick start, no-GPU sample run, layout, how to add a model. |
| [evaluation/docs/architecture.md](evaluation/docs/architecture.md) | Design principles, component diagram, base-class hierarchy, data flow, error handling. |
| [evaluation/docs/evaluation_pipeline.md](evaluation/docs/evaluation_pipeline.md) | End-to-end walkthrough (sequence diagram): loading → inference → post-processing → reports. |
| [evaluation/docs/data_formats.md](evaluation/docs/data_formats.md) | Directory variables, raw / benchmark / output layouts, file and JSONL schemas. |
| [evaluation/docs/adding_a_model.md](evaluation/docs/adding_a_model.md) | Step-by-step guide to integrating a new model from `models/_template`. |
| [evaluation/docs/docker_operations.md](evaluation/docs/docker_operations.md) | Docker / NVIDIA prerequisites, the five compose services, common workflows, troubleshooting. |
| [evaluation/docs/llm_judge.md](evaluation/docs/llm_judge.md) | LLM judge rubric (4 dimensions, 0–2), providers, configuration, CLI, report integration. |
| [evaluation/docs/trajectory_eval_details.md](evaluation/docs/trajectory_eval_details.md) | Trajectory evaluation design: BTE / CTE modes, GT extraction, prompts, parsing, ADE metrics. |
| [evaluation/docs/inference_optimization.md](evaluation/docs/inference_optimization.md) | Throughput tuning for API-served models (llama.cpp `--parallel`, VRAM budget, env vars). |
| [evaluation/docs/review_edit_sequentially.md](evaluation/docs/review_edit_sequentially.md) | Suggested reading order for the codebase and for adding a new VLA. |
| [evaluation/docker_steps.md](evaluation/docker_steps.md) | ⚪ Earlier all-in-one Docker guide, superseded by `evaluation/docs/`. |
| [evaluation/docs/models/eval_alpamayo_1_0.md](evaluation/docs/models/eval_alpamayo_1_0.md) | Alpamayo 1.0 integration notes: reasoning rollout, camera-major frames, ego-history interpolation. |
| [evaluation/docs/models/eval_alpamayo_1_5.md](evaluation/docs/models/eval_alpamayo_1_5.md) | Alpamayo 1.5 integration notes: native VQA interface, answer-token stripping, differences from 1.0. |
| [evaluation/docs/models/eval_gemma4.md](evaluation/docs/models/eval_gemma4.md) | Gemma 4 integration via OpenAI-compatible server; setup for the four variants; known issues. |
| [evaluation/docs/models/gemma4/inference_analysis.md](evaluation/docs/models/gemma4/inference_analysis.md) | Gemma 4 E4B profiling on A100: hidden thinking tokens as the bottleneck; options. |
| [evaluation/reports/insights_output/graph_analysis_report.md](evaluation/reports/insights_output/graph_analysis_report.md) | Example output of `insights/graph_analysis.py` on the bundled sample graphs. |

**Per-model docs:** [alpamayo_1_0/README.md](evaluation/models/alpamayo_1_0/README.md) · [alpamayo_1_5/README.md](evaluation/models/alpamayo_1_5/README.md) · [alpamayo_1_5/DESIGN.md](evaluation/models/alpamayo_1_5/DESIGN.md) · [omnidrive/README.md](evaluation/models/omnidrive/README.md) · [orion/SETUP.md](evaluation/models/orion/SETUP.md) · [qwen25vl/README.md](evaluation/models/qwen25vl/README.md) · [recogdrive/README.md](evaluation/models/recogdrive/README.md) · [safeauto/setup.md](evaluation/models/safeauto/setup.md) · [simlingo/setup.md](evaluation/models/simlingo/setup.md). Each model folder also has a `commands.md` cheat-sheet (see [evaluation/models/](#evaluationmodels-model-integrations)).

### Internal evaluation snapshot (`evaluation/docs/eval_20260429/`)

Working output of [evaluation/scripts/reorganize_and_report.py](evaluation/scripts/reorganize_and_report.py) for the 2026-04-29 run on `causal_nuscenes`. It is kept as a record of that run; **the paper holds the official, final numbers.**

| Document | Contents |
|---|---|
| [paper_tables.md](evaluation/docs/eval_20260429/paper_tables.md) | Draft Table 1 (VQA accuracy) and Table 2 (trajectory ADE), plus the model → run mapping. |
| [table1_vqa.tex](evaluation/docs/eval_20260429/table1_vqa.tex) · [table2_trajectory.tex](evaluation/docs/eval_20260429/table2_trajectory.tex) | LaTeX versions of the draft tables. |
| [combined_statistics.md](evaluation/docs/eval_20260429/combined_statistics.md) | Overview of all runs: completeness, accuracy breakdowns, issues. |
| [model_readiness.md](evaluation/docs/eval_20260429/model_readiness.md) | Per-run checklist of remaining work at the time. |
| `sanity_<model>_<timestamp>.md` (14 files) | One sanity-check report per run: [alpamayo_1_5](evaluation/docs/eval_20260429/sanity_alpamayo_1_5_20260428_081120.md), [cosmos_reason (1)](evaluation/docs/eval_20260429/sanity_cosmos_reason_20260427_192914.md), [cosmos_reason (2)](evaluation/docs/eval_20260429/sanity_cosmos_reason_20260429_054346.md), [intern_vl3_5_8b](evaluation/docs/eval_20260429/sanity_intern_vl3_5_8b_20260428_174406.md), [omni_q (1)](evaluation/docs/eval_20260429/sanity_omni_q_20260427_210155.md), [omni_q (2)](evaluation/docs/eval_20260429/sanity_omni_q_20260428_165719.md), [openread](evaluation/docs/eval_20260429/sanity_openread_20260428_120025.md), [qwen25vl (1)](evaluation/docs/eval_20260429/sanity_qwen25vl_20260425_072054.md), [qwen25vl (2)](evaluation/docs/eval_20260429/sanity_qwen25vl_20260428_163639.md), [qwen3_vl_8b](evaluation/docs/eval_20260429/sanity_qwen3_vl_8b_20260428_174418.md), [recogdrive (1)](evaluation/docs/eval_20260429/sanity_recogdrive_20260427_195115.md), [recogdrive (2)](evaluation/docs/eval_20260429/sanity_recogdrive_20260428_162305.md), [safeauto](evaluation/docs/eval_20260429/sanity_safeauto_20260424_193636.md), [wisead_8img](evaluation/docs/eval_20260429/sanity_wisead_8img_20260429_112651.md). |

---

## Root files

| File | Purpose |
|---|---|
| [config.yaml.example](config.yaml.example) | Template for `config.yaml`: LLM provider / model / keys, record-building settings, batch settings, per-dataset paths. Copy to `config.yaml` (git-ignored). |
| [environment.yaml](environment.yaml) | Conda environment `causal_graph` (Python 3.11): nuscenes-devkit, anthropic, openai, google-generativeai, opencv, shapely, networkx, graphviz … |
| [.gitignore](https://github.com/renesas/causaldrivebench/blob/main/.gitignore) | Ignores `config.yaml`, `dev.env`, `outputs/`, `batch_jobs/`, `batch_results/`, editor secrets, etc. |
| [LICENSE](LICENSE) | BSD 3-Clause. |
| [_config.yml](https://github.com/renesas/causaldrivebench/blob/main/_config.yml) · [_includes/head-custom.html](https://github.com/renesas/causaldrivebench/blob/main/_includes/head-custom.html) | GitHub Pages (Jekyll) site configuration and mermaid rendering hook. |
| [assets/causaldrivebench-pixel.svg](assets/causaldrivebench-pixel.svg) | Pixel-art project banner. |

---

## `scripts/`: benchmark generation pipeline

Raw driving logs → unified records → causal scene graphs → pruning / verification → clustering → causal QA → counterfactual trajectories → release package.

| File | Status | Purpose |
|---|---|---|
| [scripts/config_loader.py](scripts/config_loader.py) | 🟢 | Loads repo-root `config.yaml`, resolves `${ENV_VAR}` references; CLI args override config values. |
| [scripts/run_pipeline.sh](scripts/run_pipeline.sh) | ⚪ | Earlier real-time (non-batch) end-to-end runner: graphs → QA → shuffle → HTML → release. |
| [scripts/run_pipeline_v2.sh](scripts/run_pipeline_v2.sh) | ⚪ | Variant of `run_pipeline.sh` for an experimental v2 graph pipeline. |

### `scripts/batch_processing/`: Batch-API orchestration (current production path)

| File | Status | Purpose |
|---|---|---|
| [run_batch_pipeline.sh](scripts/batch_processing/run_batch_pipeline.sh) | ⭐ | **Main pipeline entry point.** Stages 1 → prune → 3a/3b/3c → 4cf/4np → 5 → flag-unchanged, with `--resume` / `--stop-after`. |
| [prepare_batches.py](scripts/batch_processing/prepare_batches.py) | 🟢 | Builds batch-request JSONL for each stage (1, 2, 3a, 3b, 3c, 4cf, 4ac, 4np, 5, verify); loads all prompts (graph & QA prompt version default: **v5**). |
| [submit_batch.py](scripts/batch_processing/submit_batch.py) | 🟢 | Provider-agnostic batch `submit` / `status` / `list` / `download` (Claude, OpenAI). |
| [process_results.py](scripts/batch_processing/process_results.py) | 🟢 | Parses downloaded batch results per stage → graphs, `active_qa.json` / `dormant_qa.json` / `distractor_qa.json`, trajectories. |
| [monitor_batches.py](scripts/batch_processing/monitor_batches.py) | 🟡 | Shows progress of active batch jobs; `--watch` auto-refreshes. |
| [estimate_tokens.py](scripts/batch_processing/estimate_tokens.py) | 🟡 | Samples a batch JSONL, calls `count_tokens`, extrapolates input tokens. |
| [verify_graphs.py](scripts/batch_processing/verify_graphs.py) | 🟡 | Standalone visual audit of `*_pruned.json` node / edge classifications → per-scene verification + report. |
| [verify_graphs.sh](scripts/batch_processing/verify_graphs.sh) | 🟡 | Shell wrapper running the `verify` stage via prepare / submit / process. |
| [dry_run_validate.sh](scripts/batch_processing/dry_run_validate.sh) | 🟡 | Offline validation of records, prompts and batch-JSONL generation (no API calls). |
| [README.md](scripts/batch_processing/README.md) | ⚪ | Earlier notes on the batch pipeline. |

### `scripts/build_records/`: raw data → unified records

Each builder writes `<records>/<scene>/SAMPLED_N/` with 4 timesteps of cameras, temporal BEVs, `state.json`, `calib.json`, `meta.json`.

| File | Status | Purpose |
|---|---|---|
| [build_nuscenes_records.py](scripts/build_records/build_nuscenes_records.py) | ⭐ | nuScenes v1.0-trainval + map expansion → records (4 cameras). |
| [build_openscene_records.py](scripts/build_records/build_openscene_records.py) | ⭐ | OpenScene / NAVSIM logs + sensor blobs + nuPlan maps → records. |
| [build_argoverse2_records.py](scripts/build_records/build_argoverse2_records.py) | ⭐ | Argoverse 2 sensor split → records (5 cameras). |

### `scripts/download/`: dataset download helpers

| File | Status | Purpose |
|---|---|---|
| [download_openscene_metadata.sh](scripts/download/download_openscene_metadata.sh) | 🟢 | Downloads OpenScene v1.1 trainval metadata pickles (~2 GB). |
| [download_openscene_sensor.sh](scripts/download/download_openscene_sensor.sh) | 🟢 | Downloads selected OpenScene camera-blob shards from Hugging Face (edit the CONFIG block). |
| [download_nuplan_maps.sh](scripts/download/download_nuplan_maps.sh) | 🟢 | Downloads and unzips nuPlan maps v1.1. |

### `scripts/common/`: shared libraries

| File | Status | Purpose |
|---|---|---|
| [llm_provider.py](scripts/common/llm_provider.py) | 🟢 | Unified real-time + batch interface for Claude, OpenAI and Gemini. |
| [scene_utils.py](scripts/common/scene_utils.py) | 🟢 | Scene / content-block helpers: image encoding & resizing, provider-specific message formats. |

### `scripts/causal_graph_extraction/`: graph generation, pruning, subgraph analysis

| File | Status | Purpose |
|---|---|---|
| [prune_graphs.py](scripts/causal_graph_extraction/prune_graphs.py) | ⭐ | Reverse-BFS from ego; nodes with no causal path to ego → distractors; writes `*_pruned.json`. |
| [extract_subgraphs.py](scripts/causal_graph_extraction/extract_subgraphs.py) | 🟢 | Enumerates canonical substructures (direct, chain, collider, confounding) and the QA types each licenses. |
| [subgraph_statistics.py](scripts/causal_graph_extraction/subgraph_statistics.py) | 🟡 | Markdown / JSON report of subgraph prevalence, roles, co-occurrence, licensed QA counts. |
| [dormant_subgraph_statistics.py](scripts/causal_graph_extraction/dormant_subgraph_statistics.py) | 🟡 | Same statistics for dormant subgraphs (dormant-direct, dormant parents, mixed confounding). |
| [v3_pipeline.py](scripts/causal_graph_extraction/v3_pipeline.py) | ⚪ | Earlier real-time Generator → Discriminator graph pipeline (LangGraph; prompts v1). |

### `scripts/graph_postprocessing/`: post-review clean-up → canonical `*_verified.json`

| File | Status | Purpose |
|---|---|---|
| [fix_causal_graphs.py](scripts/graph_postprocessing/fix_causal_graphs.py) | ⭐ | Rule-based edge / node clean-up of reviewed graphs (redundant crosswalks, behind-ego agents, ENV nodes, stranded nodes). |
| [align_scene_description.py](scripts/graph_postprocessing/align_scene_description.py) | ⭐ | Re-writes `scene_context` / `behavior_explanation` to match the active edges (batch or test mode). |
| [move_env_to_scene_context.py](scripts/graph_postprocessing/move_env_to_scene_context.py) | 🟡 | Folds ENVIRONMENT nodes into `scene_context.environment_conditions`. |
| [backfill_environment.py](scripts/graph_postprocessing/backfill_environment.py) | 🟡 | Fills a default `environment_conditions` where missing. |
| [fix_stranded_nodes.py](scripts/graph_postprocessing/fix_stranded_nodes.py) | 🟡 | Stranded agents → distractors; stranded infrastructure / obstacles removed. |
| [patch_verified_from_fixed.py](scripts/graph_postprocessing/patch_verified_from_fixed.py) | 🟡 | Restores orphan nodes in `_verified.json` from `_fixed.json`. |
| [prune_unreachable_nodes.py](scripts/graph_postprocessing/prune_unreachable_nodes.py) | 🟡 | In-place reverse-BFS pruning of `*_verified.json`. |

### `scripts/graph_filtering/`: diversity clustering & subset selection

| File | Status | Purpose |
|---|---|---|
| [ALGORITHM.md](scripts/graph_filtering/ALGORITHM.md) | 🟢 | Description of the hierarchical multi-modal clustering. |
| [robust_multimodal_filter.py](scripts/graph_filtering/robust_multimodal_filter.py) | ⭐ | 3-level clustering: graph family → scene / intent bucket → kernel agglomerative clustering. |
| [qa_kernel_filter.py](scripts/graph_filtering/qa_kernel_filter.py) | 🟢 | 2-level QA + topology kernel clustering, then scene-context kernels; one leader per leaf. |
| [topology_filtering.py](scripts/graph_filtering/topology_filtering.py) | 🟢 | Single-level topology-kernel clustering with elbow-selected threshold. |
| [filter_diverse_graphs.py](scripts/graph_filtering/filter_diverse_graphs.py) | 🟢 | 3-phase greedy selection of a maximally diverse subset. |
| [sample_diverse_graphs.py](scripts/graph_filtering/sample_diverse_graphs.py) | 🟢 | Budgeted sampling from clusters (group leaders + max-min extras). |
| [graph_statistics.py](scripts/graph_filtering/graph_statistics.py) | 🟡 | Markdown report of topology, node / edge distributions, ego behaviour. |
| [tsne_clusters.py](scripts/graph_filtering/tsne_clusters.py) | 🟡 | t-SNE of the averaged clustering-kernel distance. |
| [tsne_per_kernel.py](scripts/graph_filtering/tsne_per_kernel.py) | 🟡 | t-SNE panel per kernel + combined, coloured by topology. |

### `scripts/question_answers/`: causal QA generation & statistics

| File | Status | Purpose |
|---|---|---|
| [shuffle_mcq_options.py](scripts/question_answers/shuffle_mcq_options.py) | 🟢 | Balances correct-answer positions across A–D; updates reasoning and metadata. |
| [qa_statistics.py](scripts/question_answers/qa_statistics.py) | 🟡 | Aggregates active / dormant / distractor QA counts by rung, category, structure, format, difficulty. |
| [generate_causal_qa.py](scripts/question_answers/generate_causal_qa.py) | ⚪ | Earlier real-time QA generator (superseded by batch stages 3a–3c). |

### `scripts/counterfactual_trajectory/`: counterfactual & agent-intervention trajectories

| File | Status | Purpose |
|---|---|---|
| [llm_ego_trajectory.py](scripts/counterfactual_trajectory/llm_ego_trajectory.py) | 🟢 | Stage 5 (real-time): modified BEV + cameras → LLM → ego counterfactual trajectory. |
| [trajectory_utils.py](scripts/counterfactual_trajectory/trajectory_utils.py) | 🟢 | BEV redraw for remove / reposition / speed-change interventions; trajectory overlays. |
| [flag_unchanged_trajectories.py](scripts/counterfactual_trajectory/flag_unchanged_trajectories.py) | 🟢 | Writes `ego_trajectory_changed` back into QA JSONs to skip false-positive CFs. |
| [render_nuplan_results.py](scripts/counterfactual_trajectory/render_nuplan_results.py) | 🟡 | Parses nuPlan simulation results; draws original vs CF trajectories on BEV. |
| [counterfactual_pipeline.py](scripts/counterfactual_trajectory/counterfactual_pipeline.py) | ⚪ | Earlier real-time CF trajectory generation + visualisation. |
| [bev_intervention_renderer.py](scripts/counterfactual_trajectory/bev_intervention_renderer.py) | ⚪ | Earlier BEV intervention renderer + LLM CF trajectory. |

### `scripts/rendering/`: HTML viewers & editor

| File | Status | Purpose |
|---|---|---|
| [render_causal_graph.py](scripts/rendering/render_causal_graph.py) | ⭐ | Graphviz PNG + interactive per-scene D3 HTML (graph, cameras, BEV, QA); `--index`, `--root-index`, `--serve`. |
| [scene_viewer_template.py](scripts/rendering/scene_viewer_template.py) | 🟢 | HTML / JS template for the per-scene viewer (incl. edit mode). |
| [generate_index.py](scripts/rendering/generate_index.py) | 🟢 | Writes an output directory's `index.html`; optional per-scene / full zip downloads. |
| [gen_root_index.py](scripts/rendering/gen_root_index.py) | 🟢 | Root dashboard `index.html` with one card per output directory. |
| [editor_server.py](scripts/rendering/editor_server.py) | 🟢 | Flask server for human graph review: login, edit saving, per-user bookmarks. |
| [regenerate_cf_bevs.py](scripts/rendering/regenerate_cf_bevs.py) | 🟢 | Regenerates original / CF / overlay BEV and front-camera images for CF results. |
| [render_clusters.py](scripts/rendering/render_clusters.py) | 🟡 | HTML for flat clustering groups, linking into the scene viewer. |
| [render_clusters_full.py](scripts/rendering/render_clusters_full.py) | 🟡 | Simpler variant of the cluster renderer. |
| [render_qa_clusters.py](scripts/rendering/render_qa_clusters.py) | 🟡 | HTML for the 2-level QA-kernel clustering. |
| [css/index.css](scripts/rendering/css/index.css) · [css/qa_section.css](scripts/rendering/css/qa_section.css) · [css/scene_viewer.css](scripts/rendering/css/scene_viewer.css) · [css/unified_index.css](scripts/rendering/css/unified_index.css) | 🟢 | Styles for index pages, QA cards, scene viewer, unified multi-model index. |
| [README.md](scripts/rendering/README.md) | 🟢 | Editor-server documentation. |

### `scripts/release/`: release packaging

| File | Status | Purpose |
|---|---|---|
| [package_samples.py](scripts/release/package_samples.py) | ⭐ | Packages records + outputs into release folders (`meta`, `state`, `calib`, `frames`, `graph`, `graph_structure`, `qa/*`) + `manifest.json`. |

### `scripts/old/`: legacy

| File | Status | Purpose |
|---|---|---|
| [generate_questions.py](scripts/old/generate_questions.py) | ⚪ | Early template-based question generator with subgraph dedup / importance scoring. |
| [subgraph_extraction.py](scripts/old/subgraph_extraction.py) | ⚪ | Early direct / chain / fork / collider extraction. |
| [visualize_causal_graph.py](scripts/old/visualize_causal_graph.py) | ⚪ | Older copy of `render_causal_graph.py` with a `--stitch` overview grid. |
| [visualize_subgraphs.py](scripts/old/visualize_subgraphs.py) | ⚪ | Interactive HTML highlighting extracted subgraph patterns. |

---

## `prompts/`: LLM prompt templates

Loaded by [`prepare_batches.py`](scripts/batch_processing/prepare_batches.py). Graph and QA prompt versions default to **v5** (override with `batch.graph_prompt_version` / `batch.qa_prompt_version` in `config.yaml`, or `--graph-prompt-version` / `--qa-prompt-version` on `run_batch_pipeline.sh`).

| Folder / files | Stage | Status | Purpose |
|---|---|---|---|
| [causal_scene_graph/v5/system_prompt.md](https://github.com/renesas/causaldrivebench/blob/main/prompts/causal_scene_graph/v5/system_prompt.md) | 1 | ⭐ default | Causal scene-graph generator: 4 timesteps of cameras + 4 temporal BEVs (AV2 camera variables injected at load time). |
| [causal_scene_graph/v6/system_prompt.md](https://github.com/renesas/causaldrivebench/blob/main/prompts/causal_scene_graph/v6/system_prompt.md) | 1 | 🟢 | Newest graph schema; selectable as `v6`; also used as schema context by `align_scene_description.py`. |
| [causal_scene_graph/v1–v4](https://github.com/renesas/causaldrivebench/tree/main/prompts/causal_scene_graph) (`system_prompt.md`, `refinement_prompt.md`) | 1 / 2 | ⚪ | Earlier generator + refinement (Stage 2) prompt pairs. |
| [causal_qas/v5/Causal_Ladder_*](https://github.com/renesas/causaldrivebench/tree/main/prompts/causal_qas/v5) | 3a | ⭐ default | Active-edge "Causal Ladder" QA across rungs 0–3. |
| [causal_qas/v5/Dormant_Nodes_*](https://github.com/renesas/causaldrivebench/tree/main/prompts/causal_qas/v5) | 3b | ⭐ default | Dormant-edge / latent-activation QA. |
| [causal_qas/v5/Distractor_Nodes_*](https://github.com/renesas/causaldrivebench/tree/main/prompts/causal_qas/v5) | 3c | ⭐ default | Distractor-rejection QA. |
| [causal_qas/v3](https://github.com/renesas/causaldrivebench/tree/main/prompts/causal_qas/v3), [causal_qas/v4](https://github.com/renesas/causaldrivebench/tree/main/prompts/causal_qas/v4) | 3a–3c | ⚪ | Earlier QA prompt versions (v3 includes the retired `D_L_*` discrimination / latent-activation pair). |
| [alternate_trajectories/Intervention_*](https://github.com/renesas/causaldrivebench/tree/main/prompts/alternate_trajectories) | 4cf (`llm_t0`) | 🟢 | Infrastructure intervention at T=0 → 8 future ego waypoints. |
| [alternate_trajectories/CounterFactual_*](https://github.com/renesas/causaldrivebench/tree/main/prompts/alternate_trajectories) | 4cf (`llm_obs_window`) | 🟢 | Intervention in the observation window → 4 alternate-history ego waypoints. |
| [nuplan_config/](https://github.com/renesas/causaldrivebench/tree/main/prompts/nuplan_config) | 4cf (agent) / 4np | 🟢 | Agent counterfactual question → nuPlan agent-intervention config JSON. |
| [agent_intervention/CF_Ego_Trajectory_T0_*](https://github.com/renesas/causaldrivebench/tree/main/prompts/agent_intervention) | 5 | 🟢 | `apply_at_t == 0` → 8 future ego waypoints from the modified BEV. |
| [agent_intervention/CF_Ego_Trajectory_Past_*](https://github.com/renesas/causaldrivebench/tree/main/prompts/agent_intervention) | 5 | 🟢 | `apply_at_t < 0` → 4 past (alternate-history) ego waypoints. |
| [graph_verification/system_prompt.md](https://github.com/renesas/causaldrivebench/blob/main/prompts/graph_verification/system_prompt.md) | verify | 🟡 | Visual auditor that spot-checks node classes and edge justifications. |

---

## `driving_rules/`: driving-rule knowledge base

| File | Purpose |
|---|---|
| [driving_rules_reference.md](driving_rules/driving_rules_reference.md) | Human-readable reference: priority system (0 override … 8 advisory) and rule summaries. |
| [taxonomy.yaml](driving_rules/taxonomy.yaml) | Rule taxonomy, priority levels, conflict resolution. |
| [generic_rules.yaml](driving_rules/generic_rules.yaml) | Jurisdiction-agnostic rules with sign / marking constants. |
| [sg_specific_rules.yaml](driving_rules/sg_specific_rules.yaml) | Singapore (LTA) rules: left-hand traffic, bus lanes, expressways. |
| [us_ca_specific_rules.yaml](driving_rules/us_ca_specific_rules.yaml) | California (DMV / CVC) rules: HOV, curb colours, right-turn-on-red. |

---

## `evaluation/`: model evaluation framework

Containerised, resumable, multi-model evaluation: **dataloader** (`prompts.jsonl`) → **inference** (`outputs.jsonl`) → **post-processing** (metrics, optional LLM judge) → 4-tier JSON reports → static HTML reports.

### Top level

| File | Status | Purpose |
|---|---|---|
| [orchestrate.py](evaluation/orchestrate.py) | ⭐ | Multi-model CLI: builds Docker images, runs inference + post-processing, aggregates a comparison report. |
| [eval_config.yaml](evaluation/eval_config.yaml) | ⭐ | Host paths, model registry, QA types, trajectory-eval and LLM-judge settings, report schema version. |
| [generate_sample_outputs.py](evaluation/generate_sample_outputs.py) | ⭐ | No-GPU demo: simulates a run on `sample_data/` → `sample_outputs/` + `sample_reports/`. |
| [sanity_check.py](evaluation/sanity_check.py) | 🟢 | Validates runs against the dataset: expected vs actual counts, failures, trajectory completeness, health score. |
| [build_image_cache.py](evaluation/build_image_cache.py) | 🟡 | Pre-encodes benchmark images to sharded base64 pickles reusable across models. |
| [README.md](evaluation/README.md) · [docker_steps.md](evaluation/docker_steps.md) | 🟢 / ⚪ | Framework guide · earlier Docker guide. |

### `evaluation/common/`: shared framework

| File | Purpose |
|---|---|
| [base_dataloader.py](evaluation/common/base_dataloader.py) | Abstract `ModelDataLoader`: image paths, sample / prompt formatting, writes `prompts.jsonl`, ego-history interpolation. |
| [base_inference.py](evaluation/common/base_inference.py) | Abstract `ModelInference` (`load_model`, `run_single`); resumable `prompts.jsonl` → `outputs.jsonl`; timing. |
| [base_postprocess.py](evaluation/common/base_postprocess.py) | Abstract `ModelPostprocessor`: binary / MCQ answer parsing, sample & dataset reports, judge hook, standard CLI. |
| [dataset.py](evaluation/common/dataset.py) | `BenchmarkDataset`: reads the manifest and each `SAMPLED_N` (frames, state, calib, meta, graph, QA). |
| [metrics.py](evaluation/common/metrics.py) | Torch-free metrics: MCQ / binary accuracy, per-group breakdowns, F1, minADE helpers. |
| [trajectory_eval.py](evaluation/common/trajectory_eval.py) | BTE and CTE-R2 / CTE-R3 trajectory evaluation: GT extraction, alignment, ADE, analysis flags. |
| [llm_judge.py](evaluation/common/llm_judge.py) | LLM judge: 4 reasoning dimensions scored 0–2 (Anthropic, OpenAI, Azure providers). |
| [report_generator.py](evaluation/common/report_generator.py) | 4-tier JSON reports (sample → dataset → run → root), schema-versioned. |
| [html_reporter.py](evaluation/common/html_reporter.py) | Self-contained static HTML reports (inline CSS, relative links) at 5 levels. |
| [image_cache.py](evaluation/common/image_cache.py) | `PersistentImageCache`: resize → JPEG → base64, sharded pickles + JSON index. |
| [templates/report.css](evaluation/common/templates/report.css) | Stylesheet inlined into every HTML report (incl. print-to-PDF rules). |
| [prompts/llm_judge_system.txt](evaluation/common/prompts/llm_judge_system.txt) · [prompts/llm_judge_user_template.txt](evaluation/common/prompts/llm_judge_user_template.txt) | LLM-judge system prompt (rubric) and per-question user template. |
| [prompts/trajectory_bte_user.txt](evaluation/common/prompts/trajectory_bte_user.txt) · [prompts/trajectory_cte_user.txt](evaluation/common/prompts/trajectory_cte_user.txt) | Prompts for baseline (BTE) and counterfactual (CTE) trajectory prediction. |
| [utils/config_reader.py](evaluation/common/utils/config_reader.py) | `CDBConfig`: resolves host vs Docker paths (`EVAL_RUNTIME`) and frame image paths. |
| [utils/errors.py](evaluation/common/utils/errors.py) · [utils/exceptions.py](evaluation/common/utils/exceptions.py) | Fatal error classes · recoverable exception classes. |
| [utils/find_in_json.py](evaluation/common/utils/find_in_json.py) | CLI to search JSON files by key / dotted sub-path value. |
| [utils/debugging_tools.py](evaluation/common/utils/debugging_tools.py) | `inspect_object()` helper for debugging. |
| [\_\_init\_\_.py](https://github.com/renesas/causaldrivebench/blob/main/evaluation/common/__init__.py) · [utils/\_\_init\_\_.py](https://github.com/renesas/causaldrivebench/blob/main/evaluation/common/utils/__init__.py) | Package markers / re-exports. |

### `evaluation/scripts/`, `tests/`, `insights/`

| File | Purpose |
|---|---|
| [scripts/generate_html_reports.py](evaluation/scripts/generate_html_reports.py) | Builds the HTML report tree for a finished run from its JSON reports. |
| [scripts/judge_retry_batch.py](evaluation/scripts/judge_retry_batch.py) | Re-runs failed / unscored judge items via the Anthropic Batch API and recomputes summaries. |
| [scripts/reorganize_and_report.py](evaluation/scripts/reorganize_and_report.py) | Normalises submitted run outputs, runs sanity checks, rebuilds reports, writes Markdown / LaTeX tables. |
| [tests/conftest.py](evaluation/tests/conftest.py) | Pytest setup: `integration` marker, API-key loading from `dev.env`. |
| [tests/test_llm_judge.py](evaluation/tests/test_llm_judge.py) | Judge unit tests + live API tests (skipped without keys). |
| [tests/test_trajectory_eval.py](evaluation/tests/test_trajectory_eval.py) | Trajectory-evaluation tests: GT extraction, alignment, ADE, flags, parsing, report structure. |
| [tests/explore_config.py](evaluation/tests/explore_config.py) | Exploration script printing everything `CDBConfig` resolves. |
| [insights/graph_analysis.py](evaluation/insights/graph_analysis.py) | Parallel, cached analysis of scene-graph topology / causal status → Markdown report. |

### `evaluation/models/`: model integrations

Every model folder follows the template layout: `Dockerfile`, `docker-compose.yml`, `dataloader.py`, `inference.py`, `postprocess.py`, `requirements.(in|txt)`, `dev.env.example`, `commands.md`, and usually `system_prompt.txt`, `setup.sh` and `tests/` (`test_dataloader.py`, `test_postprocess.py`).

| Folder | Wraps (upstream) | Extra files / notes |
|---|---|---|
| [_template/](evaluation/models/_template/commands.md) | Skeleton to copy for a new model | `tutorial.ipynb`: pipeline walkthrough inside the container. |
| [alpamayo_1_0/](evaluation/models/alpamayo_1_0/README.md) | NVIDIA Alpamayo-R1-10B, the **reference implementation** | `README.md`, `tutorial.ipynb`. |
| [alpamayo_1_5/](evaluation/models/alpamayo_1_5/README.md) | NVIDIA Alpamayo-1.5-10B (native VQA interface) | `README.md`, `DESIGN.md` (design log, differences from 1.0). |
| [doe/](evaluation/models/doe/commands.md) | Doe-1 multimodal driving world model | none |
| [gemma4/](evaluation/models/gemma4/commands.md) | Google Gemma 4 (E2B / E4B / 26B-A4B / 31B) via an OpenAI-compatible llama.cpp server | `server/Dockerfile`, `run_inference.sh`, `run_postprocess.sh`, `monitor.sh`, `milestone_watch.sh`, `original_test.py`. |
| [impromptu_vla/](evaluation/models/impromptu_vla/commands.md) | Impromptu-VLA 3B (Qwen2.5-VL fine-tune) served with sglang | none |
| [omnidrive/](evaluation/models/omnidrive/README.md) | NVlabs OmniDrive (EVA-02 + StreamPETR + TinyLlama) | `README.md` (architecture & run guide). |
| [opendrivevla/](evaluation/models/opendrivevla/commands.md) | OpenDriveVLA-0.5B (Qwen2.5-VL backbone) | none |
| [orion/](evaluation/models/orion/SETUP.md) | ORION (EVA-ViT + QFormer + LLaMA); VQA + planning-decoder trajectories | `SETUP.md`, `tests/test_inference_calib.py`. |
| [qwen25vl/](evaluation/models/qwen25vl/README.md) | Qwen2.5-VL-3B / 7B-Instruct via HF transformers (general-VLM baseline) | `README.md`; host-run (no Docker files). |
| [recogdrive/](evaluation/models/recogdrive/README.md) | ReCogDrive-8B-RL (InternVL3-8B backbone) | `README.md`. |
| [safeauto/](evaluation/models/safeauto/setup.md) | SafeAuto (ICML 2025; LLaVA fine-tuned on DriveLM) | `setup.md`, `traj_dataloader.py`, `traj_eval.py`, `traj_postprocess.py` (BTE / CTE trajectories, ADE / FDE), `tutorial.ipynb`. |
| [simlingo/](evaluation/models/simlingo/setup.md) | SimLingo (InternVL2-1B + LoRA, CARLA-trained) | `setup.md`, `traj_eval.py` (uses SimLingo's waypoint head). |

### Sample data, outputs & reports

| Folder | Contents |
|---|---|
| [sample_data/](evaluation/sample_data/causal_nuscenes/manifest.json) | One benchmark sample each from `causal_nuscenes` (`nuscenes-scene-0072/SAMPLED_1`), `causal_argoverse2` and `causal_openscene`: manifest + `calib`, `frames`, `graph`, `meta`, `state` and `qa/{active,dormant,distractor}_qa.json`. |
| [sample_outputs/](evaluation/sample_outputs/alpamayo_1_0_sample_20260421_064223/report.json) | Simulated `alpamayo_1_0` run: `prompts.jsonl`, `outputs.jsonl` and sample / dataset / run `report.json`. |
| [sample_reports/](https://github.com/renesas/causaldrivebench/tree/main/evaluation/sample_reports) | Static, self-contained HTML report tree for the same run (root → run → dataset → scene → sample); open `index.html` in a browser. |
| [reports/insights_output/](evaluation/reports/insights_output/graph_analysis_report.md) | `graph_analysis.py` output on the sample graphs (`graph_analysis_report.md`, `aggregated_stats.json`, cache). |
