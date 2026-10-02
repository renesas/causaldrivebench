# CausalDriveBench Data Formats

Reference for all file schemas used by the evaluation framework.
This should be consistent for both the host and docker-based model evaluation.

Abbreviation: CDB -- CausalDriveBench

---

## Directory environment variables

1. The following are some of the important constants that affect the file organization.
2. These directory paths are expected to be correctly set in the eval_config.yaml to the corresponding directories on
   the host environment based on the descriptions below.
3. The docker path values are constants for the expected volume mounts in the docker environment.
4. Instruction to the user:
    1. Create a local copy of the eval_config.yaml.
    2. And, when running evaluation scripts directly on the host, appropriately update your local copy to replace the
       default values.
5. The EVAL_RUNTIME environment variable decides which paths will be used during inference, so that the evaluation
   scripts run seamlessly on the host environment and the docker environment.
6. The evaluation/common/utils/config_reader.py correctly reads the host configs from the eval_config.yaml and creates
   a Namespace mapping across evaluation code files for correctly accessing the relevant paths during runtime.

```python
import os

EVAL_RUNTIME = "host"  # or "docker"

# All the key value pairs in this dict should follow the same structure
# "dir_key": {"host": "host_path", "docker": "docker_path", "description": "describe what this path is for"}
dict_dir_maps = {
    "workspace": {
        "host": "/path/to/workspace",
        "docker": "/workspace",
        "description": "Path to your workspace."
    }
}

dict_dir_maps["dir_data_root"] = {
    "host": "/path/to/data/root",
    "docker": os.path.join(dict_dir_maps["workspace"], "data"),
    "description": "Path to the data files root"
}

dict_dir_maps["dir_raw_data"] = {
    "host": "/path/to/raw/data",
    "docker": os.path.join(dict_dir_maps["dir_data_root"], "raw_data"),
    "description": "Path to the raw_data root folder. This has subfolders for the opensource datasets."
}

dict_dir_maps["dir_causal_drive_bench"] = {
    "host": "/path/to/causal/drive/bench",
    "docker": os.path.join(dict_dir_maps["dir_data_root"], "causal_drive_bench"),
    "description": "Path to the causal_drive_bench root folder. This has subfolders for the CausalDriveBench data annotations."
}

dict_dir_maps["dir_cdb_outputs"] = {
    "host": "/path/to/cdb/outputs",
    "docker": os.path.join(dict_dir_maps["dir_data_root"], "cdb_outputs"),
    "description": "Path to the cdb_outputs root folder. This has subfolders for the CausalDriveBench inference outputs."
}

dict_dir_maps["dir_cdb_reports"] = {
    "host": "/path/to/cdb/reports",
    "docker": os.path.join(dict_dir_maps["dir_data_root"], "cdb_reports"),
    "description": "Path to the cdb_reports root folder. This has subfolders for the CausalDriveBench evaluation reports."
}

dict_dir_maps["dir_cdb_eval_repo"] = {
    "host": "/path/to/parent/of/evaluation",
    "docker": os.path.join(dict_dir_maps["workspace"], "repo"),
    "description": "Path to the parent directory of evaluation code."
}

dict_dir_maps["dir_model_code"] = {
    "host": "/path/to/model/code",
    "docker": os.path.join(dict_dir_maps["workspace"], "model"),
    "description": "Path to the model's code root folder. This has model's codebase that is being evaluated."
}

dict_dir_maps["dir_model_weights"] = {
    "host": "/path/to/model/weights",
    "docker": os.path.join(dict_dir_maps["workspace"], "weights"),
    "description": "Path to the model's trained weights root folder. This could also refer to HF_HOME cache folder, etc."
}

```

---

## Directory Layout -- dir_raw_data

This is based on the 3 open-source datasets as listed below.
These datasets have their own folder structure but that does not affect the evaluation framework.
The main usage of these files is to reference in the frames.json to read the correct image files.

```
<RAW_DATA_DIR>/                        # e.g. /raw_data
├── nuscenes/                          
├── argoverse2/                          
└── openscene/

```

---

## Directory Layout -- dir_causal_drive_bench

For each subfolder in the dir_raw_data directory above (e.g. the 3 open-source datasets), there is a corresponding
causal_<open-source-dataset> folder that has the expanded structure as shown below.

```
<CAUSAL_BENCH_DIR>/                     # e.g. evaluation/sample_data/
├── causal_nuscenes/
│   ├── manifest.json                   # Metadata about all the SAMPLED instances
│   ├── nuscenes-scene-<...>/           # e.g. nuscenes-scene-0001/
│   ├── nuscenes-scene-<...>/
│   └── nuscenes-scene-<...>/
│       ├── SAMPLED_x/
│       ├── SAMPLED_.../
│       └── SAMPLED_<N>/                    
├── causal_argoverse2/
│   ├── manifest.json
│   ├── argoverse2-<id1>/              # e.g. argoverse2-scene-0001/
│   ├── argoverse2-<id...>/
│   └── argoverse2-<id-N>/
│       ├── SAMPLED_x/
│       ├── SAMPLED_.../
│       └── SAMPLED_<N>/                    
└── causal_openscene/
    ├── manifest.json
    ├── openscene-<id1>/               # e.g. openscene-scene-0001/
    ├── openscene-<id...>/
    └── openscene-<id-N>/
        ├── SAMPLED_x/
        ├── SAMPLED_.../
        └── SAMPLED_<N>/ 

-----------
# Each SAMPLED_<N> folder above has the following folder structure of the CausalDriveBench
# Refer to the evaluation/sample_data files for example implementations.

<bench-dir-sample-folder>            # e.g. /causal_drive_bench/nuscenes-scene-0001/SAMPLED_0/
├── frames.json                      — per-frame image paths keyed by frame name
├── state.json                       — ego + agent state at T=0 (ego-centric history)
├── calib.json                       — camera calibration matrices
├── meta.json                        — scene metadata (dataset, split, tokens)
├── graph.json                       — pruned causal graph (falls back to raw)
└── qa                               — all benchmark question JSON
    ├── active_qa.json               — ladder / active questions
    ├── dormant_qa.json              — dormant link questions
    └── distractor_qa.json           — distractor node questions
    
```

Notes:

1. The sampled folders are not sequential and some folders might be malformed.
2. qa – subfolder
    1. Has the following 3 question answer JSON files generated based on the scene-graph.
        1. active_qa.json
        2. dormant_qa.json
        3. distractor_qa.json
    2. Some of the individual <type>_qa.json files may be missing in some cases.
    3. The files themselves may be malformed and may not have QA pairs even if the file exists.
3. Each <type>_qa.json file has the following structure:
    1. A "questions" key and value of list of question metadata dictionary.
    2. The question metadata dictionary is valid if:
        1. REQUIRED
            1. "id" -- question ID in the sample
            2. "question" -- String
            3. "answer_format" -- binary or mcq
            4. "options" -- if answer_format is mcq -- A.../B.../C.../D... etc. else Yes/No for binary
            5. "correct_answer": Yes/No/<correct-answer-option>
            6. "reasoning": The model's ground truth reasoning text.
        2. OPTIONAL
            1. Other metadata key value pairs
    3. If the question is invalid or cannot be processed, raise an appropriate exception, and then ignore processing
       the question.

---

## Directory Layout -- dir_cdb_outputs

1. This is quite similar to the above CDB benchmark directory layout. 
2. The corresponding JSONL files during evaluation are created for each SAMPLED directory.

```
<CAUSAL_BENCH_OUTPUTS_DIR>/             # e.g. /cdb_outputs/
├── <model-id-1>_<timestamp>/                 # e.g. alpamayo_1_0_20260409_171324
│   ├── causal_nuscenes/
│   │   ├── nuscenes-scene-<...>/           # e.g. nuscenes-scene-0001/
│   │   ├── nuscenes-scene-<...>/
│   │   └── nuscenes-scene-<...>/
│   │       ├── SAMPLED_x/
│   │       ├── SAMPLED_.../
│   │       └── SAMPLED_<N>/                    
│   ├── causal_argoverse2/
│   │   ├── argoverse2-<id1>/               # e.g. argoverse2-72737475-7678-79/
│   │   ├── argoverse2-<id...>/
│   │   └── argoverse2-<id-N>/
│   │       ├── SAMPLED_x/
│   │       ├── SAMPLED_.../
│   │       └── SAMPLED_<N>/                    
│   └── causal_openscene/
│       ├── openscene-<id1>/                # e.g. openscene-2021.06.14.17.26.26_veh-38_027/
│       ├── openscene-<id...>/
│       └── openscene-<id-N>/
│           ├── SAMPLED_x/
│           ├── SAMPLED_.../
│           └── SAMPLED_<N>/ 
└── <model-id-2>_<timestamp>/ 
    ├── causal_nuscenes/                   
    ├── causal_argoverse2/                 
    └── causal_openscene/

-----------
# Each SAMPLED_<N> folder above has the following folder structure of the CausalDriveBench

<output-bench-dir-sample-folder>      # e.g. /cdb_outputs/nuscenes-scene-0001/SAMPLED_0/
├── prompts.jsonl                     — inference input -- one line per question
└── outputs.jsonl                     — inference output -- one line per question
```

---

## File Formats — Sample Data Reference

The files below are documented using the bundled sample data under `evaluation/sample_data/`.

### `frames.json`

Stores relative image paths for each temporal frame.  The `data_root` field is
**ignored** by the evaluation framework — use `CDBConfig.resolve_frame_path()` instead.

```json
{
  "data_root": "/workspace/data",
  "frames": {
    "Tm1p5": {
      "cam_front":       "raw_data/nuscenes/samples/CAM_FRONT/<file>.jpg",
      "cam_front_right": "raw_data/nuscenes/samples/CAM_FRONT_RIGHT/<file>.jpg",
      "cam_front_left":  "raw_data/nuscenes/samples/CAM_FRONT_LEFT/<file>.jpg",
      "cam_back":        "raw_data/nuscenes/samples/CAM_BACK/<file>.jpg"
    },
    "Tm1p0": { "...": "..." },
    "Tm0p5": { "...": "..." },
    "Tp0p0": { "...": "..." }
  }
}
```

Camera keys vary by source dataset:
- **nuscenes**: `cam_front`, `cam_front_left`, `cam_front_right`, `cam_back`
- **argoverse2**: `cam_front`, `cam_front_left`, `cam_front_right`, `cam_back_left`, `cam_back_right`
- **openscene**: `cam_front`, `cam_front_left`, `cam_front_right`, `cam_back`

### `state.json`

Ego vehicle and agent states at T=0 (ego-centric frame).  Structure varies
slightly by dataset.  Ego history is extracted by `ModelDataLoader.interpolate_ego_history()`.

### `calib.json`

Camera calibration matrices (intrinsics, extrinsics) for all cameras.
Used by models that need camera-to-ego transformations.

### `meta.json`

Scene metadata: dataset name, split, source tokens, timestamp.

### `graph.json`

Pruned causal scene graph used to generate the QA questions.

### `qa/<type>_qa.json`

Each question file has the following structure:

```json
{
  "questions": [
    {
      "id":             "Q1",
      "question":       "Which element is currently preventing you from proceeding?",
      "answer_format":  "mcq",
      "options": [
        "A) The construction worker on the crosswalk",
        "B) The SUV stopped behind you",
        "C) The construction barriers on the right",
        "D) The traffic signal ahead"
      ],
      "correct_answer": "A",
      "reasoning":      "You are stopped because people are crossing directly ahead…",
      "graph_structure": "Direct",
      "question_type":   "CaI",
      "rung":            0,
      "meta":           { "...": "..." }
    }
  ]
}
```

For binary questions, `options` is `null` and `correct_answer` is `"Yes"` or `"No"`.

---

## Output JSONL File Schemas

### `prompts.jsonl`

1. One JSON object per line, one line per question. 
2. Written by `DataLoader.save_prompts()`.
3. `is_evaluated` is set to `false` when written.  Resume detection uses the
   presence of the corresponding `question_id` in `outputs.jsonl` rather than
   this flag, keeping GPU inference unblocked by file I/O.
4. `qa_text` contains the question plus a format hint only — **no ground-truth answer or reasoning** — to prevent answer leakage into the model prompt.  It is generated by `ModelDataLoader._format_inference_text()`.  `question_text` is the raw question string without any formatting.

**Minimum required fields** (framework contract):

```json
{ 
  "scene_id": "openscene-2021.06.14.17.26.26_veh-38_027",
  "sample_id": "SAMPLED_2",
  "question_id": "DQ1",
  "prompt_id": "<sequence-id-in-this-json-file>",
  "is_evaluated": "false",
  "question_json_file": "dormant_qa.json",
  "answer_format": "binary",
  "question_text": "Question: Is the silver sedan...",
  "qa_text": "Question: Is the silver sedan... \n\nFormat: Answer: Yes or No",
  "image_paths": [
    # list of image path metadata of following dict format based on the model input format
    {
      "path": "raw_data/openscene/2021.06.14.17.26.26_veh-38_02740_03036/CAM_R0/a04d050b77765886.jpg",
      "time_key": "Tm1p0",
      "camera_key": "cam_front_right"
    },
    {
      "path": "...",
      "time_key": "...",
      "camera_key": "..."
    },
    ...
  ]
}
```

**Model-specific additions** (alpamayo_1_0 example):

```json
{
  "ego_history_xyz": [
    [
      -5.7,
      1.7,
      0.0
    ],
    [
      -3.9,
      0.8,
      0.0
    ],
    [
      -2.0,
      0.2,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0
    ]
  ],
  "ego_history_rot": [
    [
      [
        1,
        0,
        0
      ],
      [
        0,
        1,
        0
      ],
      [
        0,
        0,
        1
      ]
    ],
    ...
  ]
}
```

All fields must be JSON-serialisable (no `numpy` arrays, no `PIL.Image` objects).

### `outputs.jsonl`

1. One JSON object per line, one line per question. 
2. Written by `ModelInference.run_from_jsonl()`.
3. The raw_output from the model will be used to postprocess and generate the metrics.
4. Depending on the model, there might different kinds of raw_outputs.

```json
{
  "scene_id": "openscene-2021.06.14.17.26.26_veh-38_027",
  "sample_id": "SAMPLED_2",
  "question_id": "DQ1",
  "prompt_id": "<from-prompts-jsonl>",
  "raw_output": {
    "text": "Answer: <model-answer>\nReasoning: <model-reasoning ... e.g. The truck is parked and not moving...>",
    "waypoints": [],   # optional other raw_output
    "other_key": ...,  # serializable output
  },
  "inference_time_s": 2.34,
  "timestamp": "2026-04-04T10:00:00Z"
}
```

---

## `report.json` Schema

Written by the model-specific postprocessor (e.g. `alpamayo_1_0/postprocess.py`).
Per-sample reports live at `<run_dir>/<dataset>/<scene_id>/<sample_id>/report.json`.

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-04-17T21:16:27.873834+00:00",
  "level": "sample",
  "run_name": "alpamayo_1_0_20260417_190704",
  "dataset": "causal_nuscenes",
  "scene_id": "nuscenes-scene-0007",
  "sample_id": "SAMPLED_0",
  "n_questions": 22,
  "metrics": {
    "overall": {
      "accuracy": 0.82,
      "n": 22,
      "correct": 18
    },
    "per_qa_type": {
      "ladder":     {"accuracy": 0.80, "n": 5,  "correct": 4},
      "dormant":    {"accuracy": 0.89, "n": 9,  "correct": 8},
      "distractor": {"accuracy": 0.75, "n": 8,  "correct": 6}
    },
    "confusion": {
      "matrix": {
        "Yes": {"Yes": 10, "No": 2},
        "No":  {"Yes": 2,  "No": 8}
      },
      "most_confused": [
        {"true": "Yes", "predicted": "No", "count": 2},
        {"true": "No",  "predicted": "Yes", "count": 2}
      ]
    }
  },
  "qa_results": [
    {
      "question_id":      "CI1",
      "qa_type":          "distractor",
      "answer_format":    "binary",
      "question_text":    "Is the white van in the adjacent lane to your right responsible for your current speed?",
      "predicted":        "No",
      "ground_truth":     "No",
      "correct":          true,
      "raw_output_text":  "No, the white van is not responsible for my current speed.",
      "inference_time_s": 2.34
    }
  ]
}
```

Notes:
- `level` is always `"sample"` for per-sample reports; higher-level aggregate reports use `"dataset"` or `"run"`.
- `metrics` contains the nested accuracy / confusion data; `qa_results` holds one entry per question with full traceability.
- `raw_output_text` is the extracted text from the model's `raw_output` dict (the `"text"` key).

---

## Summary Table

| File / Field          | Written by                        | Read by                       | Notes |
|-----------------------|-----------------------------------|-------------------------------|-------|
| `frames.json`         | Dataset pipeline                  | `BenchmarkDataset`, DataLoader | Ignore `data_root`; use `CDBConfig.resolve_frame_path()` |
| `qa/<type>_qa.json`   | Dataset pipeline                  | `BenchmarkDataset`, Postprocessor | Missing files are silently skipped |
| `prompts.jsonl`       | `ModelDataLoader.save_prompts()`  | `ModelInference.run_from_jsonl()` | One per SAMPLED_N dir |
| `outputs.jsonl`       | `ModelInference.run_from_jsonl()` | `ModelPostprocessor.compute_metrics()` | One per SAMPLED_N dir; appended for resume |
| `report.json`         | Model postprocessor (`postprocess.py`) | Orchestrator, HTML reporter, human review | Per sample; aggregated at dataset/run level |
| `inference.log`       | `inference.py` logging setup      | Developer debugging           | Per run, at run_dir root |