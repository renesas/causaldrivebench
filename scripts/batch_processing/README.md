# Batch Processing Pipeline

This directory contains scripts for running the causal graph and QA generation pipeline using OpenAI's Batch API, which provides 50% cost savings and better throughput for large-scale processing.

## Overview

The batch processing system processes scenes in parallel while respecting stage dependencies:

```
Stage 1: Causal Graph Generation (Generator)
   ↓
Stage 2: Causal Graph Refinement (Discriminator)
   ↓
Stage 3a: QA Generation (Active) ──┐
Stage 3b: QA Generation (Dormant)    ──┴── Merge → result.json
```

## Quick Start

### 1. Enable Batch Processing

Edit `config.yaml`:
```yaml
batch:
  enabled: true  # Enable batch API
  batch_dir: batch_jobs
  results_dir: batch_results
```

### 2. Run Full Pipeline

Process 100 scenes from nuScenes:
```bash
python run_batch_pipeline.py --dataset nuscenes --num-scenes 100
```

With continuous monitoring:
```bash
python run_batch_pipeline.py --dataset nuscenes --num-scenes 100 --monitor
```

### 3. Monitor Progress

In a separate terminal:
```bash
python monitor_batches.py --watch
```

## Components

### 1. `prepare_batches.py`
Prepares JSONL batch request files for each pipeline stage.

```bash
# Prepare Stage 1 (generator) batch
python prepare_batches.py --stage 1 --dataset nuscenes --records /path/to/records

# Prepare Stage 2 (discriminator) batch - requires Stage 1 results
python prepare_batches.py --stage 2 --dataset nuscenes --records /path/to/records --graphs /path/to/graphs

# Prepare QA batches - requires Stage 2 results
python prepare_batches.py --stage 3a --dataset nuscenes --records /path/to/records --graphs /path/to/graphs
python prepare_batches.py --stage 3b --dataset nuscenes --records /path/to/records --graphs /path/to/graphs
```

**Options:**
- `--stage`: Pipeline stage (1, 2, 3a, 3b)
- `--dataset`: Dataset to process (nuscenes, openscene, argoverse2)
- `--records`: Path to records directory
- `--graphs`: Path to graphs directory (required for stages 2+)
- `--output`: Output directory for batch files (default: batch_jobs)
- `--model`: Model to use (default: gpt-5.2)
- `--reasoning`: Reasoning level (default: none)
- `--prompt-version`: QA prompt version (default: v3)
- `--max-samples`: Maximum samples to process

### 2. `submit_batch.py`
Submits batch files to OpenAI and manages batch jobs.

```bash
# Submit a batch file
python submit_batch.py submit --file batch_jobs/nuscenes_stage1_generator.jsonl

# Check batch status
python submit_batch.py status --batch-id batch_xyz123

# List recent batches
python submit_batch.py list --limit 10

# Download completed batch results
python submit_batch.py download --batch-id batch_xyz123 --output batch_results
```

**Commands:**
- `submit`: Submit a new batch file
- `status`: Check status of a specific batch
- `list`: List recent batch jobs
- `download`: Download results from completed batch

### 3. `process_results.py`
Processes batch results and saves to appropriate output locations.

```bash
# Process Stage 1 results (saves to graphs directory)
python process_results.py --file batch_results/batch_xyz_results.jsonl --stage 1 --output /path/to/graphs

# Process Stage 2 results (saves refined graphs)
python process_results.py --file batch_results/batch_xyz_results.jsonl --stage 2 --output /path/to/graphs

# Process QA results (saves to qa_results directory)
python process_results.py --file batch_results/batch_xyz_results.jsonl --stage 3a --output /path/to/qa_results
python process_results.py --file batch_results/batch_xyz_results.jsonl --stage 3b --output /path/to/qa_results
```

**Output Files:**
- Stage 1: `{scene_name}_SAMPLED.json` (initial graph)
- Stage 2: `{scene_name}_SAMPLED_refined.json` (refined graph)
- Stage 3a: `qa_results/{scene_name}/active_qa.json`
- Stage 3b: `qa_results/{scene_name}/dormant_qa.json`
- Merged: `qa_results/{scene_name}/result.json`

### 4. `monitor_batches.py`
Monitor active batch jobs and display progress.

```bash
# Monitor all active batches
python monitor_batches.py

# Monitor specific batch with auto-refresh
python monitor_batches.py --batch-id batch_xyz123 --watch

# Check local batch info files
python monitor_batches.py --check-files

# Custom refresh interval
python monitor_batches.py --watch --interval 60
```

**Options:**
- `--batch-id`: Monitor specific batch
- `--watch`: Auto-refresh display
- `--interval`: Refresh interval in seconds (default: 30)
- `--check-files`: Check local batch info files
- `--verbose`: Show detailed information

### 5. `run_batch_pipeline.sh`
Comprehensive script that orchestrates the entire batch pipeline.

```bash
# Run complete pipeline
./run_batch_pipeline.sh nuscenes

# Resume from specific stage
./run_batch_pipeline.sh nuscenes --resume stage2

# Process limited scenes
./run_batch_pipeline.sh openscene --max-samples 50

# Custom configuration
./run_batch_pipeline.sh argoverse2 \
    --model gpt-5.2 \
    --reasoning none \
    --prompt-version v3 \
    --max-samples 100
```

**Options:**
- `--max-samples N`: Process only N samples
- `--resume STAGE`: Resume from stage (1, 2, 3a, 3b)
- `--model MODEL`: Model to use (default: gpt-5.2)
- `--reasoning LEVEL`: Reasoning level (default: none)
- `--prompt-version VER`: QA prompt version (default: v3)
- `--skip-download`: Skip downloading results

## Workflow

### Complete Pipeline Execution

1. **Prepare your environment:**
   ```bash
   export OPENAI_API_KEY="your-api-key"
   cd scripts/batch_processing
   ```

2. **Run the complete pipeline:**
   ```bash
   ./run_batch_pipeline.sh nuscenes --max-samples 100
   ```

3. **Monitor progress:**
   The script automatically monitors each batch and waits for completion before proceeding to the next stage.

### Manual Stage-by-Stage Execution

1. **Stage 1 - Generate initial graphs:**
   ```bash
   # Prepare batch
   python prepare_batches.py --stage 1 --dataset nuscenes --records /path/to/records

   # Submit batch
   python submit_batch.py submit --file batch_jobs/nuscenes_stage1_generator.jsonl

   # Monitor progress
   python monitor_batches.py --batch-id batch_xyz --watch

   # Download and process results
   python submit_batch.py download --batch-id batch_xyz
   python process_results.py --file batch_results/batch_xyz_results.jsonl --stage 1 --output /path/to/graphs
   ```

2. **Stage 2 - Refine graphs:**
   ```bash
   # Prepare batch (requires Stage 1 results)
   python prepare_batches.py --stage 2 --dataset nuscenes --records /path/to/records --graphs /path/to/graphs

   # Submit and monitor
   python submit_batch.py submit --file batch_jobs/nuscenes_stage2_discriminator.jsonl
   python monitor_batches.py --batch-id batch_abc --watch

   # Process results
   python submit_batch.py download --batch-id batch_abc
   python process_results.py --file batch_results/batch_abc_results.jsonl --stage 2 --output /path/to/graphs
   ```

3. **Stage 3 - Generate QA (parallel):**
   ```bash
   # Prepare both QA batches
   python prepare_batches.py --stage 3a --dataset nuscenes --records /path/to/records --graphs /path/to/graphs
   python prepare_batches.py --stage 3b --dataset nuscenes --records /path/to/records --graphs /path/to/graphs

   # Submit both batches (they run in parallel)
   python submit_batch.py submit --file batch_jobs/nuscenes_stage3a_active.jsonl
   python submit_batch.py submit --file batch_jobs/nuscenes_stage3b_dl.jsonl

   # Monitor both
   python monitor_batches.py  # Shows all active batches

   # Process results
   python process_results.py --file batch_results/batch_3a_results.jsonl --stage 3a --output /path/to/qa_results
   python process_results.py --file batch_results/batch_3b_results.jsonl --stage 3b --output /path/to/qa_results
   ```

## Directory Structure

```
batch_jobs/               # JSONL batch request files
├── nuscenes_stage1_generator.jsonl
├── nuscenes_stage2_discriminator.jsonl
├── nuscenes_stage3a_active.jsonl
└── nuscenes_stage3b_dl.jsonl

batch_results/            # Downloaded batch results
├── batch_xyz123_results.jsonl
├── batch_xyz123_errors.jsonl (if any)
└── ...

outputs/nuscenes/
├── graphs/              # Causal graphs
│   ├── scene1_SAMPLED.json
│   ├── scene1_SAMPLED_refined.json
│   └── ...
└── qa_results/          # QA generation results
    ├── scene1/
    │   ├── active_qa.json
    │   ├── dormant_qa.json
    │   └── result.json (merged)
    └── ...
```

## Cost Optimization

The batch API provides:
- **50% cost savings** compared to real-time API calls
- **24-hour completion window** for processing
- **Automatic retries** for failed requests
- **Parallel processing** within each stage

## Error Handling

- Failed requests are logged in error files (`batch_xyz_errors.jsonl`)
- The process_results.py script reports error breakdowns by type
- Individual scene failures don't stop the entire batch
- Resume capability allows continuing from any stage

## Tips

1. **Start small**: Test with `--max-samples 10` before processing entire datasets
2. **Monitor actively**: Use `--watch` flag for real-time progress updates
3. **Check errors**: Always review error files after batch completion
4. **Stage dependencies**: Ensure previous stages completed before starting dependent stages
5. **Parallel QA**: Stages 3a and 3b can run simultaneously since they're independent

## Integration with Existing Pipeline

The batch processing system is designed to complement the existing real-time pipeline:

- **Same output format**: Results are identical to real-time processing
- **Compatible with visualization**: Works with render_causal_graph.py
- **Shared utilities**: Uses common scene_utils.py module
- **Configurable**: Can be enabled/disabled in config.yaml

## Troubleshooting

**Batch stuck in "validating" status:**
- Usually resolves within 5-10 minutes
- If persists, cancel and resubmit

**Missing scenes in results:**
- Check error file for specific failures
- Rerun prepare_batches.py for failed scenes only

**Out of memory errors:**
- Reduce `--max-samples` parameter
- Process in smaller batches

**API rate limits:**
- Batch API has separate limits from real-time
- Contact OpenAI support for limit increases

## Future Enhancements

- [ ] Stage 4: Counterfactual trajectory generation
- [ ] Automatic error recovery and retry logic
- [ ] Progress dashboard with statistics
- [ ] Cost tracking and reporting
- [ ] Incremental processing (skip already processed scenes)
- [ ] Multi-model comparison batches