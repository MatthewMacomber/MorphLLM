# MorphLLM — The Adaptive Morphogenic Engine

MorphLLM is a proof-of-concept language model that **grows organically** during training. Instead of training a fixed-size model from scratch, MorphLLM starts small (seed model) and expands its capacity (width, depth, sparsity) based on real-time resource monitoring and loss plateau detection.

**Key Features:**
- **Dynamic Growth**: 
  - **Net2WiderNet**: Expands embedding dimension (`d_model`) and FFN width.
  - **Net2DeeperNet**: Inserts identity-initialized layers (depth growth).
  - **Sparse Scaling**: Converts dense FFNs to Mixture-of-Experts (MoE) when compute allows.
- **Resource Awareness**: Monitors VRAM, RAM, and Compute Utilization to decide *when* and *how* to grow.
- **Anti-Forgetting**:
  - **Rehearsal (SSR)**: Mixes past model generations into training batches.
  - **Self-Distillation (SDFT)**: Uses an EMA teacher to anchor the student model.
  - **Freeze-Expand-Tune**: Protects learned representations during structural adaptation.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### 1. Train a Seed Model

Start training a small model. It will grow automatically!

```bash
morphllm train --data data/input.txt --steps 1000 --output-dir checkpoints
```

### 2. Monitor Status

Inspect a saved checkpoint to see how much the model has grown.

```bash
morphllm status --checkpoint checkpoints/checkpoint_500.pt
```

Output example:
```
--- Checkpoint Status ---
Global Step: 500
Architecture:
  d_model: 128      (started at 64)
  n_layers: 4       (started at 2)
  n_head: 4
  vocab_size: 1000
```

### 3. Manual Growth (Optional)

You can mostly rely on the automatic controller, but you can also inspect logic via tests or modify `GrowthConfig`.

## Configuration

Control growth behavior via `GrowthConfig` in `morphllm/controller.py`:

- `vram_ceiling_pct`: Max VRAM usage (default 85%).
- `plateau_window`: Steps to detect loss plateau (default 50).
- `enable_moe`: Whether to allow spawning experts.

## Architecture

- `MorphLLM`: Main container (Embeddings + Stack of Blocks + Head).
- `DynamicTransformerBlock`: Growable Attention + FFN/MoE.
- `GrowthController`: The "brain" that monitors resources and loss.
- `Trainer`: Orchestrates training, growth, and migration.

## Testing

Run the full test suite (including E2E integration):

```bash
pytest tests/ -v
```

## Next Steps

- `Distributed Training`: Support multi-GPU via FSDP (currently single-GPU/CPU).
- `Advanced MoE`: Implement expert parallelism.
- `Quantization`: Implement the PRUNE action for low-VRAM scenarios.