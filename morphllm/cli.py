"""Command-line interface for MorphLLM.

Commands:
- train: Start training a model (from scratch or checkpoint).
- status: Print model architecture and training stats from a checkpoint.
- grow: Manually trigger a growth step on a saved checkpoint.
"""

import argparse
import json
import logging
import os
import sys
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from morphllm.model import MorphConfig, MorphLLM
from morphllm.trainer import Trainer, TrainerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("morphllm.cli")


class TextDataset(IterableDataset):
    """Simple iterable dataset from a text file."""

    def __init__(self, path: str, seq_len: int, vocab_size: int):
        self.path = path
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if not os.path.exists(self.path):
            # Generate random data if file doesn't exist (for testing)
            while True:
                data = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
                yield data[:-1], data[1:]
        
        with open(self.path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Very simple character-level tokenization for demo
        # In real usage, use a proper tokenizer
        tokens = [ord(c) % self.vocab_size for c in text]
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        for i in range(0, len(tokens) - self.seq_len, self.seq_len):
            chunk = tokens[i : i + self.seq_len + 1]
            if len(chunk) < self.seq_len + 1:
                break
            yield chunk[:-1], chunk[1:]


def train_command(args):
    """Run the training loop."""
    logger.info(f"Starting training with config: {args}")

    # 1. Config
    if args.config:
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        model_config = MorphConfig(**config_dict.get("model", {}))
        trainer_config = TrainerConfig(**config_dict.get("trainer", {}))
    else:
        model_config = MorphConfig(vocab_size=1000, d_model=64, n_layers=2)
        trainer_config = TrainerConfig(max_steps=args.steps, output_dir=args.output_dir)

    # 2. Model
    model = MorphLLM(model_config)
    logger.info(f"Initialized model with {model.num_parameters} parameters.")

    # 3. Data
    dataset = TextDataset(args.data, model_config.max_seq_len, model_config.vocab_size)
    # Batch size logic would go here, simple loader for now
    dataloader = DataLoader(dataset, batch_size=trainer_config.batch_size)

    # 4. Trainer
    trainer = Trainer(model, trainer_config)

    # 5. Resume?
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # To resume, we need to load the model structure first if it grew.
        # This is tricky because standard torch.load(state_dict) fails on size mismatch.
        # A robust impl would load the config from checkpoint first, init model, then load state.
        
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_config = checkpoint.get("model_config")
        if saved_config:
            # Re-init model with saved config (which captures growth)
            model = MorphLLM(saved_config)
            trainer = Trainer(model, trainer_config)
        
        trainer.load_checkpoint(args.resume)

    # 6. Run
    trainer.train_loop(dataloader)


def status_command(args):
    """Inspect a checkpoint."""
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config: MorphConfig = checkpoint.get("model_config")
    step = checkpoint.get("global_step", 0)
    
    print(f"--- Checkpoint Status: {args.checkpoint} ---")
    print(f"Global Step: {step}")
    if config:
        print(f"Architecture:")
        print(f"  d_model: {config.d_model}")
        print(f"  n_layers: {config.n_layers}")
        print(f"  n_head: {config.n_head}")
        print(f"  vocab_size: {config.vocab_size}")
    else:
        print("No model config found in checkpoint.")


def main():
    parser = argparse.ArgumentParser(description="MorphLLM CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # TRAIN
    train_parser = subparsers.add_parser("train", help="Start training")
    train_parser.add_argument("--data", type=str, default="data.txt", help="Path to text file")
    train_parser.add_argument("--steps", type=int, default=100, help="Max training steps")
    train_parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    train_parser.add_argument("--config", type=str, help="Path to config JSON")
    train_parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
    train_parser.set_defaults(func=train_command)

    # STATUS
    status_parser = subparsers.add_parser("status", help="Check model status")
    status_parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    status_parser.set_defaults(func=status_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
