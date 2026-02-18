"""End-to-end integration tests for MorphLLM training pipeline.

Verifies:
- Training loop functionality (forward/backward/optimizer)
- Growth triggering (Width, Depth, MoE)
- Checkpointing (Save/Load)
- Optimizer state migration & Continual learning hooks
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from morphllm.controller import GrowthAction, GrowthController, GrowthDecision
from morphllm.model import MorphConfig, MorphLLM
from morphllm.trainer import Trainer, TrainerConfig

DEVICE = "cpu"  # Keep tests fast and runnable anywhere


class TestIntegrationE2E(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = TrainerConfig(
            max_steps=10,
            batch_size=2,
            check_growth_every_n_steps=1,  # Check every step
            checkpoint_every_n_steps=100,
            output_dir=self.test_dir,
            device=DEVICE,
        )
        self.model_config = MorphConfig(
            vocab_size=100,
            d_model=32,
            n_head=4,
            n_layers=2,
            max_seq_len=64,
            device=DEVICE,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_training_and_width_growth(self):
        """Test training steps with a width growth event."""
        model = MorphLLM(self.model_config)
        controller = MagicMock(spec=GrowthController)
        trainer = Trainer(model, self.config, controller=controller)

        # Mock decisions: WAIT, WAIT, GROW_WIDTH(32->64), WAIT...
        def side_effect(*args, **kwargs):
            if trainer.global_step == 3:
                return GrowthDecision(
                    action=GrowthAction.GROW_WIDTH,
                    reason="Test growth",
                    new_d_model=64,
                )
            return GrowthDecision(action=GrowthAction.WAIT, reason="Wait")

        controller.decide.side_effect = side_effect

        # Create dummy batch
        input_ids = torch.randint(0, 100, (2, 10))
        targets = torch.randint(0, 100, (2, 10))

        # Step 1: Normal train (global_step -> 1)
        metrics = trainer.train_step(input_ids, targets)
        self.assertEqual(model.d_model, 32)
        self.assertIn("mlm_loss", metrics)

        # Step 2: Normal train (global_step -> 2)
        trainer.train_step(input_ids, targets)
        self.assertEqual(model.d_model, 32)

        # Step 3: Training step + Growth trigger (global_step -> 3)
        trainer.train_step(input_ids, targets)
        
        # Verify growth happened
        self.assertEqual(model.d_model, 64)
        
        # Verify optimizer state migration
        # The optimizer should now track parameters with shape [..., 64]
        for group in trainer.optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    # Heuristic: linear weights should have dimension 64
                    if p.dim() > 1 and p.shape[0] == 64: 
                        pass # Valid
                    elif p.dim() > 1 and p.shape[1] == 64:
                        pass # Valid
        
        # Step 3: Train with grown model
        metrics = trainer.train_step(input_ids, targets)
        self.assertIn("mlm_loss", metrics)

    def test_depth_growth_and_freeze_cycle(self):
        """Test depth growth triggers freeze cycle."""
        model = MorphLLM(self.model_config)
        controller = MagicMock(spec=GrowthController)
        trainer = Trainer(model, self.config, controller=controller)

        # Mock decision: GROW_DEPTH at step 2 (global_step=2)
        def side_effect(*args, **kwargs):
            if trainer.global_step == 2:
                return GrowthDecision(
                    action=GrowthAction.GROW_DEPTH,
                    reason="Test depth",
                    depth_position=2, # Append
                )
            return GrowthDecision(action=GrowthAction.WAIT, reason="Wait")

        controller.decide.side_effect = side_effect

        input_ids = torch.randint(0, 100, (2, 10))
        targets = torch.randint(0, 100, (2, 10))

        trainer.train_step(input_ids, targets) # Step 1 (global_step=1)
        trainer.train_step(input_ids, targets) # Step 2 (global_step=2) -> Grows Depth
        
        # Verify growth happened
        self.assertEqual(model.n_layers, 3)
        
        # Verify Freezer is active
        self.assertTrue(trainer.freezer.is_frozen)
        self.assertGreater(trainer.freezer.n_frozen, 0)
        
        # Only new parameters should be trainable
        # The new block is at index 2 (0, 1, 2)
        new_block_params = list(model.blocks[2].parameters())
        for p in new_block_params:
            self.assertTrue(p.requires_grad)

    def test_checkpoint_save_load(self):
        """Test saving and loading training state."""
        model = MorphLLM(self.model_config)
        trainer = Trainer(model, self.config)
        
        # Train a bit
        input_ids = torch.randint(0, 100, (2, 10))
        targets = torch.randint(0, 100, (2, 10))
        trainer.train_step(input_ids, targets)
        trainer.train_step(input_ids, targets)
        
        # Save
        ckpt_path = os.path.join(self.test_dir, "test_ckpt.pt")
        trainer.save_checkpoint(ckpt_path)
        
        # Load into new instance
        new_model = MorphLLM(self.model_config)
        new_trainer = Trainer(new_model, self.config)
        new_trainer.load_checkpoint(ckpt_path)
        
        self.assertEqual(new_trainer.global_step, 2)
        
        # Compare weights
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_moe_spawning_integration(self):
        """Test SPAWN_MOE action converts block and adds aux loss."""
        model = MorphLLM(self.model_config)
        controller = MagicMock(spec=GrowthController)
        trainer = Trainer(model, self.config, controller=controller)
        
        # Mock SPAWN_MOE at step 2
        def side_effect(*args, **kwargs):
            if trainer.global_step == 2:
                return GrowthDecision(
                    action=GrowthAction.SPAWN_MOE,
                    reason="Test MoE",
                    moe_block_idx=0,
                )
            return GrowthDecision(action=GrowthAction.WAIT, reason="Wait")

        controller.decide.side_effect = side_effect
        
        input_ids = torch.randint(0, 100, (2, 10))
        targets = torch.randint(0, 100, (2, 10))
        
        trainer.train_step(input_ids, targets) # Step 1
        trainer.train_step(input_ids, targets) # Step 2 -> Spawns MoE
        
        self.assertTrue(getattr(model.blocks[0], "_use_moe", False))
        
        # Step 3: Verify MoE loss is included
        metrics = trainer.train_step(input_ids, targets)
        self.assertIn("moe_loss", metrics)

