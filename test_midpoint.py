#!/usr/bin/env python
"""Test script for the randomized midpoint solver."""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate import ablation_sampler


# Create a simple mock network for testing
class MockNetwork:
    def __init__(self):
        self.sigma_min = 0.002
        self.sigma_max = 80

    def round_sigma(self, sigma):
        return sigma

    def __call__(self, x, sigma, class_labels=None):
        # Simple linear denoising for testing
        return x * 0.9


def test_midpoint_solver():
    """Test that the midpoint solver runs without errors."""
    print("Testing randomized midpoint solver...")

    # Initialize mock network and test inputs
    net = MockNetwork()
    batch_size = 2
    channels = 3
    img_size = 32

    # Create random latents
    latents = torch.randn(batch_size, channels, img_size, img_size)

    # Test with different solvers
    solvers = ["euler", "heun", "midpoint"]

    for solver in solvers:
        print(f"\nTesting {solver} solver...")
        try:
            result = ablation_sampler(
                net=net,
                latents=latents,
                num_steps=5,
                solver=solver,
                sigma_min=0.002,
                sigma_max=80,
            )
            print(f"  ✓ {solver} solver completed successfully")
            print(f"    Output shape: {result.shape}")
            print(f"    Output mean: {result.mean().item():.4f}")
            print(f"    Output std: {result.std().item():.4f}")
        except Exception as e:
            print(f"  ✗ {solver} solver failed: {e}")
            return False

    print("\n✓ All solvers tested successfully!")
    return True


if __name__ == "__main__":
    success = test_midpoint_solver()
    sys.exit(0 if success else 1)
