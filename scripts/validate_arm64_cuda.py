#!/usr/bin/env python3
"""
ARM64 CUDA Validation Script

Tests PyTorch CUDA functionality on ARM64 hardware (NVIDIA Grace Hopper, Grace Blackwell, Jetson).
This script verifies that PyTorch can:
1. Detect CUDA GPUs
2. Create tensors on CUDA devices
3. Perform basic operations (matrix multiplication)
4. Report correct architecture and device information

Exit codes:
  0: CUDA available and working
  1: CUDA not available
  2: CUDA available but operations failed
"""

import platform
import sys

import torch


def main():
    print("=" * 60)
    print("ARM64 CUDA Validation")
    print("=" * 60)

    # Print system information
    print(f"\nPlatform: {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")

    if not cuda_available:
        print("\n❌ CUDA is not available on this system")
        print("This script requires NVIDIA CUDA-capable hardware")
        return 1

    # Print CUDA information
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nDevice {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")

    # Verify architecture is ARM64
    arch = platform.machine()
    if arch not in ["aarch64", "arm64"]:
        print(f"\n⚠️  Warning: Expected ARM64 architecture, got {arch}")
    else:
        print(f"\n✓ Architecture confirmed: {arch}")

    # Test CUDA operations
    print("\n" + "=" * 60)
    print("Testing CUDA Operations")
    print("=" * 60)

    try:
        device = torch.device("cuda:0")
        print(f"\nUsing device: {device}")

        # Create test tensors
        print("\nCreating test tensors (1000x1000)...")
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)

        # Perform matrix multiplication
        print("Performing matrix multiplication on CUDA...")
        c = torch.matmul(a, b)

        # Verify result
        assert c.device.type == "cuda", "Result tensor not on CUDA device"
        assert c.shape == (1000, 1000), f"Unexpected result shape: {c.shape}"

        # Compute some statistics
        mean_val = c.mean().item()
        std_val = c.std().item()
        print(f"Result statistics: mean={mean_val:.4f}, std={std_val:.4f}")

        # Test synchronization
        torch.cuda.synchronize()
        print("CUDA synchronization successful")

        print("\n" + "=" * 60)
        print("✅ All CUDA tests passed!")
        print("=" * 60)
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ CUDA operations failed: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
