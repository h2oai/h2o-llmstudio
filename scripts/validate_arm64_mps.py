#!/usr/bin/env python3
"""
ARM64 MPS (Metal Performance Shaders) Validation Script

Tests PyTorch MPS functionality on Apple Silicon (M1, M2, M3, M4).
This script verifies that PyTorch can:
1. Detect MPS backend
2. Create tensors on MPS devices
3. Perform basic operations (matrix multiplication)
4. Report correct architecture and device information

Exit codes:
  0: MPS available and working
  1: MPS not available
  2: MPS available but operations failed
"""

import platform
import sys

import torch


def main():
    print("=" * 60)
    print("ARM64 MPS (Metal) Validation")
    print("=" * 60)

    # Print system information
    print(f"\nPlatform: {platform.machine()}")
    print(f"System: {platform.system()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    # Check MPS availability
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"\nMPS available: {mps_available}")

    if not mps_available:
        print("\n❌ MPS is not available on this system")
        print("This script requires Apple Silicon hardware (M1/M2/M3/M4)")
        if platform.system() != "Darwin":
            print(f"Current system: {platform.system()} (expected Darwin/macOS)")
        return 1

    # Verify architecture is ARM64
    arch = platform.machine()
    if arch not in ["arm64"]:
        print(f"\n⚠️  Warning: Expected ARM64 architecture, got {arch}")
    else:
        print(f"\n✓ Architecture confirmed: {arch}")

    # Verify we're on macOS
    if platform.system() != "Darwin":
        print(f"\n⚠️  Warning: Expected macOS (Darwin), got {platform.system()}")
    else:
        print(f"✓ System confirmed: macOS")

    # Test MPS operations
    print("\n" + "=" * 60)
    print("Testing MPS Operations")
    print("=" * 60)

    try:
        device = torch.device("mps")
        print(f"\nUsing device: {device}")

        # Create test tensors
        print("\nCreating test tensors (1000x1000)...")
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)

        # Perform matrix multiplication
        print("Performing matrix multiplication on MPS...")
        c = torch.matmul(a, b)

        # Verify result
        assert c.device.type == "mps", "Result tensor not on MPS device"
        assert c.shape == (1000, 1000), f"Unexpected result shape: {c.shape}"

        # Compute some statistics (need to move to CPU for some operations)
        mean_val = c.mean().item()
        std_val = c.std().item()
        print(f"Result statistics: mean={mean_val:.4f}, std={std_val:.4f}")

        # Test CPU <-> MPS transfer
        print("\nTesting CPU <-> MPS transfer...")
        cpu_tensor = torch.randn(100, 100)
        mps_tensor = cpu_tensor.to(device)
        back_to_cpu = mps_tensor.cpu()
        assert torch.allclose(cpu_tensor, back_to_cpu), "CPU <-> MPS transfer mismatch"
        print("CPU <-> MPS transfer successful")

        # Test MPS built-in operations
        print("\nTesting MPS built-in operations...")
        x = torch.randn(100, 100, device=device)

        # Test various operations
        _ = x.sum()
        _ = x.mean()
        _ = torch.nn.functional.relu(x)
        _ = torch.nn.functional.softmax(x, dim=1)

        print("MPS built-in operations successful")

        print("\n" + "=" * 60)
        print("✅ All MPS tests passed!")
        print("=" * 60)
        print("\nNote: MPS has some limitations compared to CUDA:")
        print("  - Single GPU only (no multi-GPU support)")
        print("  - Some operations may fall back to CPU")
        print("  - flash-attn and bitsandbytes not available")
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ MPS operations failed: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
