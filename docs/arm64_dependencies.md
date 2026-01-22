# ARM64 Dependency Research

**Research Date:** 2026-01-21
**Status:** Active Research
**Purpose:** Document ARM64 compatibility of critical dependencies for h2o-llmstudio

## PyTorch ARM64 Support

### NVIDIA ARM64 + CUDA

**Status:** ✓ Available (with limitations)

**Installation:**
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

**Index URL for pyproject.toml:**
```toml
[[tool.uv.index]]
name = "pytorch-arm64-cuda"
url = "https://download.pytorch.org/whl/cu128/"
explicit = true
```

**Platform Support:**
- NVIDIA Grace Hopper (GH200)
- NVIDIA Grace Blackwell (GB200)
- ARM64 servers with NVIDIA GPUs
- Linux aarch64 + CUDA

**Limitations:**
- No official ARM64 wheels on main PyPI (CPU-only available)
- Must use cu128 index explicitly
- Some build pipeline issues reported as of June 2025 for nightly builds
- Feature requests for official ARM64 GPU support still open

**References:**
- [PyTorch Wheel Variants Blog](https://pytorch.org/blog/pytorch-wheel-variants/)
- [PyTorch Issue #134790: Build Linux aarch64 wheels with CUDA](https://github.com/pytorch/pytorch/issues/134790)
- [PyTorch Issue #160162: aarch64 GPU wheel release in pypi](https://github.com/pytorch/pytorch/issues/160162)
- [PyTorch Forums: pytorch-cuda=12 for arm64 and Grace Hopper](https://discuss.pytorch.org/t/conda-pytorch-cuda-12-for-arm64-and-grace-hopper/203285)

---

### Apple Silicon + Metal (MPS)

**Status:** ✓ Fully Supported

**Installation:**
```bash
pip3 install torch torchvision
```

**Notes:**
- Standard PyTorch installation includes MPS backend
- No special index URL required
- Works on all Apple Silicon Macs (M1/M2/M3/M4)

**System Requirements:**
- macOS 12.3 or later
- MPS backend supported since PyTorch 1.12 (stable)

**GPU Backend:**
- Metal Performance Shaders (MPS) for GPU acceleration
- Unified memory architecture provides direct GPU memory access
- CUDA is NOT available on macOS

**Verification:**
```python
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print("MPS available:", x)
else:
    print("MPS device not found.")
```

**Limitations:**
- Some PyTorch operations not yet implemented in MPS
- Single GPU only (no multi-GPU support)
- Cannot use CUDA-specific packages

**References:**
- [Accelerated PyTorch training on Mac - Apple Developer](https://developer.apple.com/metal/pytorch/)
- [Introducing Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
- [PyTorch Apple Silicon Support](https://docs.pytorch.org/serve/hardware_support/apple_silicon_support.html)
- [HuggingFace: Accelerated PyTorch Training on Mac](https://huggingface.co/docs/accelerate/en/usage_guides/mps)

---

## Recommended PyTorch Configuration

### For pyproject.toml

```toml
[project]
dependencies = [
    "torch==2.8.0",  # or latest stable
    # ... other deps
]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu128/"
explicit = true

# Note: For Apple Silicon, standard PyPI install works
# For NVIDIA ARM64, use cu128 index
```

### Platform-Specific Installation Strategy

**NVIDIA ARM64 (Linux aarch64 + CUDA):**
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128/
```

**Apple Silicon (macOS arm64):**
```bash
pip install torch  # standard PyPI, includes MPS
```

---

## Other Critical Dependencies (To Be Researched)

### bitsandbytes
- **Status:** Research required (Task h2o-llmstudio-8tf)
- **Current version:** 0.47.0
- **Known issue:** No macOS ARM64 wheel on PyPI (see error in project setup)

### deepspeed
- **Status:** Research required (Task h2o-llmstudio-7wg)
- **Current version:** 0.17.5

### triton
- **Status:** Research required (Task h2o-llmstudio-9mt)
- **Current version:** 3.3.1
- **Note:** May be bundled with PyTorch

---

## Next Steps

1. ✓ PyTorch ARM64+CUDA: Use cu128 index
2. ✓ PyTorch Apple Silicon: Standard install with MPS
3. TODO: Research bitsandbytes ARM64 availability (Task 8tf)
4. TODO: Research deepspeed ARM64 availability (Task 7wg)
5. TODO: Research triton ARM64 availability (Task 9mt)
6. TODO: Test installation on both platforms (Tasks 568, pol)

---

## Summary

| Dependency | NVIDIA ARM64+CUDA | Apple Silicon+MPS | Notes |
|------------|-------------------|-------------------|-------|
| **PyTorch** | ✓ Available | ✓ Fully Supported | Use cu128 index for NVIDIA ARM64 |
| **torchvision** | ✓ Available | ✓ Supported | Follows PyTorch installation |
| **bitsandbytes** | ? Research needed | ✗ No wheel | Quantization library |
| **deepspeed** | ? Research needed | ? Research needed | Distributed training |
| **triton** | ? Research needed | ? Research needed | GPU kernels (via PyTorch?) |

**Legend:**
- ✓ = Confirmed available
- ✗ = Known to be unavailable
- ? = Research required
