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

## Other Critical Dependencies

### bitsandbytes (Quantization Library)

**Current version in project:** 0.47.0
**Latest stable:** 0.49.1 (released Jan 8, 2026)

#### NVIDIA ARM64 (Linux aarch64) + CUDA

**Status:** ✓ Fully Available

**Installation:**
```bash
pip install bitsandbytes==0.49.1
```

**Wheel availability:**
- PyPI has official wheels: `bitsandbytes-0.49.1-py3-none-manylinux_2_24_aarch64.whl`
- Built with GCC 11.4, minimum glibc 2.24
- CUDA support for Linux aarch64 (sbsa architecture)
- Supports NVIDIA ARM64 platforms: Grace Hopper (GH200), NVL32/NV72

**Preview/development builds:**
```bash
pip install --force-reinstall https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_aarch64.whl
```

**References:**
- [bitsandbytes PyPI](https://pypi.org/project/bitsandbytes/)
- [bitsandbytes Releases](https://github.com/bitsandbytes-foundation/bitsandbytes/releases)
- [Installation Guide](https://huggingface.co/docs/bitsandbytes/main/en/installation)
- [Issue #1437: aarch64 whl in PyPi](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1437)

---

#### Apple Silicon (macOS ARM64) + MPS

**Status:** ⚠️ Limited Support (Experimental)

**Installation:**
```bash
pip install bitsandbytes==0.49.1
```

**Wheel availability:**
- PyPI has macOS ARM64 wheels: `bitsandbytes-0.49.1-py3-none-macosx_14_0_arm64.whl`
- Can be installed, but functionality is limited

**Critical Limitations:**
- **CUDA-only features do NOT work on macOS** - bitsandbytes core quantization depends on CUDA
- MPS backend not supported for 8-bit optimizations
- Experimental support only - not recommended for production use
- Installation succeeds but runtime errors expected for quantization operations

**Recommended Alternatives for Apple Silicon:**
- **MLX**: Apple's ML framework optimized for Apple Silicon
- **llama.cpp**: Efficient CPU/Metal inference with quantization
- **Ollama**: Local LLM runtime with built-in quantization
- **GGUF format**: Quantized model format with better macOS support

**Fallback Strategy:**
- Make bitsandbytes optional dependency for macOS ARM64
- Disable quantization features when running on Apple Silicon
- Use platform markers in pyproject.toml: `bitsandbytes==0.49.1 ; sys_platform != 'darwin' or platform_machine != 'arm64'`

**References:**
- [Issue #1460: bitsandbytes for macos M1/M2/M3 chips](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1460)
- [Issue #252: Support for Apple silicon](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/252)
- [Discussion #1340: Multi-backend support: Apple Silicon / Mac](https://github.com/bitsandbytes-foundation/bitsandbytes/discussions/1340)

---

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
3. ✓ bitsandbytes ARM64: Available for NVIDIA ARM64, make optional for Apple Silicon
4. TODO: Research deepspeed ARM64 availability (Task 7wg)
5. TODO: Research triton ARM64 availability (Task 9mt)
6. TODO: Update pyproject.toml with platform markers for bitsandbytes
7. TODO: Test installation on both platforms (Tasks 568, pol)

---

## Summary

| Dependency | NVIDIA ARM64+CUDA | Apple Silicon+MPS | Notes |
|------------|-------------------|-------------------|-------|
| **PyTorch** | ✓ Available | ✓ Fully Supported | Use cu128 index for NVIDIA ARM64 |
| **torchvision** | ✓ Available | ✓ Supported | Follows PyTorch installation |
| **bitsandbytes** | ✓ Available (v0.49.1+) | ⚠️ Limited (make optional) | CUDA-only features don't work on macOS |
| **deepspeed** | ? Research needed | ? Research needed | Distributed training |
| **triton** | ? Research needed | ? Research needed | GPU kernels (via PyTorch?) |

**Legend:**
- ✓ = Confirmed available and functional
- ⚠️ = Available but with limitations (see notes)
- ✗ = Known to be unavailable
- ? = Research required
