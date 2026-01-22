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

## NVIDIA CUDA Library Dependencies (ARM64)

**Research Date:** 2026-01-22
**Status:** ✓ Available

### Overview

NVIDIA CUDA library packages (nvidia-cuda-*, nvidia-cudnn-*, nvidia-cublas-*, etc.) are **transitive dependencies** of PyTorch. They are automatically resolved when installing PyTorch from the cu128 index and should **NOT** be explicitly listed in pyproject.toml.

### ARM64 Availability

All major NVIDIA CUDA packages now provide ARM64 (aarch64) wheels on PyPI:

| Package | Latest Version (Jan 2026) | ARM64 Support | Notes |
|---------|---------------------------|---------------|-------|
| nvidia-cuda-runtime-cu12 | 12.9.79 | ✓ Yes | Available since June 2025 |
| nvidia-cudnn-cu12 | 9.18.0.77 | ✓ Yes | Latest Jan 16, 2026 |
| nvidia-nccl-cu12 | 2.26.2 | ✓ Yes | Multi-GPU communication |
| nvidia-cuda-nvrtc-cu12 | 12.9.86 | ✓ Yes | Runtime compilation |
| nvidia-cublas-cu12 | 12.6.4.1 | ✓ Yes | Linear algebra |
| nvidia-nvshmem-cu12 | 3.5.19 | ✓ Yes | Shared memory |

### PyTorch 2.7.1 CUDA Dependencies

PyTorch 2.7.1 with CUDA 12.8 (cu128) on ARM64 automatically includes:

```
nvidia-cuda-nvrtc-cu12==12.8.61
nvidia-cuda-runtime-cu12==12.8.57
nvidia-cuda-cupti-cu12==12.8.57
nvidia-cudnn-cu12==9.7.1.26
nvidia-nccl-cu12==2.26.2
```

### Platform-Specific Resolution

The requirements.txt file may show different CUDA library versions depending on the platform where it was generated:

- **x86_64 systems**: May resolve to CUDA 12.6.x dependencies
- **ARM64 systems**: Will resolve to CUDA 12.8.x dependencies

This is **expected behavior** - uv resolves platform-specific wheels automatically.

### Configuration

**No action required** in pyproject.toml. The CUDA libraries are pulled transitively from PyTorch:

```toml
dependencies = [
    "torch==2.7.1",
    # CUDA dependencies resolved automatically
]

[tool.uv]
# Use unsafe-best-match to allow selecting best version across all indexes
# This is required because the PyTorch cu128 index may have older versions
# of common packages (e.g., requests==2.28.1) that conflict with requirements
# from other packages (e.g., datasets>=4.4.1 requires requests>=2.32.2)
index-strategy = "unsafe-best-match"

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu128/"
# CUDA libs available on both cu128 index and PyPI
```

### Dependency Resolution

The PyTorch cu128 index contains some common packages (like `requests`) at older versions than required by other dependencies. Using `index-strategy = "unsafe-best-match"` allows uv to select the best matching version from any index (PyPI or cu128), resolving version conflicts while still getting PyTorch with CUDA support.

### Verification

To verify CUDA libraries are correctly installed on ARM64:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected output on ARM64 with CUDA:
```
CUDA available: True
CUDA version: 12.8
```

### References

- [nvidia-cuda-runtime-cu12 on PyPI](https://pypi.org/project/nvidia-cuda-runtime-cu12/)
- [nvidia-cudnn-cu12 on PyPI](https://pypi.org/project/nvidia-cudnn-cu12/)
- [PyTorch 2.7 Release Notes](https://pytorch.org/blog/pytorch-2-7/)
- [AWS Deep Learning Containers PyTorch 2.7](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-pytorch-2-7-training-sagemaker.html)

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

### deepspeed (Distributed Training Library)

**Current version in project:** 0.17.5
**Latest stable:** 0.17.5 (as of Jan 2026)

#### NVIDIA ARM64 (Linux aarch64) + CUDA

**Status:** ⚠️ Build from Source Required

**Official Support:**
- No official ARM64 wheels on PyPI
- DeepSpeed officially supports: Intel Xeon CPU, Intel Data Center Max Series XPU, Intel Gaudi HPU, Huawei Ascend NPU
- Primary testing on NVIDIA GPUs (Pascal, Volta, Ampere, Hopper) - x86_64 only
- **Linux ARM64/aarch64 not officially listed** as supported platform

**Build Requirements:**
- PyTorch >= 1.9 (must be installed first)
- CUDA or ROCm compiler (nvcc or hipcc) required
- DeepSpeed includes C++/CUDA extensions that build just-in-time (JIT)
- Significant compilation time and dependencies

**Known Issues:**
- Issue #950: Build fails on AArch64 systems (reported on Fedora 33)
- Some users attempting to run on Jetson Orin AGX (ARM64) as of Aug 2025
- No prebuilt wheels available from any source (piwheels, third-party repos)

**Fallback Strategy:**
- Make deepspeed optional dependency for ARM64
- Use platform markers: `deepspeed==0.17.5 ; platform_machine != 'aarch64' and platform_machine != 'arm64'`
- Disable distributed training features on ARM64
- Alternative: Use PyTorch native DistributedDataParallel (DDP) instead

**References:**
- [deepspeed PyPI](https://pypi.org/project/deepspeed/)
- [DeepSpeed Installation Details](https://www.deepspeed.ai/tutorials/advanced-install/)
- [Issue #950: build fails on AArch64](https://github.com/microsoft/DeepSpeed/issues/950)
- [Issue #5308: build prebuilt wheels (REQUEST)](https://github.com/deepspeedai/DeepSpeed/issues/5308)

---

#### Apple Silicon (macOS ARM64) + MPS

**Status:** ✗ Not Supported

**Official Position:**
- DeepSpeed does not officially support macOS or Apple Silicon
- Not listed in supported platforms documentation

**Critical Limitations:**
- Requires CUDA or ROCm compiler (nvcc/hipcc) - **not available on macOS**
- C++/CUDA extensions cannot be compiled without CUDA toolkit
- MPS backend not supported by DeepSpeed
- Features like bitsandbytes, QLoRA, and DeepSpeed explicitly listed as "not available on M-series Macs"

**Community Efforts:**
- Issue #130: DeepSpeed install on Mac (opened but no resolution)
- Issue #1580: M1 Max support request (Nov 2021, not implemented)
- Issue #3364: Unable to install deepspeed with M2 arm64 (reported bug)
- PR #3907: Create accelerator for Apple Silicon GPU (attempted but not merged as of Jan 2026)

**Recommended Alternatives for macOS:**
- **PyTorch DDP**: Native distributed training without DeepSpeed
- **Apple MLX**: Apple's ML framework for Apple Silicon
- **PyTorch MPS backend**: GPU acceleration without distributed features
- **Remove deepspeed dependency** for macOS builds

**Fallback Strategy:**
- Use platform markers: `deepspeed==0.17.5 ; sys_platform != 'darwin'`
- Exclude deepspeed entirely from macOS installations
- Disable all DeepSpeed-dependent features in code when on macOS

**References:**
- [Issue #130: DeepSpeed install in Mac](https://github.com/microsoft/DeepSpeed/issues/130)
- [Issue #1580: M1 Max support REQUEST](https://github.com/microsoft/DeepSpeed/issues/1580)
- [Issue #3364: Unable to install deepspeed with M2 arm64](https://github.com/deepspeedai/DeepSpeed/issues/3364)
- [PR #3907: Apple Silicon GPU Acceleration](https://github.com/deepspeedai/DeepSpeed/pull/3907)
- [DeepSpeed Getting Started](https://www.deepspeed.ai/getting-started/)

---

### triton (GPU Kernel Compiler)

**Current version in project:** 3.3.1 (transitive dependency from PyTorch)
**Latest stable:** 3.5.1 (with ARM64 wheels as of Jan 2026)

#### NVIDIA ARM64 (Linux aarch64) + CUDA

**Status:** ⚠️ Emerging Support (wheels available, integration incomplete)

**Official Wheels:**
- PyPI has ARM64 wheels as of triton 3.5.1
- `triton-3.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`
- Major improvement from previous lack of aarch64 support

**Known Limitations:**
- AWS Deep Learning Containers (PyTorch 2.6, 2.7) still document: "There is no official Triton distribution for ARM64/aarch64 yet"
- Some `torch.compile` workloads fail with: `RuntimeError: Cannot find a working triton installation`
- Integration with PyTorch ARM64 CUDA builds still being refined

**Recent Developments:**
- Issue #147857: Triton aarch64 and SBSA support for GH200, Jetson Thor
- NVIDIA merging SBSA (Server Base System Architecture) and ARM64 together
- PyTorch 2.9 includes "Expanded and optimized convolution, activation, and quantized ops on AArch64"

**Historical Context:**
- Issue #130558 (July 2024): Triton not built for aarch64, making torch.compile unavailable on Grace-Hopper and Graviton+GPU
- Situation improved significantly in late 2024/early 2025

**Installation:**
```bash
pip install triton==3.5.1  # ARM64 wheels available
```

**Impact on flash-attn:**
- Flash Attention depends on triton for efficient attention kernels
- ARM64 support for triton enables potential flash-attn ARM64 support
- As of Jan 2026, integration still evolving

**References:**
- [triton PyPI](https://pypi.org/project/triton/)
- [Issue #130558: Build Triton for aarch64](https://github.com/pytorch/pytorch/issues/130558)
- [Issue #147857: Triton aarch64 and triton sbsa](https://github.com/pytorch/pytorch/issues/147857)
- [AWS DLC PyTorch 2.7 ARM64 Training](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-pytorch-2-7-arm64-training-ec2.html)

---

#### Apple Silicon (macOS ARM64) + MPS

**Status:** ✗ Not Supported (CUDA-only)

**Official Position:**
- No official Apple Silicon support as of Jan 2026
- Triton compiler architecture is "quite tailored to NVIDIA" GPUs
- Requires CUDA device for full functionality

**Technical Limitations:**
- Triton targets NVIDIA PTX (Parallel Thread Execution) assembly
- Designed for CUDA programming model, not Metal/MPS
- No backend for Metal Performance Shaders
- Cannot generate Metal shader code

**Community Efforts:**
- Issue #3443: Build Triton on MacOS with Apple silicon
- Some experimental builds possible (CPU-only, many tests fail)
- Requires build system modifications, non-trivial compilation
- Even when compiled, only runs CPU code (no GPU acceleration)
- No substantial upstream PRs for Apple Silicon support

**PyTorch Alternative:**
- PyTorch implemented **native Metal codegen** in TorchInductor (2024-2025)
- Metal codegen replaces Triton's role for PyTorch on macOS
- Users get `torch.compile` optimization without Triton dependency
- Performance improvements via Metal-native code generation

**The Gap:**
- No Python-based custom kernel authoring tool on macOS (equivalent to CUDA/Triton)
- Users must write Metal shaders directly or use PyTorch's built-in ops
- Libraries like Numba/Triton not available for GPU work on Apple Silicon

**Fallback Strategy:**
- Let triton install on macOS (if PyTorch pulls it), but expect limited functionality
- PyTorch's Metal backend handles optimization without Triton
- Disable Triton-dependent features (flash-attn) on macOS
- Use PyTorch native ops instead of custom triton kernels

**References:**
- [Issue #3443: Build Triton on MacOS with Apple silicon](https://github.com/triton-lang/triton/issues/3443)
- [Does TRITON work on Apple Silicon?](https://doesitarm.com/app/triton)
- [Issue #1465: Package does not exist on macOS (intel)](https://github.com/triton-lang/triton/issues/1465)
- [Triton Installation Documentation](https://triton-lang.org/main/getting-started/installation.html)

---

## Next Steps

1. ✓ PyTorch ARM64+CUDA: Use cu128 index
2. ✓ PyTorch Apple Silicon: Standard install with MPS
3. ✓ bitsandbytes ARM64: Available for NVIDIA ARM64, make optional for Apple Silicon
4. ✓ deepspeed ARM64: Not available, make optional for all ARM64 platforms
5. ✓ triton ARM64: Wheels available for NVIDIA ARM64, not for Apple Silicon (PyTorch handles Metal)
6. ✓ Update pyproject.toml with platform markers for bitsandbytes (done)
7. TODO: Update pyproject.toml with platform markers for deepspeed
8. TODO: Test installation on both platforms (Tasks 568, pol)

---

## Summary

| Dependency | NVIDIA ARM64+CUDA | Apple Silicon+MPS | Notes |
|------------|-------------------|-------------------|-------|
| **PyTorch** | ✓ Available | ✓ Fully Supported | Use cu128 index for NVIDIA ARM64 |
| **torchvision** | ✓ Available | ✓ Supported | Follows PyTorch installation |
| **bitsandbytes** | ✓ Available (v0.49.1+) | ⚠️ Limited (make optional) | CUDA-only features don't work on macOS |
| **deepspeed** | ⚠️ Build from source | ✗ Not supported | Make optional on all ARM64 |
| **triton** | ⚠️ Emerging (v3.5.1+) | ✗ Not supported | PyTorch Metal codegen used on macOS |

**Legend:**
- ✓ = Confirmed available and functional
- ⚠️ = Available but with limitations (see notes)
- ✗ = Known to be unavailable
- ? = Research required
