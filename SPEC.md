ARM64 Port Specification
========================

## Objective
Port H2O LLMStudio to ARM64 architecture supporting dual GPU platforms:
- NVIDIA ARM64 + CUDA (Grace Hopper, ARM servers)
- Apple Silicon + Metal (M1/M2/M3)

## Scope & Constraints
- **Feature Scope**: Core functionality only (training/inference). Incremental feature additions post-MVP.
- **Success Criteria**: Buildable, runnable code. Performance optimization deferred.
- **Compatibility**: Unified codebase with runtime platform detection. No x86/ARM64 model checkpoint compatibility required.
- **Testing**: Physical ARM64 hardware available. User approval required before machine modifications.
- **Distribution**: Docker containers + native pip/pyproject.toml installation.
- **CI/CD**: Manual for MVP. Automation Phase 2.
- **Timeline**: Opportunistic. No hard deadline.
- **Risk Mgmt**: Separate branch until validated. Minimal documentation (README updates only).

## Technical Requirements

### Python Version
- **Upgrade from 3.10 → 3.12**
- Rationale: Superior ARM64 ecosystem support, more pre-built wheels, performance gains
- Update `requires-python = "==3.12.*"` in pyproject.toml

### PyTorch Strategy
- **Dual GPU Backend Support**:
  - NVIDIA ARM: PyTorch with ARM64 CUDA (determine CUDA version based on PyTorch ARM64 requirements)
  - Apple Silicon: PyTorch with Metal Performance Shaders (MPS)
- Runtime detection via platform checks
- Update torch source index in pyproject.toml for ARM64 wheels

### Critical Dependencies - ARM64 Compatibility

**High-Risk Compiled Extensions** (require ARM64-native builds):
1. `bitsandbytes==0.47.0` - Quantization library
2. `deepspeed==0.17.5` - Distributed training framework
3. `triton==3.3.1` - GPU kernel compiler (via torch)
4. `flash-attn==2.8.3` (optional) - Attention optimization
5. CUDA libraries: nvidia-cublas, nvidia-cudnn, nvidia-nccl, etc. (15+ packages)

**Strategy**:
- Source ARM64 pre-built wheels from HuggingFace, PyTorch ARM repos, community builds
- For NVIDIA ARM: Replace x86 CUDA libs with ARM64 CUDA toolkit equivalents
- For Apple Silicon: Remove CUDA deps, leverage MPS backend
- Document missing/unavailable packages; make optional or remove features

**Lower-Risk Dependencies** (likely ARM64-compatible):
- Core ML: transformers, accelerate, peft, datasets, tokenizers - Check HuggingFace ARM64 support
- Data: numpy, pandas, pyarrow, fastparquet - Mature ARM64 wheels exist
- Compiled C extensions: cffi, cryptography, lxml, psutil - Verify ARM64 wheels availability

### Platform Detection Architecture
- Detect architecture at runtime: `platform.machine()` → 'arm64' or 'aarch64'
- Detect GPU backend:
  - macOS ARM64 → MPS (Metal)
  - Linux ARM64 + NVIDIA GPU → CUDA
  - Fallback → CPU-only
- GPU initialization code must handle backend-specific imports
- Unified codebase - no separate ARM64 code paths unless absolutely necessary

### Build System
- **Primary**: pip + pyproject.toml (current approach maintained)
- Add platform-specific dependency groups:
  - `[dependency-groups.arm64-cuda]` for NVIDIA ARM
  - `[dependency-groups.arm64-metal]` for Apple Silicon
- Update `[tool.uv.sources]` and `[[tool.uv.index]]` for ARM64 PyTorch wheels

## Implementation Checklist

### Phase 1: Dependency Resolution
- [ ] Audit all 140+ dependencies for ARM64 wheel availability
- [ ] Identify ARM64-compatible PyTorch build (CUDA + MPS variants)
- [ ] Locate ARM64 builds: bitsandbytes, deepspeed, triton
- [ ] Test-install critical packages on target ARM64 hardware
- [ ] Document unavailable packages → feature impact analysis

### Phase 2: Python & PyTorch Upgrade
- [ ] Upgrade Python 3.10 → 3.12 in pyproject.toml
- [ ] Update torch source URLs for ARM64 index
- [ ] Add platform detection logic for PyTorch backend selection
- [ ] Test PyTorch installation on both NVIDIA ARM64 + Apple Silicon

### Phase 3: Code Modifications
- [ ] Implement GPU backend detection (CUDA vs MPS vs CPU)
- [ ] Update CUDA-specific code paths for ARM64 CUDA toolkit differences
- [ ] Add MPS backend support where GPU operations exist
- [ ] Handle missing dependencies gracefully (optional features)
- [ ] Update Makefile: `make setup-dev`, `make wave`, `make test`

### Phase 4: Validation
- [ ] `make setup-dev` succeeds on ARM64 (both platforms)
- [ ] `make wave` launches application UI
- [ ] `make test-unit` passes core unit tests
- [ ] `make test` integration tests (may skip GPU-intensive tests initially)
- [ ] Basic training run: small model (<7B) trains for 10 steps
- [ ] Basic inference: model generates text output

### Phase 5: Distribution
- [ ] Create ARM64 Dockerfile (multi-stage build)
- [ ] Test native pip installation on clean ARM64 systems
- [ ] Update README with ARM64 installation instructions
- [ ] Document known limitations, platform-specific quirks

## Key Risks & Unknowns
1. **bitsandbytes ARM64 availability** - Critical for quantization. May require building from source or fork.
2. **deepspeed ARM64 support** - Distributed training may be unavailable initially. Acceptable for MVP.
3. **CUDA toolkit ARM64 differences** - NVIDIA ARM CUDA may have API/behavior differences from x86 CUDA.
4. **MPS backend limitations** - Apple Metal may not support all PyTorch CUDA operations.
5. **triton kernel compilation** - May fail on ARM64. Could block flash-attention and other optimizations.

## Out of Scope
- Performance benchmarking/optimization
- CI/CD automation
- Comprehensive documentation
- Backward compatibility with x86 model checkpoints
- Supporting additional ARM64 GPU vendors (AMD, Qualcomm, etc.)
- Multi-architecture Docker manifests

