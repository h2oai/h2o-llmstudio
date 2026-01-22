MUST:
- every new change has to be buildable and tested.
- every new change has to be covered by a test.
- if you need to refine any of task, first create a new entry using `bd` command and then start working on the task.
- after each iteration tests MUST be executed to validate the buildability and functionality.


## Build

```
make setup-dev
```

## Run the product

```
make wave
```

## Unit Test

```
make test-unit
```


## Integration Test

```
make test
```

## ARM64 Platform Support

### Platform Detection

The project includes platform and GPU backend detection utilities in `llm_studio/src/utils/gpu_utils.py`:

```python
from llm_studio.src.utils.gpu_utils import detect_platform, detect_gpu_backend

# Detects CPU architecture
platform = detect_platform()  # Returns "arm64" or "x86_64"

# Detects available GPU backend
backend = detect_gpu_backend()  # Returns "cuda", "mps", or "cpu"
```

### Supported ARM64 Configurations

1. **NVIDIA ARM64 + CUDA** (Linux aarch64)
   - Platforms: Grace Hopper GH200, Grace Blackwell GB200, Jetson
   - GPU Backend: CUDA 12.8
   - PyTorch: 2.7.1 with cu128 index
   - Installation: `make setup-dev`

2. **Apple Silicon + Metal** (macOS ARM64)
   - Platforms: M1, M2, M3, M4
   - GPU Backend: MPS (Metal Performance Shaders)
   - PyTorch: 2.7.1 from PyPI
   - Installation: `make setup-dev`

### PyTorch Configuration

The project uses a unified PyTorch configuration that works across platforms:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu128/"
# No explicit=true → enables PyPI fallback for macOS
```

- **Linux (x86_64 & ARM64)**: Installs from cu128 index
- **macOS (ARM64)**: Falls back to PyPI for MPS support

### Platform-Specific Dependencies

Some dependencies are conditionally installed based on platform:

```toml
# Excluded on macOS ARM64 (not compatible)
bitsandbytes==0.47.0 ; sys_platform != 'darwin' or platform_machine != 'arm64'

# Excluded on macOS (CUDA-only)
flash-attn==2.8.3 ; sys_platform != 'darwin'
```

### Known Limitations

**Not available on ARM64:**
- **deepspeed**: No ARM64 support (excluded from dependencies)
- **triton**: Limited ARM64 support (emerging for NVIDIA, not for Apple)

**Not available on macOS ARM64:**
- **bitsandbytes**: Windows and Linux only
- **flash-attn**: CUDA-only, not compatible with MPS

### Platform-Specific Testing

Use pytest markers to run platform-specific tests:

```bash
# Run only ARM64 + CUDA tests
pytest -m arm64_cuda

# Run only ARM64 + MPS tests
pytest -m arm64_mps

# Run only x86_64 tests
pytest -m x86_64

# Skip GPU-requiring tests
pytest -m "not requires_gpu"

# Run only CUDA tests
pytest -m requires_cuda

# Run only MPS tests
pytest -m requires_mps
```

**Available markers** (configured in `pyproject.toml`):
- `arm64_cuda`: Tests for ARM64 + NVIDIA CUDA
- `arm64_mps`: Tests for ARM64 + Apple MPS
- `x86_64`: Tests for x86_64 architecture
- `requires_gpu`: Tests requiring GPU hardware
- `requires_cuda`: Tests requiring NVIDIA CUDA
- `requires_mps`: Tests requiring Apple MPS

### Verification Commands

**Check platform and GPU backend:**
```python
python -c "
from llm_studio.src.utils.gpu_utils import detect_platform, detect_gpu_backend
print(f'Platform: {detect_platform()}')
print(f'GPU Backend: {detect_gpu_backend()}')
"
```

**Check PyTorch CUDA/MPS availability:**
```python
import torch
import platform

print(f'Platform: {platform.machine()}')
print(f'PyTorch version: {torch.__version__}')

if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Metal) available')
else:
    print('CPU-only mode')
```

### Build Notes

**NVIDIA ARM64:**
- Ensure CUDA 12.8 toolkit is installed
- Driver version must support CUDA 12.8
- Use `nvidia-smi` to verify GPU detection (optional in Makefile)

**Apple Silicon:**
- No additional GPU drivers needed (Metal built-in)
- Some operations may fall back to CPU if not MPS-compatible
- Single GPU only (no multi-GPU support)

### Reference Documentation

See `docs/arm64_dependencies.md` for comprehensive dependency research and compatibility notes.

@AGENTS.md
