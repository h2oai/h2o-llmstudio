# ARM64 Port Implementation Plan

**Generated:** 2026-01-21
**Status:** Ready for Execution
**Repository:** h2oai/h2o-llmstudio
**Branch:** arm64-port (to be created)

## Plan Overview

This plan transforms the ARM64 port specification into actionable epics and tasks tracked in the beads system. The plan follows a sequential epic structure:

1. **Epic 0**: Research & branch setup (foundation)
2. **Epic 1**: NVIDIA ARM64 + CUDA implementation
3. **Epic 2**: Apple Silicon + Metal (MPS) implementation
4. **Epic 3**: Cross-platform integration & validation

## Key Principles

- **Atomic Tasks**: Each task is a commitable unit with tests or validation
- **Sequential Epics**: Complete Epic 0 → Epic 1 → Epic 2 → Epic 3
- **Branch Safety**: All work in `arm64-port` branch, user approval for hardware changes
- **Test Coverage**: Mock-based unit tests + hardware-gated integration tests
- **Priority Levels**: P0 (critical path), P1 (important), P2 (nice-to-have)

## Epic Breakdown

### Epic 0: ARM64 Dependency Research & Branch Setup
**ID:** h2o-llmstudio-lsb
**Priority:** P0
**Status:** Ready to Start
**Deliverable:** docs/arm64_dependencies.md + arm64-port branch

#### Tasks:
1. **h2o-llmstudio-1vh** [P0] - Create arm64-port branch from main
   - No dependencies, can start immediately
   - Validation: `git branch --show-current` returns 'arm64-port'

2. **h2o-llmstudio-2ck** [P0] - Research PyTorch ARM64 builds (CUDA + MPS)
   - Can run parallel with branch creation
   - Output: Index URLs for pyproject.toml [[tool.uv.index]]
   - Files: docs/arm64_dependencies.md

3. **h2o-llmstudio-8tf** [P0] - Research bitsandbytes ARM64 availability
   - Depends on: Task 1vh (branch creation)
   - Includes fallback strategy if unavailable

4. **h2o-llmstudio-7wg** [P0] - Research deepspeed ARM64 availability
   - Depends on: Task 1vh
   - Includes 'make optional' fallback

5. **h2o-llmstudio-9mt** [P0] - Research triton ARM64 availability
   - Depends on: Task 1vh
   - Note: May be bundled with PyTorch

6. **h2o-llmstudio-aoi** [P0] - Validate make setup-dev on x86 (baseline)
   - Depends on: Task 1vh
   - Establishes baseline before ARM64 changes

---

### Epic 1: NVIDIA ARM64 + CUDA Support
**ID:** h2o-llmstudio-18q
**Priority:** P0
**Status:** Blocked (depends on Epic 0)
**Deliverable:** Working NVIDIA ARM64 build, PyTorch CUDA tensors functional

#### Tasks:
1. **h2o-llmstudio-e05** [P0] - Add platform detection to gpu_utils.py
   - No dependencies within epic (can start after Epic 0)
   - Files: llm_studio/src/utils/gpu_utils.py
   - Functions: detect_platform(), detect_gpu_backend()
   - Returns: ('arm64'|'x86_64', 'cuda'|'mps'|'cpu')

2. **h2o-llmstudio-3vs** [P0] - Write tests for platform detection (mocked)
   - Depends on: Task e05
   - Files: tests/src/utils/test_gpu_utils.py (create)
   - Test: `pytest tests/src/utils/test_gpu_utils.py`

3. **h2o-llmstudio-de9** [P0] - Update pyproject.toml with PyTorch ARM64 CUDA index
   - Depends on: Task 2ck (PyTorch research)
   - Files: pyproject.toml [[tool.uv.index]]
   - Validation: `uv lock --dry-run` succeeds

4. **h2o-llmstudio-5za** [P1] - Update CUDA library deps for ARM64
   - Review nvidia-cuda-*, nvidia-cublas-*, nvidia-cudnn-* (requirements.txt lines 308-344)
   - May need platform-specific dependency groups
   - Files: pyproject.toml dependencies

5. **h2o-llmstudio-568** [P0] - Test make setup-dev on NVIDIA ARM64 hardware
   - Depends on: Tasks de9, 5za
   - **USER APPROVAL REQUIRED** before running
   - Test: Core packages install (torch, transformers, accelerate)

6. **h2o-llmstudio-9n0** [P0] - Create validation script: PyTorch CUDA tensor test
   - Depends on: Task 568
   - Files: scripts/validate_arm64_cuda.py (create)
   - Test: Prints "CUDA available: True" on NVIDIA ARM64

7. **h2o-llmstudio-lg5** [P1] - Update gpu_utils.py OOM error detection for ARM64 CUDA
   - Files: llm_studio/src/utils/gpu_utils.py (lines 45-51)
   - Verify is_cuda_out_of_memory() matches ARM64 CUDA errors

8. **h2o-llmstudio-6gj** [P0] - Test make wave on NVIDIA ARM64
   - Depends on: Task 9n0
   - **USER APPROVAL REQUIRED**
   - Test: UI launches on localhost:10101, GPU detected

9. **h2o-llmstudio-pij** [P1] - Create Dockerfile.arm64-cuda for NVIDIA ARM64
   - Files: Dockerfile.arm64-cuda (create)
   - Base: nvidia/cuda:12.x-base-ubuntu22.04 (ARM64)

---

### Epic 2: Apple Silicon + Metal (MPS) Support
**ID:** h2o-llmstudio-am5
**Priority:** P0
**Status:** Partially Ready (depends on Epic 0, some tasks ready)
**Deliverable:** Working Apple Silicon build, PyTorch MPS functional

#### Tasks:
1. **h2o-llmstudio-qca** [P0] - Add MPS backend detection to gpu_utils.py
   - Depends on: Task e05 (platform detection)
   - Extend detect_gpu_backend() for torch.backends.mps

2. **h2o-llmstudio-cot** [P0] - Update pyproject.toml with PyTorch MPS index
   - Depends on: Task 2ck (PyTorch research)
   - Files: pyproject.toml [[tool.uv.index]]

3. **h2o-llmstudio-oy3** [P0] - Make CUDA dependencies optional for macOS ARM64
   - No Epic 0 dependency (can start anytime)
   - Wrap nvidia-* deps with platform markers: `sys_platform != 'darwin' or platform_machine != 'arm64'`
   - Files: pyproject.toml

4. **h2o-llmstudio-2qx** [P1] - Handle bitsandbytes for Apple Silicon
   - Depends on: Task 8tf (bitsandbytes research)
   - Strategy: Use ARM64 build if available, make optional if not

5. **h2o-llmstudio-pol** [P0] - Test make setup-dev on Apple Silicon
   - Depends on: Tasks cot, oy3
   - **USER APPROVAL REQUIRED**
   - Test: PyTorch with MPS support installs

6. **h2o-llmstudio-mug** [P0] - Create validation script: PyTorch MPS tensor test
   - Depends on: Task pol
   - Files: scripts/validate_arm64_mps.py (create)
   - Test: Prints "MPS available: True" on Apple Silicon

7. **h2o-llmstudio-9lh** [P1] - Update gpu_utils.py for MPS error handling
   - Add is_mps_out_of_memory() function
   - Files: llm_studio/src/utils/gpu_utils.py

8. **h2o-llmstudio-75i** [P0] - Test make wave on Apple Silicon
   - Depends on: Task mug
   - **USER APPROVAL REQUIRED**
   - Test: Device detection shows 'mps'

9. **h2o-llmstudio-czd** [P2] - Create Dockerfile.arm64-mps for Apple Silicon
   - Files: Dockerfile.arm64-mps (create)
   - Note: Docker on macOS has limitations

---

### Epic 3: Cross-Platform Integration & Validation
**ID:** h2o-llmstudio-dfe
**Priority:** P1
**Status:** Blocked (depends on Epics 1 and 2)
**Deliverable:** Unified codebase, tests passing, documentation complete

#### Tasks:
1. **h2o-llmstudio-1uw** [P0] - Ensure gpu_utils.py handles all backends uniformly
   - No Epic 1/2 dependency (can start anytime)
   - Review sync_across_processes(), OOM detection for CUDA/MPS/CPU
   - Files: llm_studio/src/utils/gpu_utils.py

2. **h2o-llmstudio-1wg** [P1] - Update Makefile for ARM64 compatibility
   - Make nvidia-smi optional (Makefile line 161)
   - Files: Makefile
   - Test: `make llmstudio` works on Apple Silicon without nvidia-smi

3. **h2o-llmstudio-e5a** [P0] - Add platform-specific test markers
   - No dependency (can start anytime)
   - Add pytest markers: @pytest.mark.arm64_cuda, @pytest.mark.arm64_mps
   - Files: tests/conftest.py or pytest.ini

4. **h2o-llmstudio-0fi** [P0] - Run make test-unit on NVIDIA ARM64
   - Depends on: Tasks 1uw, e5a
   - **USER APPROVAL REQUIRED**
   - Document failures, fix ARM64-specific issues

5. **h2o-llmstudio-8vj** [P0] - Run make test-unit on Apple Silicon
   - Depends on: Tasks 1uw, e5a
   - **USER APPROVAL REQUIRED**
   - Document failures, fix MPS-specific issues

6. **h2o-llmstudio-d7j** [P0] - Run smoke test: minimal model inference on NVIDIA ARM64
   - Depends on: Task 0fi
   - Load GPT-2, run 10 steps, verify GPU utilization
   - **USER APPROVAL REQUIRED**

7. **h2o-llmstudio-21m** [P0] - Run smoke test: minimal model inference on Apple Silicon
   - Depends on: Task 8vj
   - Load GPT-2, run 10 steps on MPS
   - **USER APPROVAL REQUIRED**

8. **h2o-llmstudio-7ad** [P0] - Update README.md with ARM64 installation instructions
   - No dependency (can start anytime)
   - Add "ARM64 Support" section
   - Files: README.md

9. **h2o-llmstudio-nl3** [P1] - Update CLAUDE.md with ARM64 build/test notes
   - Files: CLAUDE.md
   - Document: platform detection, test markers, known limitations

10. **h2o-llmstudio-18y** [P1] - Create summary report: ARM64 port status
    - Files: docs/arm64_port_summary.md (create)
    - Document: completed features, tested platforms, known limitations

---

## Dependency Graph Summary

```
Epic 0 (Research & Setup)
  └─ Epic 1 (NVIDIA ARM64)
  └─ Epic 2 (Apple Silicon)
      └─ Epic 3 (Integration)
```

### Critical Path (P0 tasks blocking epic completion):
1. Epic 0: 1vh → 2ck → [research tasks] → aoi
2. Epic 1: e05 → 3vs → de9 → 568 → 9n0 → 6gj
3. Epic 2: qca → cot → pol → mug → 75i
4. Epic 3: e5a + 1uw → 0fi + 8vj → d7j + 21m

### Parallel Opportunities:
- Epic 0: Tasks 1vh and 2ck can run in parallel
- Epic 1: Tasks e05 (platform detection) and de9 (pyproject.toml) independent
- Epic 2: Task oy3 (CUDA exclusion) can start anytime
- Epic 3: Tasks 1uw, e5a, 7ad can start before Epics 1/2 complete

---

## File Modification Summary

### Core Code Changes:
- **llm_studio/src/utils/gpu_utils.py**: Platform detection, backend routing, MPS OOM handling
- **pyproject.toml**: PyTorch index URLs, CUDA dependency platform markers
- **Makefile**: nvidia-smi optional for non-NVIDIA platforms

### Test Files:
- **tests/src/utils/test_gpu_utils.py** (create): Platform detection unit tests
- **tests/conftest.py or pytest.ini**: Platform-specific pytest markers

### Scripts:
- **scripts/validate_arm64_cuda.py** (create): NVIDIA ARM64 validation
- **scripts/validate_arm64_mps.py** (create): Apple Silicon validation

### Documentation:
- **docs/arm64_dependencies.md** (create): Dependency research findings
- **docs/arm64_port_summary.md** (create): Final status report
- **README.md**: ARM64 installation instructions
- **CLAUDE.md**: ARM64 build/test notes

### Docker:
- **Dockerfile.arm64-cuda** (create): NVIDIA ARM64 container
- **Dockerfile.arm64-mps** (create): Apple Silicon container

---

## Getting Started

### View Ready Tasks:
```bash
bd ready
```

### Start Epic 0:
```bash
# Task 1: Create branch
bd update h2o-llmstudio-1vh --status=in_progress
git checkout -b arm64-port
bd close h2o-llmstudio-1vh

# Task 2: Research PyTorch (can run parallel)
bd update h2o-llmstudio-2ck --status=in_progress
# ... perform research ...
bd close h2o-llmstudio-2ck
```

### Check Project Status:
```bash
bd stats
bd blocked  # Show blocked issues
bd list --status=in_progress  # Show active work
```

### Sync Progress:
```bash
bd sync  # Commit beads changes to git
```

---

## Known Constraints

1. **User Approval Required**: Tasks involving hardware changes require explicit approval
2. **Python 3.10**: Staying on Python 3.10 for now (3.12 upgrade deferred)
3. **Sequential Epics**: Epic 1 must complete before Epic 2 work begins (per user preference)
4. **Opportunistic Timeline**: No hard deadlines, can pause at any epic boundary
5. **Branch Safety**: All work in arm64-port branch, no main branch modifications

---

## Success Criteria

### Epic 0 Complete:
- ✓ docs/arm64_dependencies.md exists with PyTorch, bitsandbytes, deepspeed, triton findings
- ✓ arm64-port branch created
- ✓ Baseline x86 `make setup-dev` validated

### Epic 1 Complete:
- ✓ Platform detection code in gpu_utils.py with tests
- ✓ `make setup-dev` succeeds on NVIDIA ARM64
- ✓ PyTorch CUDA tensors work on NVIDIA ARM64
- ✓ `make wave` launches on NVIDIA ARM64

### Epic 2 Complete:
- ✓ `make setup-dev` succeeds on Apple Silicon
- ✓ PyTorch MPS tensors work on Apple Silicon
- ✓ `make wave` launches on Apple Silicon
- ✓ CUDA dependencies excluded on macOS ARM64

### Epic 3 Complete:
- ✓ `make test-unit` passes on both platforms
- ✓ Smoke tests (GPT-2 inference) pass on both platforms
- ✓ README.md has ARM64 instructions
- ✓ docs/arm64_port_summary.md documents status

---

## Next Steps

1. Run `bd ready` to see available tasks
2. Start with Task h2o-llmstudio-1vh (create arm64-port branch)
3. Work through Epic 0 sequentially
4. Sync progress with `bd sync` regularly
5. Request user approval before hardware-modifying tasks

For questions or clarifications, refer to:
- Specification: SPEC.md
- User constraints: CLAUDE.md
- Dependency details: `bd show <task-id>`
