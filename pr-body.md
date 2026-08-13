## Problem

LLM Studio crashes at startup on some deployments, before serving anything, so the app shows a blank screen:

```
File ".../deepspeed/ops/transformer/inference/triton/matmul_ext.py", line 37, in is_nfs_path
  fs_type = lines[1].split()[1].lower()  # File system type is the second column
IndexError: list index out of range
```

`import deepspeed` is unconditional at module scope in `llm_studio/src/utils/modeling_utils.py:10`, inside the app's import chain, so a failure there kills the whole app rather than just training. It fires regardless of whether DeepSpeed is enabled for any experiment.

## Root cause

`TRITON_CACHE_DIR` points at `/mount`, the persistent volume. DeepSpeed shells out to `df` on that path to check whether it's NFS. **BusyBox `df` wraps a device name longer than ~20 characters onto its own line**, so `lines[1]` holds only the device and `.split()[1]` raises `IndexError`. The image has no coreutils, so BusyBox `df` is what runs.

It depends on the device name, which depends on the node's instance type — which is why it looks intermittent. Verified on two instances of the same version in the same environment:

| Node | `df /mount` device | Result |
|---|---|---|
| `g5.8xlarge` (1 NVMe) | `/dev/nvme3n1` (12 ch) | one line → parses → healthy |
| `g5.48xlarge` (8 NVMe) | `/dev/dvol05d356c8837b6754f` (25 ch) | wraps → **IndexError** |

A fresh instance often boots fine the first time and only crashes on a later restart, which is how it gets past QA.

## Fix

Point the Triton JIT cache at node-local `/tmp` (`overlay`, 7 chars — never wraps).

This doesn't regress #935, which moved persistent data to `/mount` as a path simplification; the Triton cache came along with it rather than being deliberately persisted. Unlike `HF_HOME` (expensive model downloads) and `H2O_WAVE_DATA_DIR` (real state), this is a compile cache that is cheap to rebuild. Upstream DeepSpeed's own default is `~/.triton`, also node-local.

## Validation

Patched a crashlooping instance live (`TRITON_CACHE_DIR=/tmp/.triton/cache`) — same node, same image, same volume:

- **before:** `CrashLoopBackOff`, 7 restarts, 0/1 available
- **after:** `1/1 Running`, 0 restarts, 0 `IndexError`, Wave reaches `listen`
- survived a forced pod delete, ruling out a one-off clean boot

## Affected versions

All releases from **v1.14.4** (where #935 landed) through **v1.14.15**. v1.14.3 and earlier are unaffected. Rolling back is not a viable workaround — FedRAMP needs 1.14.15 for CVEs.

Upgrading DeepSpeed doesn't help: the parse is unchanged through 0.19.5.

## Follow-up (not in this PR)

Worth guarding the top-level `import deepspeed` at `modeling_utils.py:10` so a failure in an optional training dependency can't take down the whole app. Same shape as #967.
