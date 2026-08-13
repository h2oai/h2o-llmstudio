## Problem

LLM Studio shows a blank screen on some deployments — the app dies during import, so nothing ever listens on the port:

```
File ".../deepspeed/ops/transformer/inference/triton/matmul_ext.py", line 37, in is_nfs_path
  fs_type = lines[1].split()[1].lower()  # File system type is the second column
IndexError: list index out of range
```

## Root cause — two separate defects

**1. A training dependency is imported in the app's startup path.**
`modeling_utils.py` imports `deepspeed` at module scope, and that file is in the UI import chain (`app.py` → `chat_update.py` → `text_causal_language_modeling_model.py` → `modeling_utils.py`). So an import-time failure inside DeepSpeed takes down the entire app — including for users who never enable DeepSpeed. Every use of those symbols in the file is already behind `if cfg.environment.use_deepspeed`, so the module-scope import buys nothing.

**2. DeepSpeed misparses `df` output.** `is_nfs_path()` runs `df -T <path>` and does:

```python
lines = output.strip().split('\n')
if len(lines) > 1:                          # lines[1] IS guarded
    fs_type = lines[1].split()[1].lower()   # .split()[1] is NOT
```

BusyBox `df` wraps a device name longer than ~20 chars onto its own line, so `lines[1]` holds only the device, `.split()` returns one element, and `[1]` raises. The image ships BusyBox, not coreutils.

On AWS the EBS device name is derived from the volume ID — `vol-05d356c8837b6754f` → `/dev/dvol05d356c8837b6754f` (25 chars) — so it is always long for EBS-backed volumes. Whether it wraps depends on the node, which is why this looks intermittent:

| Node | `df /mount` device | Result |
|---|---|---|
| `g5.8xlarge` | `/dev/nvme3n1` (12 ch) | one line → parses → healthy |
| `g5.48xlarge` | `/dev/dvol05d356c8837b6754f` (25 ch) | wraps → **IndexError** |

A fresh instance often boots fine and only crashes on a later restart, which is how it gets past QA.

## This PR

**Commit 1 — the real fix:** move the three `deepspeed` imports in `modeling_utils.py` into the `use_deepspeed` branches that already guard their use. An optional training dependency can no longer break app startup. `train.py` keeps its module-scope import — that's the training entrypoint where DeepSpeed belongs.

**Commit 2 — interim, and explicitly a workaround:** point `TRITON_CACHE_DIR` at `/tmp`. Commit 1 alone stops the *app* from dying, but a user who actually enables DeepSpeed would still hit the same `IndexError`, just later during training. This avoids that until the parse is fixed upstream.

I'd rather not keep commit 2 long-term — it writes to the container layer, and relocating the cache is dodging the bug rather than fixing it. It should be reverted once DeepSpeed handles the wrapped-device case (`len(fields) < 2`, or `df -P` so it never wraps). Happy to drop it from this PR if you'd prefer to ship only commit 1 and accept that DeepSpeed users stay broken until then.

## Validation

Patched a crashlooping instance live — same node, same image, same volume:

- **before:** `CrashLoopBackOff`, 7 restarts, 0/1 available
- **after:** `1/1 Running`, 0 restarts, 0 `IndexError`, Wave reaches `listen`
- survived a forced pod delete, ruling out a one-off clean boot

## Affected versions

All releases from **v1.14.4** (where the cache moved to `/mount` in #935) through **v1.14.15**. v1.14.3 and earlier are unaffected. Rolling back isn't viable — FedRAMP needs 1.14.15 for CVEs.

Upgrading DeepSpeed doesn't help; the parse is unchanged through 0.19.5.
