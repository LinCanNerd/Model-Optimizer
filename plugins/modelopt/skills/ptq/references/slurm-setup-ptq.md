# SLURM Setup for PTQ

PTQ-specific SLURM details. For generic SLURM patterns (account discovery, job template,
monitoring), see the common skill's `slurm-setup.md`.

---

## 1. Container

Use `nvcr.io/nvidia/pytorch:26.09-py3` for Hugging Face PTQ (AMD64/ARM64).
Keep inference frameworks in the downstream serving environment. Look for a
cached `.sqsh` of this image and verify its provenance before reuse:

```bash
ls *.sqsh ../*.sqsh ~/containers/*.sqsh 2>/dev/null
```

**If a `.sqsh` exists**, use it directly with `--container-image=<path>`. Skip import.

**If no `.sqsh` exists**, import with enroot (caches for subsequent smoke tests and reruns):

```bash
export ENROOT_CACHE_PATH=/path/to/writable/enroot-cache
export ENROOT_DATA_PATH=/path/to/writable/enroot-data
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"
enroot import --output /path/to/container.sqsh docker://nvcr.io#nvidia/pytorch:26.09-py3
```

If enroot import fails (e.g., permission errors on lustre), use pyxis inline pull as fallback — pass `--container-image="nvcr.io/nvidia/pytorch:26.09-py3"`. Note this re-pulls on every job.

### Container dependency pitfalls

Do not assume the base image supplies compatible Python dependencies. Choose a
model-tested Transformers pin within the source ModelOpt's supported range,
not unconstrained latest. For Qwen3.5/3.8 (`qwen3_5`), use `5.14.1`; `5.5.4`
incorrectly classifies VLM language-model weights as missing during export.
Record the image digest and resolved ModelOpt, Torch, Transformers, and Accelerate
versions. Preserve the container's compatible Torch/CUDA stack when resolving deps.

```bash
pip install "transformers==5.14.1"  # Qwen3.5/3.8; select other models' pins separately
```

For unlisted models that need unreleased transformers (e.g., from git), see `references/unsupported-models.md` Step A.

**Prefer `pip install -e ".[hf]" --no-build-isolation`** (run from the Model-Optimizer repo root) to make the synced ModelOpt source importable in the container — this matches how `examples/hf_ptq/slurm/multinode_fsdp2_ptq.slurm` sets up the job, and unlike `PYTHONPATH` it surfaces packaging/build issues instead of masking them. Avoid `pip install -U nvidia-modelopt[hf]` from PyPI, which can upgrade PyTorch and break other packages.

```bash
pip install -e ".[hf]" --no-build-isolation
```

If you specifically need to leave the container's installed packages untouched (e.g. to sidestep a dependency conflict), fall back to `PYTHONPATH` — but note it skips the editable install, so a missing compiled extension only surfaces at import time:

```bash
export PYTHONPATH=/path/to/Model-Optimizer:$PYTHONPATH
```

**Watch for pip dependency conflicts** — NGC containers set `PIP_CONSTRAINT` to pin versions, causing `ResolutionImpossible` errors. Unset it first so pip can resolve freely:

```bash
unset PIP_CONSTRAINT
pip install "transformers==5.14.1"  # Qwen3.5/3.8 example; recheck dependency compatibility
```

If that still conflicts, fall back to `--no-deps` (skips new deps — may need to add missing ones manually):

```bash
pip install "transformers==5.14.1" --no-deps  # Qwen3.5/3.8 example; then check imports/deps
```

---

## 2. GPU Sizing

Estimate GPU count from model size and available GPU memory. `hf_ptq.py` uses `device_map="auto"` so it fills GPUs automatically — request only as many as needed.

For multi-node PTQ (200B+ params), use `hf_ptq.py --use_fsdp2`. For the launch commands (`sbatch`
and manual `torchrun`) and the `--recipe` format, see the *Multi-Node Post-Training Quantization with
FSDP2* section of `examples/hf_ptq/README.md`.

Sizing guidance specific to this path: when the per-rank decoder shard approaches GPU capacity (200B+ at low rank count), either add more nodes (more ranks → smaller shard per rank) or add `--cpu_offload`. Layer detection is automatic; no YAML config needed.

Use the multi-node template from the common skill's `slurm-setup.md` section 4 as the job script wrapper.

---

## 3. Smoke Test

Before the full calibration run, submit a smoke test with `--calib_size 4` and `--time=00:30:00`.
This catches script errors cheaply before using GPU quota on a real run.

See the common skill's `slurm-setup.md` section 2 for the smoke test partition pattern.

Only submit the full calibration job after the smoke test exits cleanly.
