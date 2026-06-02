# ScalarLM

A closed-loop LLM experimentation platform: inference and post-training run in the same deployment, with a shared checkpoint so the model updates without restarting. Runs on NVIDIA, AMD, and CPU hardware from a single codebase.

## Language

### Platform

**ScalarLM**:
The full system — chat UI, inference server, training scheduler, and the scripts that tie them together.
_Avoid_: "the app", "the stack"

**Closed-loop experimentation**:
The practice of querying a running model, constructing training signal from the results, and submitting a post-training job against the same deployment — all without touching infrastructure.

**Checkpoint**:
The model weights artifact written by Megatron-LM and loaded by vLLM. The shared checkpoint is the seam between training and inference.
_Avoid_: "model", "weights file"

### Inference

**Inference server**:
The vLLM instance exposing an OpenAI-compatible HTTP endpoint. Backed by the vllm-fork.
_Avoid_: "the API", "the model server"

**vllm-fork**:
The project's pinned fork of vLLM, maintained at `github.com/supermassive-intelligence/vllm-fork`. Diverges from upstream for custom kernel bindings and model support. Not kept in sync with upstream on a fixed cadence.

**Attention backend**:
The kernel library vLLM uses for attention computation. On NVIDIA: flashinfer (default). On AMD: flash_attn triton kernels or native PyTorch fallback.

### Training

**Training job**:
A Megatron-LM post-training run submitted through the Slurm scheduler embedded in the platform. Writes a new checkpoint on completion.

**Slurm**:
The job scheduler running inside the deployment. Dispatches training jobs across GPUs. Not exposed externally.

### Deployment

**Target device**:
The hardware backend selected at build time: `cuda` (NVIDIA), `amd`, or `cpu`. Controlled by the `VLLM_TARGET_DEVICE` build arg.
_Avoid_: "hardware type", "GPU type"

**NGC base image**:
The NVIDIA GPU Cloud container used as the NVIDIA target's base. Ships pre-compiled torch, CUDA toolkit, cuDNN, NCCL, and HPC-X MPI.
_Avoid_: "nvidia base", "CUDA container"

**System venv**:
The Python virtual environment created with `--system-site-packages`, making NGC system packages (torch, flash_attn, etc.) visible at lower precedence than packages installed into the venv.

---

## Example dialogue

> **Dev A**: "Why does the venv have a fake flash_attn in it?"
>
> **Dev B**: "That's an ABI stub. The NGC base image ships a flash_attn compiled for an older torch. The system venv would find it, and it would crash because it's incompatible with the torch version we install into the venv. The stub shadows it."
>
> **Dev A**: "So the attention backend is broken?"
>
> **Dev B**: "No — the inference server's CUDA path uses vllm's bundled attention library, not the external flash_attn. The stub only affects the AMD rotary embedding path, which falls back to native PyTorch."
