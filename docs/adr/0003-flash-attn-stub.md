# Shadow NGC flash_attn with a deep importable stub

The NGC system flash_attn is compiled against torch 2.9 and ABI-crashes when loaded under torch 2.10 (`c10::cuda::c10_cuda_check_implementation` symbol mismatch). It must not be imported.

vllm-fork's `rotary_embedding/common.py` guards the flash_attn import with `find_spec()`:

```python
self.apply_rotary_emb_flash_attn = None
if find_spec("flash_attn") is not None:
    from flash_attn.ops.triton.rotary import apply_rotary
    self.apply_rotary_emb_flash_attn = apply_rotary
```

With `--system-site-packages`, `find_spec("flash_attn")` finds the NGC system package (non-None) and triggers the import chain, which loads the ABI-incompatible `.so`.

We create a stub package in the venv that shadows the NGC system flash_attn:

```
flash_attn/__init__.py          # empty
flash_attn/ops/__init__.py      # empty
flash_attn/ops/triton/__init__.py   # empty
flash_attn/ops/triton/rotary.py    # apply_rotary = None
```

`find_spec()` finds the stub (non-None), the import chain completes without loading any `.so`, and `apply_rotary_emb_flash_attn` is set to `None`.

`forward_cuda` is unaffected: it uses `vllm.vllm_flash_attn` (vllm's bundled attention library, installed because `torch.version.cuda == "12"`) and never consults `apply_rotary_emb_flash_attn`.

`forward_hip` (AMD) uses `apply_rotary_emb_flash_attn` but has a native-PyTorch fallback when it is `None`, which is acceptable.

xformers is not required by vllm-fork at all; its stub uses `raise ImportError()` to block any code that probes for it.

## Considered options

- **Build flash_attn from source** — tried (`a5d2bb5`). NGC 25.10 has CUDA 13.0 toolkit; torch+cu128 reports `torch.version.cuda = "12.8"`; PyTorch's `_check_cuda_version` raises on major-version mismatch (13 ≠ 12). Patching `cpp_extension.py` and using `--no-deps` works but adds ~30 min to the build; rejected in favour of the stub.
- **Install pre-built flash_attn wheel** — a pre-built wheel for the exact (cu128, torch 2.10, Python 3.12, cxx11abi) combination was not confirmed available; skipped.
- **Deep importable stub** — accepted. Zero compilation, correct for the CUDA path, acceptable fallback for AMD.
