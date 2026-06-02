# Use NGC 25.10 as the NVIDIA base image

NGC 26.01 ships PyTorch 2.10 natively, which is exactly what vllm-fork requires. We tried it first (`e2546e5`) but had to revert (`2b57637`) because NGC 26.01 requires NVIDIA driver ≥ 590.48 and the dev workstation runs driver 570.195.03 (Ubuntu 25.04). NGC 25.10 ships driver-compatible with 570.x and can be used there.

The consequence is that the torch upgrade (2.9 → 2.10) must be done manually inside the venv at build time. See ADR-0002.

## Considered options

- **NGC 26.01** — torch 2.10 natively, no upgrade needed. Rejected: requires driver ≥ 590.48, unavailable on dev machine.
- **NGC 25.10** — torch 2.9, requires manual upgrade. Accepted: compatible with driver 570.x.

## Consequences

The Dockerfile's torch upgrade block includes a version guard (`int(v.split('.')[1]) >= 10`) so that the upgrade step is skipped automatically when the base image already ships torch ≥ 2.10 (e.g. NGC 26.01+ in production). No Dockerfile change needed when the production driver constraint is lifted.
