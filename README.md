# PINN HF/SRV/USRV Experiments

This repository collects multiple Python/PyTorch PINN experiment versions for HF/SRV/USRV pressure-field modeling.

## Project Folders

- `pinn_hf_srv_usrv`: original project version.
- `pinn_hf_srv_usrv_v2`: independent v2 rewrite.
- `pinn_hf_srv_usrv_v3_partition_mlp`: partitioned MLP version.
- `pinn_hf_srv_usrv_v4_hard_dirichlet`: v3-based version with production-side hard Dirichlet constraint.
- `pinn_hf_srv_usrv_v5_base_correction`: v3-based base-correction residual-pressure version.
- `pinn_hf_srv_usrv_v6_mixed_flux`: mixed pressure-flux form version.
- `pinn_hf_srv_usrv_v7_v5based_single_mlp`: v5-based single shared MLP version.
- `mph_extract_simple`: auxiliary extraction project.

Training outputs, virtual environments, checkpoints, caches, and generated figures are intentionally ignored by Git.
