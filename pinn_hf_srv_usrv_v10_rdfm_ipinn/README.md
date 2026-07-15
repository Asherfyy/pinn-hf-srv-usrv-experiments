# pinn_hf_srv_usrv_v10_rdfm_ipinn

v10 is an independent RDFM/I-PINN implementation for the existing HF/SRV/USRV
pressure-diffusion setup. It follows the method in *Improved physics-informed
neural networks for the reinterpreted discrete fracture model*: a neural
network predicts nodal pressure values from normalized 2D coordinates, while the loss is built from the
finite-element residual of a reinterpreted discrete fracture model.

## Method

- Matrix cells cover the full reservoir rectangle and are classified as SRV or
  USRV by cell center.
- The original thin HF rectangles are converted to RDFM centerline fractures.
  Their aperture is the short side of each rectangle.
- The residual uses Q1 rectangular FEM matrices:
  - matrix mass: `int_Omega Fai N_i N_j dx`
  - matrix stiffness: `int_Omega D grad N_i . grad N_j dx`
  - fracture mass: `sum int_l epsilon Fai_HF N_i N_j ds`
  - fracture stiffness: `sum int_l epsilon D_HF d_tau N_i d_tau N_j ds`
- Each time step uses implicit Euler:
  `r = M (u_next - u_prev) / dt + A u_next`.
- Production Dirichlet values are imposed directly on nodal predictions before
  residual evaluation.

## Run

```powershell
python main.py test
python main.py train --epochs-per-step 2
python main.py solve
python main.py evaluate
python main.py plot
python main.py mesh
```

The default full training budget is controlled by:

```yaml
training:
  epochs_per_step: 200
```

`train` keeps the I-PINN workflow from the paper. `solve` uses the same
assembled RDFM/FEM matrices and solves the implicit-Euler linear systems with
PCG. This is useful as a reference when the neural-network residual training
does not reproduce the high-conductivity fracture response. By default, `solve`
backs up an existing `outputs/snapshots.npz` and writes the FEM reference
snapshots to `outputs/snapshots.npz`, so the normal `evaluate` and `plot`
commands can be reused.

Outputs are written under `outputs/`:

- `outputs/snapshots.npz`
- `outputs/checkpoints/final.pt`
- `outputs/logs/loss_history.csv`
- `outputs/tables/diagnostics.csv`
- `outputs/tables/fem_solver_history.csv`
- `outputs/figures/field_*.png`
- `outputs/figures/profile_*.png`
- `outputs/figures/mesh_overview.png`
- `outputs/figures/mesh_srv_zoom.png`

## Notes

- v10 is intentionally grid-dependent because the selected route is the
  faithful I-PINNs/FEM residual route.
- It does not import COMSOL meshes. The structured FEM mesh is generated from
  `mesh.nx` and `mesh.ny`.
- The first version solves the current pressure-diffusion problem only. It does
  not implement contaminant transport or bound-preserving concentration.
