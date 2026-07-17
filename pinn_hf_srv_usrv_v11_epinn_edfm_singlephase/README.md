# pinn_hf_srv_usrv_v11_epinn_edfm_singlephase

v11 is an independent E-PINN/EDFM single-phase flow project for the existing
HF/SRV/USRV geometry. The current version solves two pressure components,
`P12` and `P13`, in one network pass while keeping the single-phase pressure
physics. It borrows the 2024 two-phase-flow paper's network idea: the neural
network maps the previous full-field pressure vector to the next full-field
pressure vector, while the physics loss is built from a finite-volume residual.

This is not a two-phase simulator: there is no saturation equation, IMPES
splitting, capillary pressure, or relative permeability. `P12` and `P13` are
two pressure-like components whose sum is reported as `Ptotal`.

## Method

- Matrix cells use a cell-centered Cartesian FVM grid.
- The original thin HF rectangles are converted to EDFM centerline fracture
  segments and added as extra cells.
- Connections include matrix-matrix `T_mm`, matrix-fracture `T_mf`, and
  fracture-fracture `T_ff` transmissibilities.
- The E-PINN input is normalized pressure `[P12^t, P13^t]` over all cells, and
  its output is normalized pressure `[P12*, P13*]` for the next time step. The
  tensor shape is `[num_cells, 2]`.
- The network uses sparse adjacency-location message passing, BatchNorm,
  adaptive ReLU, skip connection, gated updating, and a per-cell update bias
  for faster pressure-vector convergence on a fixed EDFM graph.
- v11 now uses only the legacy diffusion coefficient scale from the earlier
  strong-form projects: `Fai/D1/D2`. The discrete residual is applied
  independently to each pressure component:

```text
R_i,c = Fai_i * V_i * (p_i,c* - p_i,c^t) / dt_day
        - sum_j T_ij,c * (p_j,c* - p_i,c*)
loss = mean((R_i,c / row_scale_i,c)^2)
```

- Connection transmissibility is assembled as
  `T_ij,c = seconds_per_day * D_c,face * A / L`. This keeps the old second-based
  diffusion coefficients consistent with v11's day-based time grid.
- `row_scale_i,c = Fai_i V_i / dt_day + sum_j T_ij,c`, so large time steps and
  weakly connected far-field cells are not hidden by raw residual units.
- The BHP constraint is applied to both the nearest matrix cell and the
  connected fracture endpoint cell. This is important for a producer located at
  the main-fracture tip.
- HF cells can receive a larger residual weight, and HF-HF connections include
  an optional pressure-jump penalty. These training terms help the E-PINN see
  the high-conductivity fracture path instead of averaging it away among the
  matrix cells.
- External no-flow boundaries are natural in the FVM graph: no outside
  connection is added.
- `pressure.C13_C12` controls the default split between the two components. For
  example, total initial pressure `25 MPa` becomes `P12 ~= 24.730 MPa` and
  `P13 ~= 0.270 MPa` when `C13_C12 = 0.010900084`.
- `P12` uses `physics.D1` and `P13` uses `physics.D2`.
- `python main.py solve` runs the same single-phase EDFM/FVM equations with a
  direct implicit linear solve for each pressure component. Use it as the
  reference solution when checking whether a bad plot comes from the
  discretization or from E-PINN undertraining. It writes a numerical reference
  snapshot, not a PINN-trained snapshot.

## Run

```powershell
python main.py test
python main.py train
python main.py train --time-steps 2 --epochs-per-step 2 --grid-nx 8 --grid-ny 6
python main.py solve
python main.py evaluate
python main.py plot
python main.py mesh
```

The default grid uses the sparse E-PINN path:

```yaml
grid:
  nx: 180
  ny: 150
edfm:
  max_dense_elements: 5000
  fracture_tangential_multiplier: 1.0
physics:
  seconds_per_day: 86400.0
  transmissibility_scale: 1.0
  diffusivity_keys: ["D1", "D2"]
  Fai:
    HF: 0.1
    SRV: 0.05
    USRV: 0.05
  D1:
    HF: 1.0
    SRV: 1.0e-7
    USRV: 1.0e-9
  D2:
    HF: 1.0
    SRV: 9.95e-8
    USRV: 9.9e-10
model:
  architecture: "sparse"
  output_dim: 2
  hidden_dim: 32
  message_passing_steps: 2
pressure:
  components: ["P12", "P13"]
  C13_C12: 0.010900084
  normalization: "component_affine"
time_grid:
  times_days: [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 200.0, 500.0, 750.0, 1000.0]
well:
  constrain_connected_fracture: true
training:
  fracture_residual_weight: 50.0
  fracture_flux_weight: 10.0
  cell_update_lr_multiplier: 10.0
```

Outputs are written under `outputs/`:

- `outputs/snapshots.npz`
- `outputs/checkpoints/final.pt`
- `outputs/logs/loss_history.csv`
- `outputs/tables/diagnostics.csv`
- `outputs/tables/well_history.csv`
- `outputs/figures/field_P12_t*.png`
- `outputs/figures/field_P13_t*.png`
- `outputs/figures/field_Ptotal_t*.png`
- `outputs/figures/mesh_edfm.png`
- `outputs/figures/main_fracture_profile.png`
- `outputs/figures/loss_history.png`

## Notes

- v11 uses single-phase physics only. The two-output mode is a dual pressure
  component solve, not a two-phase flow model.
- Storage always uses `physics.Fai`, and transmissibility always uses the
  component diffusion keys listed in `physics.diffusivity_keys`.
- It does not import COMSOL meshes. Matrix cells and EDFM fracture segments are
  generated in code.
- The E-PINN training path is sparse and uses EDFM/FVM `edge_index` message
  passing instead of dense `N x N` adjacency tensors. `edfm.max_dense_elements`
  only controls whether a small-grid dense adjacency is also built for checks;
  it is not a training size limit.
- The direct `solve` command uses a matrix-free preconditioned conjugate
  gradient solve, so the refined default grid and denser time scale remain
  practical without SciPy.
- `train` and `solve` both write `outputs/snapshots.npz`. Check the `solver`
  field in that file when comparing results: `sparse_epinn_train` means PINN
  training output, and `direct_fvm_edfm` means direct FVM reference output.
- If `train` and `solve` disagree strongly, trust `solve` for the current
  single-phase case first. That means the EDFM/FVM physics is consistent and
  the neural network needs more epochs or stronger training weights.
