# pinn_hf_srv_usrv_v13_meshfree_p12_fvm

v13 is a pure-physics mesh-free PINN project for the single pressure component
`P12`. The trainable solver uses only PDE residuals, boundary conditions,
interface continuity, and line-fracture coupling constraints. It does not use
`loss_fvm_teacher`, FVM snapshots, or any FVM pressure values during training.

## Goal

- Solve only `P12`, not `P13` or `Ptotal`.
- Keep the PINN mesh-free: the network input is continuous `[x, y, t]`, and
  losses are evaluated at random collocation points.
- Treat the main fracture and five secondary fractures as one-dimensional line
  segments embedded in the SRV matrix.
- Use different HF/SRV/USRV `Fai`, diffusivity `D`, and permeability values.
- Obtain a reasonable time-dependent pressure field from PDE and boundary
  constraints only. FVM may be used only as an offline comparison, never as a
  training loss.

## PINN Methods Used

The implementation follows the standard PINN idea of minimizing the governing
PDE residual plus initial/boundary residuals. To make the time-dependent
fracture problem trainable without FVM supervision, v13 also includes:

- hard initial-condition/base-correction output, so the network learns a
  pressure correction rather than the whole high-pressure field from scratch;
- hard production-end Dirichlet embedding near the well point, so the BHP
  boundary is satisfied by construction;
- causal time weighting for PDE residuals, which emphasizes early-time
  residual reduction before late-time residuals dominate;
- residual-adaptive collocation, which periodically samples a larger PDE
  candidate pool and keeps high-residual points;
- explicit time-anchor sampling at the evaluation days, so late-time targets
  such as `750 d` and `1000 d` are repeatedly constrained by physics residuals
  instead of being visited only by chance in log-random time sampling;
- SRV collocation enrichment around dead-end fracture tips, so the matrix PDE
  sees the sharp pressure curvature around secondary-fracture endpoints without
  introducing a mesh;
- one-dimensional HF line PDE residuals along fracture tangents, instead of a
  two-dimensional thin-rectangle Laplacian.
- zero tangential-flux constraints at dead-end fracture tips, so branch tips
  and the closed main-fracture end behave as no-flow line endpoints;
- a mesh-free HF leakoff balance term that estimates two-sided SRV normal
  gradients at random line-fracture points and inserts that exchange into the
  one-dimensional fracture balance;
- a geometry symmetry loss `u(x, y, t) = u(x, 150-y, t)` for HF/SRV/USRV
  random point pairs. This uses only the physical symmetry of the current
  reservoir, not a numerical reference field;
- mesh-free local integral conservation on randomly sampled SRV/USRV rectangles:
  the loss integrates `u_t` over each temporary rectangle and balances it
  against first-derivative boundary fluxes on the rectangle sides, without
  assembling FVM cells, EDFM connections, or transmissibilities;
- smaller HF-tip local conservation rectangles are sampled separately from the
  global matrix rectangles, so secondary-fracture endpoint neighborhoods get
  sub-meter-to-few-meter weak conservation probes without creating cells;
- optional HF-line segment conservation and HF-junction flux conservation
  losses are implemented for research runs. They operate on random line
  segments or four-branch junction probes, not on an EDFM/FVM grid. They are
  disabled by default because short continuation runs did not reduce the
  current worst offline MaxAbs;
- an optional matrix-side local fracture expert is implemented as a
  Gaussian-gated subnet around the line fractures. It uses continuous
  nearest-line features and is still mesh-free. It is disabled by default
  because a frozen-base continuation run worsened the current benchmark;
- an erfc-type analytical diffusion base field derived from the pressure
  diffusion equation and BHP boundary condition;
- configurable aggregation across fracture-line drawdown candidates. The
  default `probabilistic_union` mode approximates overlap from multiple
  connected fractures better than a pure smooth maximum, while capping total
  drawdown by the current BHP drawdown so pressure never drops below the
  producer pressure;
- shared SRV/USRV matrix-pressure subnet plus a smooth SRV/USRV analytical-base
  transition, so pressure stays continuous across the matrix-material interface
  while PDE coefficients still use each region's own `D/Fai`;
- a maximum-principle output clamp, enforcing `0 <= P12_hat <= 1` for this
  source-free depletion problem.
- fixed validation-set checkpoint selection with bounded correction
  regularization, so long random-collocation training does not save a visually
  degraded late-epoch model only because one stochastic training batch happened
  to have a lower loss.

## Model

The model is a partitioned MLP with one subnet per region:

- `HF`: one-dimensional fracture-line samples.
- `SRV`: stimulated matrix region.
- `USRV`: outer matrix region.

The public input is always normalized continuous coordinates:

```text
[x_hat, y_hat, t_hat]
```

The output is one normalized pressure:

```text
u = P12_hat
P12 = u * (P_t0 - P_out) + P_out
```

Default output structure:

```text
u_raw = u_base(x, y, t) + envelope(t)^p * correction_NN(x, y, t)
envelope(t) = 1 - exp(-decay_rate * t)
u = hard_dirichlet_blend(u_raw, BHP(t), distance_to_producer)
u = clamp(u, 0, 1)
```

With `base_checkpoint: null`, `u_base = 1`, which is the normalized initial
pressure unless `model.analytic_base.enabled=true`. In the default analytical
base mode, `u_base` is an erfc diffusion approximation that propagates BHP
drawdown along the high-conductivity line fractures and then into the matrix
with the local SRV/USRV diffusivity.

The analytical base uses the diffusion length

```text
L_region = length_multiplier * sqrt((D_region / Fai_region) * t_seconds)
```

and multiplies an along-fracture erfc factor with a normal-to-fracture erfc
factor. Multiple fracture candidates are combined with a differentiable
softmax-weighted smooth maximum:

```yaml
model:
  analytic_base:
    length_multiplier: 2.5
    main_length_scale: 0.03
    secondary_length_scale: 0.013
    secondary_length_scale_gradient: 1.2
    secondary_length_scale_center: 0.5
    aggregation: "probabilistic_union"
    smooth_max_tau: 0.02
    endpoint_taper_enabled: false
    endpoint_taper_m: 1.0
    endpoint_taper_power: 2.0
    matrix_length_mode: "smooth_srv_usrv"
    srv_usrv_blend_width_m: 20.0
```

This is not an FVM solution and is not a teacher target. It is only a
physics-shaped base field; the PINN still trains from residual losses.

The default `smooth_srv_usrv` length blend keeps the base field continuous
across the material interface while still allowing the strong-form residual to
use each region's own `D/Fai`. A more aggressive experimental
`matrix_length_mode: "srv_halo"` is implemented but not enabled by default,
because it can over-spread drawdown into USRV for the current geometry.
`main_length_scale` and `secondary_length_scale` reduce early-time pressure
propagation along line fractures relative to the nominal HF diffusivity,
representing finite fracture storage/connectivity impedance without adding a
mesh or a teacher solution.
`secondary_length_scale_gradient` increases the effective communication length
for farther branch fractures and decreases it for branches close to the
producer, which better matches the asymmetric pressure propagation expected
along a finite-conductivity fracture network.
`endpoint_taper_enabled` is an experimental finite-line correction that reduces
the analytical drawdown outside sealed fracture endpoints. It remains disabled:
it fixes some endpoint-overrun behavior, but a scan of taper lengths increased
the late-time worst offline MaxAbs for the current trained correction.

SRV and USRV share the same matrix pressure subnet by default:

```yaml
model:
  share_srv_usrv_subnet: true
  subnet_input_dim: 3
```

This is intentional. Pressure should be continuous across the SRV/USRV material
interface; the discontinuity belongs in the effective diffusivity and flux
response, not in pressure itself. The old independent-subnet behavior can still
be restored by setting `share_srv_usrv_subnet: false`, but late-time pressure
jumps are then much harder to control.

## Physics Losses

For SRV and USRV, the mesh-free PDE loss is the strong-form diffusion equation:

```text
u_t - kx * u_xx - ky * u_yy = 0
```

For HF, the fracture is lower-dimensional, so the PDE residual is evaluated
only along the line tangent:

```text
u_t - k_tau * u_tau_tau = 0
```

Each region uses its own effective diffusivity:

```text
K_region = D_region / Fai_region
```

Default coefficients:

```yaml
physics:
  Fai:
    HF: 0.1
    SRV: 0.05
    USRV: 0.05
  D:
    HF: 1.0
    SRV: 1.0e-7
    USRV: 1.0e-9
  permeability_mD:
    HF: 30000.0
    SRV: 10.0
    USRV: 1.0
```

The training loss contains:

- HF one-dimensional PDE residual;
- SRV/USRV two-dimensional PDE residuals;
- production-end Dirichlet pressure, also embedded as a hard constraint;
- outer no-flow Neumann boundary;
- line HF-SRV pressure coupling;
- mesh-free HF leakoff balance from two-sided SRV normal gradients;
- SRV-USRV pressure and flux continuity;
- main fracture smoothness and secondary-fracture junction coupling;
- dead-end HF tip no-flow constraints;
- symmetry consistency across the reservoir centerline `y=75 m`;
- mesh-free local conservation on random matrix control rectangles;
- pressure range penalty.

There is intentionally no `loss_fvm_teacher` term and no FVM/EDFM residual
training term. If `loss_weights.fvm_residual` or `fvm_residual.enabled` is
introduced in the config, validation fails deliberately.

The local conservation term is configured by random rectangle counts and size
limits:

```yaml
sampler:
  time_anchor_days: [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 200.0, 500.0, 750.0, 1000.0]
  time_anchor_fraction: 0.85
  n_near_hf_tip_srv: 160
  hf_tip_band_radius_m: 3.0
  n_conservation_srv: 80
  n_conservation_usrv: 80
  n_conservation_hf_tip_srv: 160
  conservation_hf_tip_radius_m: 8.0
  conservation_hf_tip_min_half_size_m: 0.25
  conservation_hf_tip_max_half_size_m: 4.0
  n_hf_tip_neumann: 96
  n_hf_leakoff_balance: 128
  hf_leakoff_offset_m: 0.05
  n_symmetry_hf: 96
  n_symmetry_srv: 128
  n_symmetry_usrv: 96
  conservation_min_half_size_m: 1.0
  conservation_max_half_size_m: 18.0
loss_weights:
  hf_leakoff_balance: 0.2
  symmetry: 5.0
  local_conservation: 1.0
```

For a sampled rectangle `V`, the residual is the normalized balance

```text
Integral_V u_t dA - Integral_boundary(V) (k_x u_x n_x + k_y u_y n_y) ds = 0
```

These rectangles are regenerated by the sampler and are not a mesh. This keeps
the training objective mesh-free while adding a conservative weak-form pressure
constraint.

`n_conservation_hf_tip_srv` adds extra random SRV rectangles around the
dead-end fracture tips. These are still temporary mesh-free control volumes;
they are used because the largest observed local error is concentrated near
secondary-fracture endpoints.
Unlike the global SRV/USRV conservation rectangles, these tip rectangles use a
smaller default half-size range, `0.25-4 m`, compared with `1-18 m` for global
matrix conservation.

Two additional experimental line-network losses exist but are off by default:

```yaml
sampler:
  n_hf_segment_conservation: 0
  n_hf_junction_flux: 0
loss_weights:
  hf_segment_conservation: 0.0
  hf_junction_flux: 0.0
```

`hf_segment_conservation` samples random HF line segments and balances HF
storage, tangential flux difference, and two-sided SRV leakoff over that
segment. `hf_junction_flux` samples four outward branch points around each
main/secondary fracture intersection and enforces a Kirchhoff-style line-flux
balance. Both are mesh-free collocation constraints. They are kept disabled in
the default config because short continuation runs did not reduce the current
worst offline MaxAbs.

The HF leakoff balance is also mesh-free. At each sampled fracture-line point it
uses the HF line residual and SRV normal derivatives at two small offsets:

```text
HF_line_residual - seconds * D_SRV * (du/dn_+ + du/dn_-) / (epsilon * Fai_HF) = 0
```

The offsets are random collocation probes, not EDFM matrix-fracture
transmissibilities. The term is deliberately kept at a modest weight because it
is a local conservation correction on top of the main PDE/interface losses.

The SRV/USRV interface has its own stronger weights:

```yaml
loss_weights:
  interface_pressure_srv_usrv: 120.0
  interface_flux_srv_usrv: 10.0
```

`interface_pressure` and `interface_flux` remain as fallback weights for older
configs and for HF/SRV terms when no split weight is provided.

## Long-Run Stability

Random collocation loss is noisy. A late epoch can look good on its current
random sample but produce a worse pressure map. v13 therefore separates the
logged validation residual from the checkpoint-selection metric:

```yaml
training:
  validation:
    enabled: true
    every: 25
    selection_metric: "diagnostic"
```

`loss_validation` remains the full validation loss. `loss_validation_selection`
is the metric used to save `outputs/checkpoints/best.pt` and finally copy it to
`outputs/checkpoints/final.pt`. By default, the selection metric excludes the
large HF analytical-base PDE offset and focuses on matrix PDE residual,
interface consistency, fracture coupling, local conservation, pressure range,
and correction size.

The correction is also bounded:

```yaml
model:
  hard_dirichlet:
    radius_m: 3.6
  correction_scale: 0.05
  correction_scale_by_region:
    HF: 0.05
    SRV: 0.05
    USRV: 0.05
  correction_activation: "tanh"
loss_weights:
  pde_hf: 0.001
  pde_srv: 1.0
  pde_usrv: 1.0
  hf_tip_neumann: 2.0
  correction_regularization: 20.0
```

This keeps the trained network from destroying the erfc base field during long
runs while still allowing local physics corrections.
`correction_scale_by_region` is available for experiments where one region
needs more correction capacity than the others. A quick HF-only scale scan did
not change the current checkpoint appreciably because the dominant benchmark
error is in SRV/USRV points near the line fractures rather than in the HF line
subnet itself, so the default keeps all regions at `0.05`.
`correction_activation: "softsign"` is available as an experimental
non-saturating alternative for longer correction retraining, but the default
remains `tanh` because the current checkpoint benchmark is better with the
bounded tanh correction.
An optional `local_tip_expert` subnet is also implemented for research runs. It
adds a Gaussian-gated correction around dead-end fracture tips and can be
trained with `training.freeze_base_for_tip_expert: true` so only the local
expert updates. It is disabled by default because the current benchmark did not
improve when this expert was enabled.
An optional `local_fracture_expert` extends the same idea from tips to the
matrix neighborhood of all line fractures. It uses continuous nearest-line
distance, projected fracture coordinate, tangent direction, and time features;
there are no cells or FVM connections. It can be trained alone with
`training.freeze_base_for_local_experts: true`, but the first frozen-base
candidate increased worst MaxAbs, so it remains disabled.
`training.gradient_enhanced_pde` implements a gPINN-style optional loss on the
spatial gradient of the SRV/USRV PDE residual. It is mesh-free and uses only
collocation derivatives, but is disabled by default because the current
benchmark did not improve when this term was weighted in.
The low `pde_hf` weight avoids forcing the analytical finite-impedance fracture
base back into an unrealistically fast one-dimensional HF diffusion solution;
SRV/USRV PDE residuals remain fully weighted.

## SRV/USRV Continuity Diagnostics

`evaluate` reports direct MPa jumps on the left, bottom, and top SRV/USRV
interfaces at `750 d` and `1000 d`, for example:

```text
srv_usrv_jump_left_rms_mpa_t1000
srv_usrv_jump_bottom_rms_mpa_t1000
srv_usrv_jump_top_rms_mpa_t1000
srv_usrv_jump_max_abs_mpa_t1000
```

`plot` also writes `outputs/figures/section_srv_usrv_jump_t1000.png` and
`outputs/tables/srv_usrv_jump_profiles.csv` for direct visual inspection.

## FVM Comparison

`src/fvm_reference.py` and `src/fvm_solve.py` are retained only for offline
comparison experiments. The default config sets:

```yaml
fvm_reference:
  enabled: false
  comparison_only: true
```

Changing this section must not add an FVM term to `loss_weights`; config
validation rejects `loss_weights.fvm_teacher`.

The direct FVM reference is a cell-centered FVM/EDFM solve on the generated
matrix grid and line-fracture segments. For each time interval it solves the
implicit Euler system directly:

```text
Fai_i * V_i * (P_i^{n+1} - P_i^n) / dt
    - sum_j T_ij * (P_j^{n+1} - P_i^{n+1}) = 0
```

The transmissibility uses the same legacy diffusion coefficients as the PINN:

```text
T_mm ~ D_matrix * seconds_per_day * area / distance
T_ff ~ D_HF     * seconds_per_day * aperture / segment_length
T_mf ~ EDFM matrix-fracture connection transmissibility
```

The producer pressure is imposed as a hard BHP value on the well-connected FVM
cells. External no-flow boundaries are natural FVM boundaries: no boundary face
connection is added, so no external flux can enter the residual.

Before a reference field is accepted, `diagnose_fvm_solution` checks:

- grid sizes, connection counts, storage, and transmissibility validity;
- finite pressure values and maximum-principle bounds between BHP and initial
  pressure;
- the free-cell implicit linear residual at every saved time step.

The main diagnostic output is:

```text
outputs/tables/fvm_diagnostics.csv
```

For the default grid, a trustworthy solve should report `fvm_bad_connection_count
= 0`, `fvm_bad_storage_count = 0`, `fvm_nonfinite_pressure_count = 0`, and a very
small `fvm_residual_rel_l2_max`.

Run the standalone FVM reference solve:

```powershell
python -m src.fvm_solve --config config/default.yaml --output-name trusted_fvm_reference_snapshots.npz
```

Run the IDE-friendly PINN/FVM/Error viewer:

```powershell
python plot_pinn_fvm_error.py
```

The viewer opens a Matplotlib window with a time input box and `Plot`, `Prev`,
`Next`, and `Save` buttons. Each plot compares:

- current PINN pressure;
- direct FVM pressure;
- `PINN - FVM` error.

Saved figures are written as:

```text
outputs/figures/pinn_fvm_error_t*.png
```

This comparison is diagnostic only. A large PINN/FVM error means the PINN has
not converged to the conservative FVM reference; it does not mean the training
used FVM targets.

## Run

```powershell
python main.py test
python main.py train --epochs 1 --no-resume
python main.py train --no-resume
python main.py evaluate
python main.py plot
python main.py all --no-resume
```

## Outputs

- `outputs/checkpoints/final.pt`
- `outputs/logs/loss_history.csv`
- `outputs/tables/diagnostics.csv`
- `outputs/tables/fvm_diagnostics.csv`
- `outputs/tables/section_profiles.csv`
- `outputs/figures/field_P12_t*.png`
- `outputs/figures/pinn_fvm_error_t*.png`
- `outputs/figures/section_main_fracture.png`
- `outputs/figures/section_secondary_1.png`
- `outputs/figures/section_cross_region_y75.png`
