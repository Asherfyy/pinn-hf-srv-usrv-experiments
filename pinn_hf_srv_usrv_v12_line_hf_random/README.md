# pinn_hf_srv_usrv_v12_line_hf_random

v12 is based on the v3 partitioned-MLP PINN, but the hydraulic fractures are
represented as lower-dimensional lines instead of 0.01 m thick rectangular PDE
regions.

## Main Changes From v3

- HF rectangles in `config/default.yaml` are source geometry only. They are
  converted to centerline segments at startup.
- The matrix domain is still split into SRV and USRV by rectangular geometry.
  The old HF thin area is now part of SRV unless a point lies exactly on a
  fracture centerline.
- HF PDE collocation points are sampled directly on fracture lines.
- The production Dirichlet point is the main-fracture endpoint `(360, 75)`.
- HF-SRV coupling is no longer a thin-rectangle two-sided interface residual.
  It is now a line-to-SRV pressure coupling: sample a point on the fracture
  line, offset it by `eps_hf_srv` along the normal into SRV, and penalize the
  pressure jump between the HF subnet and SRV subnet.
- SRV-USRV interface pressure/flux losses are retained from v3.
- Training uses random collocation by default:

```yaml
sampler:
  sampling_mode: "random"
  time_sampling_mode: "random"
  time_pairing_mode: "cartesian"
  n_time_pde: 16
training:
  fixed_collocation_points: false
  resample_every: 1
  use_lbfgs: false
```

`time_pairing_mode: "cartesian"` means each sampled spatial set is expanded
against a smaller sampled time set. For example, `n_pde_srv: 64`,
`n_near_hf_srv: 64`, `n_near_srv_usrv: 64`, and `n_time_pde: 16` produce
`(64 + 64 + 32) * 16 = 2560` SRV PDE spacetime points with tensor shape
`[2560, 3]`.

`resample_every: 1` means every epoch uses a fresh collocation batch. If the
cartesian batch is too expensive, reduce the spatial counts or the `n_time_*`
counts before switching back to `time_pairing_mode: "paired"`.

## Model

The model remains a three-subnet PINN:

- `HF` subnet: trained on fracture centerline points.
- `SRV` subnet: trained in the stimulated reservoir matrix region.
- `USRV` subnet: trained in the outer reservoir region.

The public input is still physical coordinate `[x, y, t]`. Internally it is
normalized to `[x_hat, y_hat, t_hat]`. The current default gives each subnet
the legacy 5D local-feature input:

```yaml
model:
  subnet_input_dim: 5
```

In 5D mode the subnet input is `[x_local, y_local, x_hat, y_hat, t_hat]`. For
line HF, the local coordinate across the thin fracture aperture is fixed at
`0.5`, so the HF subnet mainly sees variation along the fracture length. The
pure coordinate mode is still supported by setting `subnet_input_dim: 3`.

The output is unchanged from v3:

- `u12`
- `u13`

These are converted to physical pressure components `P12/P13`, and `Ptotal` is
reported as their sum.

The default output constraint now follows the v5 base-correction idea:

```text
u = u_base(x, y, t) + envelope(t)^p * correction_NN(x, y, t)
envelope(t) = 1 - exp(-decay_rate * t)
```

```yaml
model:
  constraint_mode: "ic_base_correction"
  base_checkpoint: null
  base_time_lag_days: 1.0
  correction_envelope_power: 1.0
```

With `base_checkpoint: null`, `u_base` is the uniform initial-pressure field
`u=1`. If `base_checkpoint` points to a previous v12 checkpoint, that frozen
model is evaluated at `max(t - base_time_lag_days, t_min)` and the active
network learns only the correction relative to that baseline. Set
`constraint_mode: "ic_hard"` to recover the older v12 form
`u = 1 + envelope(t) * NN(x, y, t)`.

## Physics Loss

The strong-form effective diffusion residual is still used:

```text
u_t - kx * u_xx - ky * u_yy = 0
```

For HF line samples, the HF local coordinate keeps the normal coordinate fixed,
so the HF subnet mainly learns variation along the fracture tangent direction.

The total loss includes:

- PDE residuals on HF line, SRV area, and USRV area.
- Dirichlet pressure at the main-fracture production endpoint.
- Outer no-flow Neumann boundary loss.
- Line HF-SRV pressure coupling.
- SRV-USRV pressure and flux continuity.
- Main-fracture fast pressure-link loss.
- Secondary-fracture pressure-link loss.
- Junction coupling near main-secondary intersections.
- Soft pressure range penalty.

## Run

```powershell
python main.py test
python main.py train --no-resume
python main.py train --epochs 1000 --no-resume
python main.py evaluate
python main.py plot
python main.py all --no-resume
```

For a quick smoke test:

```powershell
python main.py train --epochs 1 --no-resume
```

## Outputs

- `outputs/checkpoints/final.pt`
- `outputs/logs/loss_history.csv`
- `outputs/tables/diagnostics.csv`
- `outputs/tables/section_profiles.csv`
- `outputs/figures/field_P12_t*.png`
- `outputs/figures/field_P13_t*.png`
- `outputs/figures/field_Ptotal_t*.png`
- `outputs/figures/section_main_fracture.png`
- `outputs/figures/section_secondary_1.png`
- `outputs/figures/section_cross_region_y75.png`

The field plots fill SRV/USRV as 2D domains and draw HF as line overlays. HF is
not triangulated as a filled 2D region in v12.
