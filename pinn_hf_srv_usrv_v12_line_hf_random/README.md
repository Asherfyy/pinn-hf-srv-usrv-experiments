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
training:
  fixed_collocation_points: false
  resample_every: 1
  use_lbfgs: false
```

`resample_every: 1` means every epoch uses a fresh collocation batch. Adam is
the default optimizer because LBFGS repeatedly reevaluates one closure and is
not a good default for per-epoch random collocation.

## Model

The model remains a three-subnet PINN:

- `HF` subnet: trained on fracture centerline points.
- `SRV` subnet: trained in the stimulated reservoir matrix region.
- `USRV` subnet: trained in the outer reservoir region.

The public input is still physical coordinate `[x, y, t]`. Internally it is
normalized to `[x_hat, y_hat, t_hat]`. By default, each subnet receives exactly
that 3D input:

```yaml
model:
  subnet_input_dim: 3
```

The older 5D local-feature mode `[x_local, y_local, x_hat, y_hat, t_hat]` is
still supported by setting `subnet_input_dim: 5`, but it is no longer the v12
default.

The output is unchanged from v3:

- `u12`
- `u13`

These are converted to physical pressure components `P12/P13`, and `Ptotal` is
reported as their sum.

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
