# pinn_hf_srv_usrv_v3_partition_mlp

这是一个独立的 HF/SRV/USRV 分区 PINN 项目，不覆盖 `pinn_hf_srv_usrv` 或 `pinn_hf_srv_usrv_v2`。

## 核心变化

- 使用三套分区 MLP：`HF`、`SRV`、`USRV` 各自一个子网络。
- 几何范围、HF/SRV/USRV 分区和物理系数保持与 v2 一致。
- 公共输入仍是物理坐标 `x,y,t`，模型内部归一化为 `x_hat,y_hat,t_hat`。
- 子网络使用局部特征 `[x_local, y_local, x_hat, y_hat, t_hat]`。
- HF 薄裂缝只沿裂缝长轴变化，裂缝厚度方向固定为 0.5，避免 0.01 m 裂缝孔径导致二阶导爆炸。
- 使用 `u = 1 + (1-exp(-decay*t))*N_region` 施加初始压力硬约束。
- 保留生产端 Dirichlet soft loss、外边界 Neumann soft loss、HF-SRV/SRV-USRV 压力连续和通量连续 loss。
- 增加轻量压力范围 loss，抑制无量纲压力跑出 `[0, 1.05]`。
- 增加 `hf_main_link` 高导流主裂缝连通 loss，用于避免生产端压降只停留在井口附近。

## 本轮调试结论

第一轮训练发现，如果 HF 子网直接使用裂缝厚度方向局部坐标，HF PDE loss 会被巨大二阶导主导。已修复为“HF 只沿裂缝长轴学习”，并将 MLP 最后一层零初始化，使初始解严格为常数初始压力场。

第二轮将 `interface_pressure` 权重从 20 提高到 50，`neumann` 权重从 0.5 降到 0.3，并继续训练到 600 epoch。

后续长训到 3000 epoch 后发现 HF 主裂缝远端仍偏高，说明仅靠 PINN 残差和端点 Dirichlet soft loss，生产端影响没有稳定传播到整条高导流裂缝。当前版本增加 `hf_main_link`，并从 epoch 3000 继续训练到 3300，使 HF 总压范围从约 `2.6~24.7 MPa` 收敛到约 `2.7~6.3 MPa`。

当前已生成：

- `outputs/checkpoints/final.pt`
- `outputs/logs/loss_history.csv`
- `outputs/tables/diagnostics.csv`
- `outputs/figures/field_P12_t*.png`
- `outputs/figures/field_P13_t*.png`
- `outputs/figures/field_Ptotal_t*.png`
- `outputs/figures/loss_history.png`

最终诊断摘要：

- `loss_pde`: about `3.87e-1`
- `loss_hf_main_link`: about `8.43e-2`
- `dirichlet_rmse`: about `1.02e-2`
- `rms_interface_pressure_hf_srv`: about `8.55e-2`
- `rms_interface_pressure_srv_usrv`: about `1.35e-2`
- `negative_pressure_points`: `0`
- `nonfinite_pressure_points`: `0`
- `Ptotal` range at final evaluation: about `2.68 MPa` to `20.37 MPa`
- `HF_Ptotal` range at final evaluation: about `2.70 MPa` to `6.31 MPa`

## 运行

```bash
python main.py test
python main.py train
python main.py train --resume --epochs 3600
python main.py train --resume --resume-path outputs/checkpoints/epoch_3000.pt --epochs 3300
python main.py evaluate
python main.py plot
```

或者分别运行：

```bash
python -m src.train --config config/default.yaml
python -m src.evaluate --config config/default.yaml --checkpoint outputs/checkpoints/final.pt
python plot_cloud_maps.py
python plot_loss_history.py
```

继续训练示例：

```bash
python -m src.train --config config/default.yaml --epochs 900 --resume outputs/checkpoints/final.pt
```
