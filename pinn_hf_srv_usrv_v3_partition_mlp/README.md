# pinn_hf_srv_usrv_v3_partition_mlp

这是一个独立的 HF/SRV/USRV 分区 PINN 项目，不覆盖 `pinn_hf_srv_usrv` 或 `pinn_hf_srv_usrv_v2`。

## 当前默认 PINN 设置

v3 是“分区 MLP”版本。模型外部仍然接收物理坐标 `x,y,t`，内部先归一化为 `z=[x_hat,y_hat,t_hat]`，再根据解析几何判断采样点属于 `HF`、`SRV` 还是 `USRV`，最后调用对应区域的子网络。

### 网络结构

- 区域网络：`HF`、`SRV`、`USRV` 各自使用一个独立 MLP。
- 公共输入维度：`input_dim=3`，对应物理坐标 `x,y,t` 归一化后的 `x_hat,y_hat,t_hat`。
- 子网输入维度：默认 `subnet_input_dim=5`，子网实际输入为：

```text
[x_local, y_local, x_hat, y_hat, t_hat]
```

- 输出维度：`output_dim=2`，对应 `u12` 和 `u13` 两个无量纲压力变量。
- 默认激活函数：`tanh`。
- 默认子网规模：HF 为 4 个隐藏层、每层 96 个神经元；SRV 和 USRV 为 5 个隐藏层、每层 128 个神经元。
- 初始化方式：每个子网最后一层权重和偏置初始化为 0，使训练开始时 raw 输出为 0。
- 初始条件硬约束：当前版本只支持 `constraint_mode="ic_hard"`，网络输出形式为：

```text
u = 1 + (1 - exp(-decay_rate * t)) * N_region(x,y,t)
```

因此在 `t=0` 时严格满足 `u=1`，即初始压力场为常数初始场。

### HF 极薄裂缝局部坐标处理

普通矩形区域的局部坐标定义为：

```text
x_local = (x - rect.x_min) / rect.width
y_local = (y - rect.y_min) / rect.height
```

如果直接对 0.01 m 厚度的 HF 裂缝使用这个局部坐标，厚度方向的归一化会除以 `0.01`。网络只要在厚度方向产生很小的变化，换算回物理坐标后都会对应很大的梯度和二阶导，进而让强形式 PDE 中的 `u_xx/u_yy` 残差非常容易放大，训练会被极薄尺度主导。

v3 对 HF 区域做了特殊处理：

- 水平主裂缝：保留长度方向 `x_local`，固定厚度方向 `y_local=0.5`。
- 竖直细裂缝：保留长度方向 `y_local`，固定厚度方向 `x_local=0.5`。

也就是说，HF 子网仍然知道裂缝沿长度方向的位置，但不会通过局部坐标看到被 `0.01 m` 厚度强烈放大的横向坐标。模型输入里仍保留全局归一化坐标 `x_hat,y_hat,t_hat`，但全局坐标在 0.01 m 裂缝厚度内的变化量很小，不会像局部厚度坐标那样把极薄尺度直接放大到 `[0,1]`。这个处理相当于把 HF 近似成高导流、厚度方向近似等压的通道，使 HF 压力主要沿裂缝中心线长度方向和时间变化。

该处理不改变几何范围、HF/SRV/USRV 分区或采样区域，只改变 HF 子网看到的输入特征。PDE 残差仍然对物理坐标归一化后的 `x_hat,y_hat,t_hat` 求导。

### PDE、边界和界面损失

v3 仍使用强形式有效扩散方程：

```text
u_t - kappa_x * u_xx - kappa_y * u_yy = 0
```

其中 `kappa_x/kappa_y` 由各区域的渗透率/扩散系数和孔隙度等参数换算后得到，并在残差中做尺度归一化。

默认损失项包括：

- `pde`：HF、SRV、USRV 内部 PDE 残差。
- `dirichlet`：生产端压力边界 soft loss。
- `neumann`：外边界无流边界 soft loss。
- `interface_pressure`：HF-SRV 和 SRV-USRV 两侧压力连续。
- `interface_flux`：HF-SRV 和 SRV-USRV 两侧法向通量连续。
- `hf_main_link`：主裂缝中心线高导流连通损失，使主裂缝沿线更快接近生产端目标压力。
- `hf_secondary_link`：细裂缝中心线与其主裂缝交汇点同压，模拟细裂缝快速等压。
- `hf_junction`：主裂缝和细裂缝交汇处成对采样，强化交汇连通。
- `pressure_range`：约束无量纲压力大致留在 `[0, 1.05]` 内，减少非物理解。

### 采样和训练

- 默认 `sampling_mode=random`，但固定 collocation 点训练；可切换为均匀采样。
- 随机采样模式下，主裂缝和细裂缝的 link 点仍沿中心线均匀采样，避免高导流裂缝约束被随机点遗漏。
- HF/SRV、SRV/USRV 界面附近额外加密采样，用于增强界面两侧 PDE 和连续性约束。
- 默认训练使用 `float64`、CPU、固定采样点、checkpoint 保存和 loss CSV 日志。
- 当前版本已经加入 `use_lbfgs` 开关，可从已有 checkpoint 继续训练并使用 LBFGS 做后期微调。

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
