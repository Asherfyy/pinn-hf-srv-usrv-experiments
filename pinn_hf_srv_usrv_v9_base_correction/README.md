# pinn_hf_srv_usrv_v9_base_correction

v9 是基于当前 v8 的独立实验版本。它保留 v8 的几何、三分区 MLP、空间 LHS 采样、混合时间采样、界面线段最低点数分配、界面通量法向归一化、生产端 Dirichlet soft loss、外边界 Neumann soft loss，以及 HF-SRV/SRV-USRV 界面压力和通量连续约束。

v9 的核心变化是：网络不再直接学习完整压力场，而是学习相对基准场的修正项。

```text
u(x, y, t) = u_base(x, y, t) + envelope(x, y, t) * correction_NN(x, y, t)
```

v9 默认使用 `float32`：

```yaml
runtime:
  dtype: "float32"
```

这与 v8 的 CPU/float64 稳定性优先思路不同。v9 使用 base-correction 降低学习难度，同时采用单精度以减少内存占用，并为后续 GPU 加速做准备。

## 基准场

默认配置中：

```yaml
constraint_mode: "ic_base_correction"
base_checkpoint: null
base_time_lag_days: 1.0
correction_envelope_power: 1.0
```

当 `base_checkpoint: null` 时：

```text
u_base = 1
```

也就是基准场为初始均匀无量纲压力场。由于最后一层零初始化，训练初始时 `correction_NN=0`，因此初始预测场就是 `u=1`。

如果以后指定：

```yaml
base_checkpoint: "outputs/checkpoints/final.pt"
```

则 v9 会加载该 checkpoint 作为冻结基准模型，并在：

```text
t_base = max(t - base_time_lag_days, t_min)
```

处评价 `u_base`。当前训练网络只学习相对该冻结场的修正。

## 时间包络

修正项前乘以时间包络：

```text
envelope(t) = (1 - exp(-decay_rate * t)) ** correction_envelope_power
```

因此在 `t=0` 时：

```text
envelope(0) = 0
u(x, y, 0) = u_base(x, y, 0)
```

默认 `u_base=1` 时，初始条件仍被硬满足。这样网络不需要从零学习完整压力场，而是从一个物理上合理的基准场出发，只学习生产端扰动和区域耦合造成的偏离。

## 与 v8 的关系

v9 保留 v8 的以下设置：

- HF/SRV/USRV 三个分区子网络。
- 当前 v8 的几何范围、主裂缝、五条次级裂缝和 SRV/USRV 分区。
- 当前 v8 的 PDE、边界条件和界面连续损失。
- 空间 LHS 采样。
- 连续 log1p LHS + 固定时间切片的混合时间采样。
- HF-SRV 和 SRV-USRV 界面线段最低点数分配。
- 基于界面法向方向的通量残差归一化。

v9 的对照目标是判断：在 v8 的采样和界面约束改进基础上，base-correction 输出结构是否能降低训练难度，并改善生产端压力扰动向主裂缝远端、次级裂缝和 SRV 区域传播的学习效果。

## 运行

```powershell
python main.py test
python main.py train
python main.py evaluate
python main.py plot
```

也可以分别运行：

```powershell
python -m src.train --config config/default.yaml
python -m src.evaluate --config config/default.yaml --checkpoint outputs/checkpoints/final.pt
python plot_cloud_maps.py
python plot_loss_history.py
```

## 注意

v9 的 checkpoint 与 v8 不兼容，因为 `project_version` 已变为：

```text
partition_mlp_v9_base_correction
```

如需使用 v8 或其他版本作为 `base_checkpoint`，建议先确认网络结构和配置兼容；否则应从头训练 v9。
