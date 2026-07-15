# pinn_hf_srv_usrv_v8_no_hf_local_coord

v8 是基于 `pinn_hf_srv_usrv_v3_partition_mlp` 的独立对照实验版本。它保留 v3 的三分区 MLP、几何范围、物理参数、强形式扩散 PDE、生产端 Dirichlet soft loss、外边界 Neumann soft loss，以及 HF-SRV/SRV-USRV 界面压力连续和法向通量连续约束。

需要注意：目录名中的 `no_hf_local_coord` 来自早期实验命名。当前 v8 实现并不是完全取消 HF 局部坐标，而是采用“HF 短轴局部坐标固定为 0.5”的处理方式；同时删除 HF 主裂缝/细裂缝连通增强 loss，并引入 LHS 动态重采样和界面通量法向尺度归一化。

v8 的目标不是保证结果一定优于 v3，而是把“没有额外 HF 连通增强时，分区 MLP + HF 短轴坐标固定 + LHS 动态重采样 + 法向通量归一化”这一组设置单独拿出来观察，判断压力传播、界面连续性和训练稳定性是否更合理。

## 版本定位

相对 v3，v8 的核心变化包括：

- 保留 HF/SRV/USRV 三个独立子网络，不改成单网络。
- 保留原有几何范围、主裂缝、五条次级裂缝、SRV/USRV 分区和生产端位置。
- 保留强形式扩散 PDE：`u_t - kappa_x*u_xx - kappa_y*u_yy = 0`。
- 保留生产端 Dirichlet soft loss 和外边界 Neumann soft loss。
- 保留 HF-SRV、SRV-USRV 界面压力连续和法向通量连续。
- 删除 `hf_main_link`、`hf_secondary_link`、`hf_junction` 三类 HF 连通增强 loss。
- HF PDE 点仍采样在完整 0.01 m 裂缝矩形内，不退化为中心线采样。
- HF 子网输入中的短轴局部坐标固定为 `0.5`，减少极薄裂缝短轴坐标对自动微分导数的放大。
- HF/SRV/USRV、边界和界面空间位置默认使用 Latin hypercube sampling；时间使用连续 log1p LHS 与固定时间切片混合采样。
- 默认关闭固定 collocation 点，每 `100` 个 epoch 重新生成一套训练点。
- 界面通量残差不再使用统一 `max(1, |kappa_x|, |kappa_y|)` 归一化，而是按界面法向方向计算尺度。

## PINN 结构

v8 使用分区 MLP：

```text
HF   -> HF 子网络
SRV  -> SRV 子网络
USRV -> USRV 子网络
```

每个点先根据几何位置判断属于哪个区域，再调用对应子网络。默认每个子网络输入为：

```text
[x_local, y_local, x_hat, y_hat, t_hat]
```

其中：

- `x_hat, y_hat, t_hat` 是全局归一化坐标。
- `x_local, y_local` 是对应区域矩形内的局部坐标。
- HF 的局部坐标经过特殊处理，见下一节。

网络输出为：

```text
[u12, u13]
```

它们是 `P12/P13` 两个压力分量的无量纲形式。后处理时再还原为 MPa，并可相加得到总压力。

默认网络规模：

```yaml
region_hidden_layers:
  HF: 4
  SRV: 5
  USRV: 5
region_hidden_units:
  HF: 96
  SRV: 128
  USRV: 128
activation: "tanh"
constraint_mode: "ic_hard"
```

`constraint_mode: "ic_hard"` 表示初始条件通过输出结构硬嵌入，使 `t=0` 时 `u12/u13` 严格等于初始无量纲压力 `1`。

## HF 局部坐标处理

普通矩形局部坐标定义为：

```text
x_local = (x - rect.x_min) / rect.width
y_local = (y - rect.y_min) / rect.height
```

HF 裂缝厚度只有 `0.01 m`。如果把厚度方向完整映射到 `[0, 1]`，那么物理空间中极小的横向变化会在局部特征中变成很大的归一化变化。对强形式 PDE 来说，这会放大 `u_xx/u_yy` 中的短轴方向二阶导，使训练更容易出现局部振荡或 loss 不稳定。

因此 v8 对 HF 子网输入采用：

- 水平主裂缝：保留长度方向 `x_local`，固定厚度方向 `y_local = 0.5`。
- 竖直次级裂缝：固定厚度方向 `x_local = 0.5`，保留长度方向 `y_local`。

这只改变进入 HF 子网的局部特征，不改变 PDE 点的真实物理位置。HF 采样点仍然在完整裂缝矩形内，且全局坐标 `x_hat/y_hat/t_hat` 仍保留在输入中。因此 v8 不是严格的一维裂缝模型，而是弱化 HF 短轴方向局部坐标对网络表达和自动微分的影响。

## 删除 HF 连通增强

v3 中曾额外加入三类 HF 连通增强 loss：

```text
hf_main_link
hf_secondary_link
hf_junction
```

它们的作用是让主裂缝中心线快速受生产端影响、让次级裂缝快速等压，并强化主裂缝与次级裂缝交汇处连通。

v8 删除这些项后，HF 远端和次级裂缝是否能形成合理压力响应，主要依赖：

- HF 区域 PDE 残差。
- 生产端 Dirichlet soft loss。
- HF-SRV 界面压力连续。
- HF-SRV 界面法向通量连续。
- HF 子网络自身表达能力。
- LHS 动态重采样对训练点覆盖的改善。

如果 v8 中主裂缝远端或次级裂缝响应仍然不足，这正是该版本要暴露的问题：单靠 PDE、边界和界面约束是否足以学出高导流裂缝内的快速压力传播。

## LHS 采样

`sampler.sampling_mode` 支持：

```text
random
uniform
latin_hypercube
lhs
latin-hypercube
```

默认设置为：

```yaml
sampling_mode: "latin_hypercube"
time_sampling_mode: "latin_hypercube"
time_strategy: "hybrid_log1p_fixed"
time_continuous_fraction: 0.5
fixed_collocation_points: false
resample_every: 100
```

各类采样含义如下：

| 对象 | 默认采样方式 | 说明 |
| --- | --- | --- |
| HF PDE 点 | 各裂缝矩形内二维 LHS | 按主裂缝和细裂缝矩形面积分配点数 |
| SRV PDE 点 | SRV 背景矩形内 LHS 候选点 | 通过几何判断排除 HF |
| USRV PDE 点 | 全域矩形内 LHS 候选点 | 通过几何判断排除 SRV 和 HF |
| 时间点 | 连续 log1p LHS + 固定时间切片 | 50% 连续采样，50% 固定在关键时刻 |
| Dirichlet 边界 | 生产端线段一维 LHS | 用于生产端压力 soft loss |
| Neumann 边界 | 外边界线段一维 LHS | 右边界会排除生产端邻域 |
| HF-SRV 界面 | 界面线段一维 LHS | 用于压力连续和法向通量连续 |
| SRV-USRV 界面 | 界面线段一维 LHS | 用于压力连续和法向通量连续 |

SRV 和 USRV 是带排除区的非规则区域，因此程序采用“LHS 候选点 + 几何拒绝筛选”。筛选后不再是严格数学意义的完整 LHS，但通常比完全独立随机抽样覆盖更均匀。

时间采样采用混合方式：

```yaml
time_continuous_fraction: 0.5
time_fixed_slices:
  - {time: 1.0, fraction: 0.1}
  - {time: 100.0, fraction: 0.1}
  - {time: 500.0, fraction: 0.1}
  - {time: 1000.0, fraction: 0.2}
```

这表示每一批需要时间坐标的采样点中，50% 仍在 `log(1+t)` 空间做连续 LHS，用于保持早期时间加密；另外 50% 固定在 `1 d`、`100 d`、`500 d` 和 `1000 d`，用于保证典型观察时刻和末期压力场都有足够训练点。

HF-SRV 界面线段不是单纯按线段长度分配点数。程序会先给每条 HF 边界线段分配最低点数，再把剩余点数按线段长度分配。这样 0.01 m 的裂缝尖端和主裂缝端部不会因为长度太短而得到 0 个界面点，同时长侧壁仍然保留更高采样密度。

## 默认采样数量

当前 `config/default.yaml` 中的主要采样设置：

```yaml
n_pde_hf: 6000
n_pde_srv: 12000
n_pde_usrv: 6000
n_near_hf_srv: 0
n_near_srv_usrv: 0
n_dirichlet: 3000
n_neumann: 3000
n_interface_hf_srv: 6000
n_interface_srv_usrv: 4000
min_points_per_hf_srv_interface_segment: 100
min_points_per_srv_usrv_interface_segment: 100
```

`n_near_hf_srv` 和 `n_near_srv_usrv` 设为 `0` 并不表示取消界面约束。界面连续性由 `n_interface_hf_srv` 和 `n_interface_srv_usrv` 单独负责。PDE 点负责区域内部方程，界面点负责压力连续和通量连续，两类点的作用不同。

默认几何下 HF 包含 1 条水平主裂缝和 5 条竖直次级裂缝，共 6 个矩形、24 条候选边界线段。`min_points_per_hf_srv_interface_segment: 100` 会先占用 `2400` 个 HF-SRV 界面点，剩余 `3600` 个点再按线段长度分配。SRV-USRV 的 3 条界面线段也使用同样机制，默认每条至少 `100` 个点。

## 界面通量归一化

v8 当前的界面通量残差采用基于界面法向方向的尺度归一化。

旧的统一尺度写法相当于：

```text
scale = max(1, |kappa_x|, |kappa_y|)
```

这个尺度不区分界面方向。由于全局长度尺度在 `x` 和 `y` 方向不同，归一化后的 `kappa_y` 往往大于 `kappa_x`。对于水平主裂缝上下界面，法向在 `y` 方向，通量主要由 `kappa_y * du/dy` 控制，使用最大尺度基本合理；但对于竖直次级裂缝左右界面，法向在 `x` 方向，通量主要由 `kappa_x * du/dx` 控制。如果仍除以较大的 `kappa_y`，竖直裂缝界面的通量跳跃会被人为缩小，导致 `interface_flux` loss 对次级裂缝左右界面的惩罚偏弱。

现在的尺度定义为：

```text
sn = max(1, |kappa_x * n_x| + |kappa_y * n_y|)
```

并且界面两侧材料分别计算 `sn`，最终取较大值归一化通量跳跃：

```text
normalized_flux_jump = raw_flux_jump / max(sn_minus, sn_plus)
```

这样：

- 竖直界面主要按 `kappa_x` 缩放。
- 水平界面主要按 `kappa_y` 缩放。
- 不同朝向界面的同等相对通量误差会受到更接近的 loss 惩罚。
- 次级裂缝左右界面的通量连续约束不会被较大的 `kappa_y` 人为削弱。

该实现位于 `src/physics.py` 的 `normal_flux_scale()` 和 `interface_residual()` 中。

## 损失函数

当前总 loss 包含：

- `pde`：HF、SRV、USRV 内部强形式 PDE 残差。
- `dirichlet`：生产端压力 soft loss。
- `neumann`：外边界无流 soft loss。
- `interface_pressure`：HF-SRV 和 SRV-USRV 压力连续。
- `interface_flux`：HF-SRV 和 SRV-USRV 法向通量连续。
- `pressure_range`：无量纲压力范围 soft penalty。

当前总 loss 不再包含：

- `hf_main_link`
- `hf_secondary_link`
- `hf_junction`

默认 loss 权重：

```yaml
pde: 10.0
dirichlet: 20.0
neumann: 5
interface_pressure: 5.0
interface_flux: 20
pressure_range: 0.1
```

## 训练设置

当前默认训练设置：

```yaml
epochs: 6000
learning_rate: 3.0e-4
fixed_collocation_points: false
resample_every: 100
use_lbfgs: false
save_every: 500
```

`fixed_collocation_points=false` 时，`train.py` 会按 `resample_every` 定期重新调用 `sampler.sample_all()`。由于采样模式是 LHS，每次重采样都会生成一套新的分层采样点，用于减轻固定 collocation 点导致的局部过拟合。

本版本改变了 HF 输入特征含义、训练点分布和界面通量归一化方式，不建议继续使用旧 checkpoint。建议从头训练。

## 运行入口

在 v8 项目目录下运行：

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

如果之前已经训练过其他 v8 设置，建议至少清理或备份：

```text
outputs/checkpoints/
outputs/logs/loss_history.csv
outputs/figures/
```

避免新旧训练日志、checkpoint 和云图混在一起。

## 结果观察重点

建议重点观察：

- `loss_pde` 是否比随机采样更平稳。
- `loss_interface_flux_hf_srv` 和 `loss_interface_flux_srv_usrv` 是否更均衡。
- 次级裂缝左右界面附近是否出现更合理的压力梯度。
- 主裂缝远端是否仍难以受到生产端边界影响。
- 取消 HF link loss 后，细裂缝是否失去快速等压特征。
- HF 区域是否出现厚度方向振荡。
- HF-SRV 界面压力连续和法向通量连续是否恶化。
- 云图色阶是否统一，避免把绘图假象误判为真实不连续。

v8 适合作为 v3 的消融对照版本：如果 v8 弱于 v3，说明 HF 连通增强 loss 对当前问题可能仍然必要；如果 v8 接近或优于 v3，说明采样、局部特征处理和通量归一化可能比额外连通 loss 更关键。
