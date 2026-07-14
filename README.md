# HF/SRV/USRV 分区储层压力场 PINN 实验项目

本仓库用于记录一组基于 Python/PyTorch 的 PINN（Physics-Informed Neural Network）实验项目，研究对象是带有主裂缝、次级细裂缝、SRV 区域和 USRV 区域的二维储层压力场求解问题。

仓库中的多个版本不是彼此独立无关的项目，而是一条连续的建模与调试路线：从原始 PINN 实现，到分区 MLP、生产端硬约束、基准场修正、混合通量形式，再到取消分区网络的单 MLP 对照实验。保留这些版本的目的，是便于比较不同 PINN 结构和约束方式对压力传播、裂缝连通、界面连续性和训练稳定性的影响。

## 研究问题

当前项目关注的是 HF/SRV/USRV 复杂裂缝储层中的压力扩散问题。

主要几何区域包括：

- HF：Hydraulic Fracture，高导流裂缝区，包括一条水平主裂缝和多条垂直次级细裂缝。
- SRV：Stimulated Reservoir Volume，压裂改造区，围绕裂缝分布。
- USRV：Unstimulated Reservoir Volume，未改造储层区域。

主要研究目标：

- 在给定初始压力和生产端压力边界条件下，求解储层中随时间变化的压力场。
- 观察生产端压力降低是否能够合理传播到主裂缝远端、次级裂缝和 SRV/USRV 区域。
- 检查 HF/SRV、SRV/USRV 界面处压力是否连续、通量是否合理。
- 比较不同 PINN 结构是否能改善训练稳定性和压力云图连续性。
- 分析分区神经网络、单一神经网络、硬约束、基准场修正和混合通量形式对结果的影响。

## 物理与数学模型概述

大多数版本采用有效扩散形式的压力方程。无量纲化后，典型强形式为：

```text
u_t - Kx * u_xx - Ky * u_yy = 0
```

其中：

- `u` 是无量纲压力变量。
- `Kx`、`Ky` 是由物理扩散系数、孔隙度、时间尺度和空间尺度组合得到的无量纲有效扩散系数。
- 不同区域 HF/SRV/USRV 使用不同物理系数。
- 项目中通常同时求解两个压力组分 `u12` 和 `u13`。

边界与约束主要包括：

- 初始条件：初始压力场为均匀高压。
- 生产端 Dirichlet 边界：生产端压力随时间衰减至较低压力。
- 外边界 Neumann 边界：外边界无流。
- HF-SRV 界面压力连续。
- SRV-USRV 界面压力连续。
- 界面法向通量连续。
- 主裂缝中心线高导流连通约束。
- 次级裂缝中心线快速等压约束。
- 主裂缝与次级裂缝交汇处强连通约束。

v6 版本对 PDE 做了重要改写，引入显式通量变量，避免直接对压力求二阶导：

```text
q_x = -D_x * u_x
q_y = -D_y * u_y
Fai * u_t + q_x,x + q_y,y = 0
```

这种混合形式的目的，是降低 HF 极薄区域和极大扩散系数导致的二阶导训练不稳定问题。

## 几何与边界条件概述

项目的几何范围在各版本中保持一致或基本一致：

- 全局计算区域约为 `x = 0 ~ 360 m`，`y = 0 ~ 150 m`。
- SRV 位于储层右侧中部区域。
- 主裂缝为靠近 `y = 75 m` 的水平极薄裂缝，右端连接生产端。
- 多条次级细裂缝为垂直极薄裂缝，与主裂缝相交。
- 生产端 Dirichlet 边界位于主裂缝右端附近。

压力设定通常包括：

- 初始压力 `P_t0 = 25 MPa`。
- 生产端目标压力 `P_out = 3 MPa`。
- `P12` 和 `P13` 使用组分比例拆分，并分别做仿射无量纲化，使初始值接近 `1`，长期生产端目标接近 `0`。

## 损失函数组成

不同版本的损失函数略有差别，但主体结构基本包括：

- `loss_pde`：各区域 PDE 残差。
- `loss_dirichlet`：生产端压力边界损失。
- `loss_neumann`：外边界无流损失。
- `loss_interface_pressure`：界面压力连续损失。
- `loss_interface_flux`：界面通量连续损失。
- `loss_hf_main_link`：主裂缝高导流连通损失。
- `loss_hf_secondary_link`：次级细裂缝快速等压损失。
- `loss_hf_junction`：主裂缝与细裂缝交汇处强连通损失。
- `loss_pressure_range`：压力范围 soft penalty，用于抑制负压和过高压力。

这些损失项的权重主要在各版本的 `config/default.yaml` 中调整。

## 项目目录说明

```text
D:\CursorProject
├── pinn_hf_srv_usrv
├── pinn_hf_srv_usrv_v2
├── pinn_hf_srv_usrv_v3_partition_mlp
├── pinn_hf_srv_usrv_v4_hard_dirichlet
├── pinn_hf_srv_usrv_v5_base_correction
├── pinn_hf_srv_usrv_v6_mixed_flux
├── pinn_hf_srv_usrv_v7_v5based_single_mlp
├── pinn_hf_srv_usrv_v8_no_hf_local_coord
├── mph_extract_simple
├── README.md
└── .gitignore
```

各子项目一般包含：

```text
config/default.yaml       # 几何、物理参数、采样数量、训练参数、损失权重
main.py                   # IDE 直接运行入口
plot_cloud_maps.py        # 压力云图绘制入口
plot_loss_history.py      # loss 曲线绘制入口
src/model.py              # 神经网络结构
src/physics.py            # PDE 残差、边界导数、界面通量
src/losses.py             # 总损失函数
src/sampler.py            # PDE/边界/界面采样
src/train.py              # 训练流程
src/evaluate.py           # 诊断和评估
tests/                    # 单元测试
```

## 版本演化说明

### `pinn_hf_srv_usrv`

原始项目版本。

主要作用：

- 建立最初的 HF/SRV/USRV PINN 框架。
- 实现基本几何、采样、PDE 残差、边界条件和结果生成。
- 包含初始的诊断、采样可视化和结果后处理脚本。

当前默认 PINN 设置：

- 网络结构：使用单个全域 MLP，而不是 HF/SRV/USRV 分区子网络。网络输入维度为 `9`，输出维度为 `2`。
- 输入特征：`[x_hat, y_hat, t_hat]` 三个归一化坐标，加上 HF/SRV/USRV 的三维 one-hot 区域标识，再加上到外边界、HF 裂缝、生产端 Dirichlet 边界的三个距离特征。
- 输出变量：`u12` 和 `u13`，对应 `P12/P13` 两个压力分量的无量纲压力；后处理时再还原为 MPa，并可相加得到总压力。
- MLP 参数：默认 `hidden_layers=6`、`hidden_units=64`、激活函数 `tanh`，权重使用 Xavier 初始化。
- 约束模式：默认 `constraint_mode="dirichlet_hard"`，即采用 `u = u_ref(x,y,t) + B_D(x,y) * N_theta(x,y,t)`。其中 `B_D` 是生产端 Dirichlet 边界的 ADF 距离因子，在生产端边界上为 0，因此生产端压力由结构精确满足。
- Dirichlet 参考场：默认 `hard_constraint_reference="local_initial"`，生产端附近参考低压边界，远离生产端逐渐回到初始高压场，避免把全域参考场都拉到生产端低压。
- 初始条件：HF/SRV/USRV 初始压力均来自 `P_t0=25 MPa` 按组分比例拆分出的 `P12/P13`，作为 soft initial loss 参与训练；若切换到 `constraint_mode="ic_hard"`，则初始条件会被硬嵌入。
- PDE 形式：强形式二阶扩散方程，对 `u12/u13` 分别计算 `u_t`、`u_xx`、`u_yy`。当前默认启用 `use_physical_scaling=true`，把 COMSOL 风格的 `Fai` 与 `D` 系数转换为归一化坐标下的无量纲 PDE 系数。
- 分区物性：HF、SRV、USRV 分别使用不同孔隙/储集系数和扩散系数。默认 HF 扩散系数为 `1.0 m2/s`，SRV 为 `1e-7` 量级，USRV 为 `1e-9` 量级。
- 损失函数：总损失包含 PDE 残差、初始条件、生产端 Dirichlet 诊断/soft loss、外边界 Neumann 无流、HF-SRV/SRV-USRV 界面压力连续、界面法向通量连续，以及可选 COMSOL 数据监督项。
- PDE 残差归一化：默认开启 `pde_residual_normalization.enabled=true`，把 `u12/u13` 在 HF/SRV/USRV 中的六个 PDE 分量分别按固定尺度归一化，避免 HF 极大扩散系数主导优化。
- 采样方式：每个 epoch 重新随机采样。HF 点直接在主裂缝和 5 条次级裂缝矩形内采样；SRV/USRV 使用拒绝采样；界面点按线段长度采样并在 loss 中沿法向微小偏移到两侧。
- 时间采样：默认 `time_strategy="log1p_uniform"`，在 `log1p(t)` 空间均匀采样，再映射回 `0~1000 day`，以增强早期和中期压力衰减阶段的覆盖。
- 训练设置：默认 CPU-only、`float64`、Adam 优化器、`epochs=1000`、`learning_rate=1e-3`、梯度裁剪 `10.0`；LBFGS 接口保留但默认关闭。

该版本用于保存最早的建模思路和代码基础，同时也是后续 v2~v8 改造版本的对照基线。

### `pinn_hf_srv_usrv_v2`

独立重写版本。

主要作用：

- 在不覆盖原始项目的前提下，重新组织代码结构。
- 增加更清晰的训练、评估和绘图入口。
- 使用更规范的 `src/`、`tests/`、`config/` 结构。

当前默认 PINN 设置：

- 网络结构：使用单个全域 MLP，不区分 HF/SRV/USRV 子网络。网络输入维度为 `3`，输出维度为 `2`。
- 输入特征：只输入归一化坐标 `[x_hat, y_hat, t_hat]`。v2 不把区域 one-hot、距离特征、SDF 或 ADF 作为网络输入。
- 输出变量：输出 `u12` 和 `u13` 两个无量纲压力分量，对应物理压力 `P12/P13`。
- MLP 参数：默认 `hidden_layers=4`、`hidden_units=128`、激活函数 `tanh`，权重使用 Xavier 初始化。代码还支持 `silu`。
- 约束模式：v2 只实现 `constraint_mode="ic_hard"`，输出结构为 `u = 1 + t_hat * N_theta(x_hat,y_hat,t_hat)`。因此在 `t=0` 时，两个压力分量严格等于初始无量纲压力 `1`。
- 生产端边界：生产端 Dirichlet 不是硬约束，而是通过 soft loss 约束，目标值由 `P_t0=25 MPa`、`P_out=3 MPa`、`decay_rate=0.018` 和组分比例 `C13_C12` 给出。
- PDE 形式：使用有效扩散强形式 `u_t - kappa_x*u_xx - kappa_y*u_yy = 0`。其中 `K=D/Fai`，`kappa_x=K*T/Lx^2`，`kappa_y=alpha_y*K*T/Ly^2`，导数都在归一化坐标上计算。
- 分区物性：HF、SRV、USRV 分别使用不同 `Fai/D1/D2`。默认 HF 的扩散系数为 `1.0`，SRV 为 `1e-7` 量级，USRV 为 `1e-9` 量级。
- 损失函数：总损失包含 PDE 残差、生产端 Dirichlet soft loss、外边界 Neumann soft loss、HF-SRV/SRV-USRV 界面压力连续、界面有效扩散通量连续。v2 没有单独 initial loss，因为初始条件已被 `ic_hard` 结构满足。
- 界面处理：界面点先按几何线段采样，再沿法向偏移到两侧，过滤出真实跨越 HF-SRV 或 SRV-USRV 的点；压力跳跃和通量跳跃都作为 loss。
- 采样方式：HF/SRV/USRV 分区采样，并额外在 HF-SRV、SRV-USRV 附近加密 PDE 点。默认 `fixed_collocation_points=true`，训练开始时采样一次并复用；若设为 false，则按 `resample_every=500` 重采样。
- 时间采样：v2 只支持 `time_strategy="log1p_uniform"`，在 `log1p(t)` 空间随机均匀采样，再映射回 `0~1000 day`。
- 训练设置：默认 CPU-only、`float64`、Adam 优化器、`epochs=3500`、`learning_rate=1e-3`、梯度裁剪 `10.0`。配置里保留 `use_lbfgs` 字段，但当前 v2 训练主流程实际使用 Adam。

该版本是后续多个实验版本的结构基础之一，也可以作为“简化输入、初始条件硬约束、生产端 soft loss”的对照基线。

### `pinn_hf_srv_usrv_v3_partition_mlp`

分区 MLP 版本。

当前默认 PINN 设置：

- 网络结构：为 `HF`、`SRV`、`USRV` 分别建立独立子网络，由几何分区决定每个点调用哪个子网。
- 公共输入：外部仍输入物理坐标 `x,y,t`，模型内部归一化为 `z=[x_hat,y_hat,t_hat]`。
- 子网输入：默认 `subnet_input_dim=5`，每个子网接收 `[x_local, y_local, x_hat, y_hat, t_hat]`。
- 子网规模：HF 为 4 层、每层 96；SRV/USRV 为 5 层、每层 128；激活函数 `tanh`，输出 `u12/u13`。
- 初始条件硬约束：只支持 `constraint_mode="ic_hard"`，输出为 `u = 1 + (1-exp(-decay_rate*t))*N_region`；最后一层零初始化，使训练初始场严格为 `u=1`。
- HF 极薄裂缝局部坐标：对水平主裂缝保留长度方向 `x_local`，固定厚度方向 `y_local=0.5`；对竖直细裂缝保留长度方向 `y_local`，固定厚度方向 `x_local=0.5`。这等价于让 HF 子网主要沿裂缝中心线方向表达变化，不让 0.01 m 厚度方向通过局部坐标被放大成高频输入。
- PDE 形式：继续使用有效扩散强形式 `u_t-kappa_x*u_xx-kappa_y*u_yy=0`，按区域和变量计算残差并用固定尺度归一化。
- 边界和界面：生产端 Dirichlet、外边界 Neumann 仍为 soft loss；HF-SRV/SRV-USRV 使用两侧偏移点施加压力连续和法向通量连续。
- 裂缝连通增强：新增 `hf_main_link`、`hf_secondary_link`、`hf_junction`。主裂缝中心线被拉向生产端目标压力；细裂缝中心线与其主裂缝交汇点同压；交汇点附近成对采样强制主裂缝和细裂缝连通。
- 采样和训练：默认随机采样但固定 collocation 点训练；HF 主裂缝和细裂缝 link 点沿中心线均匀采样；默认启用 LBFGS，适合从 checkpoint 继续微调。

该版本主要用于解决一个重要问题：单纯依靠 PDE 残差和生产端边界，生产端压力影响难以稳定传播到整条主裂缝。

### `pinn_hf_srv_usrv_v4_hard_dirichlet`

生产端硬约束版本。

核心变化：

- 基于 v3。
- 将生产端 Dirichlet 边界从 soft loss 改为硬约束结构。
- 同时保留初始条件硬约束。

该版本用于验证：生产端压力边界如果被严格嵌入网络输出形式，是否能更稳定地影响 HF 主裂缝和周围储层。

### `pinn_hf_srv_usrv_v5_base_correction`

基准场修正版本。

核心变化：

- 基于 v3。
- 不再让网络直接从零学习完整压力场，而是学习相对基准场的修正：

```text
u = u_base(x, y, t) + envelope(x, y, t) * correction_NN(x, y, t)
```

- 默认 `u_base` 是初始均匀压力场。
- 也可以指定已有 checkpoint 作为冻结基准模型，让当前模型学习相对上一阶段或上一时刻的修正。

该版本的目标是降低 PINN 学习难度，使网络不必从零学习“生产端压力扰动如何传播”的完整过程。

### `pinn_hf_srv_usrv_v6_mixed_flux`

混合压力-通量形式版本。

核心变化：

- 基于 v3 的思想，但将 PDE 从压力强形式改为混合形式。
- 网络输出从：

```text
u12, u13
```

扩展为：

```text
u12, u13, q12_x, q12_y, q13_x, q13_y
```

- loss 包括：

```text
q + D * grad(u) = 0
Fai * u_t + div(q) = 0
```

- 不再直接对压力 `u` 求二阶导。

该版本主要用于处理 HF 极薄、扩散系数极大时，二阶导残差导致训练不稳定的问题。

需要注意：虽然 v6 避免了压力二阶导，但因为引入了通量输出和多个一阶导残差，计算量仍然较大。如果启用 LBFGS，每个 epoch 内部会多次调用 closure，训练可能显著变慢。

### `pinn_hf_srv_usrv_v7_v5based_single_mlp`

基于 v5 的单 MLP 对照版本。

核心变化：

- 保留 v5 的 base-correction 思路。
- 删除 HF/SRV/USRV 三个分区子网络。
- 使用一个共享 MLP 表达全域压力修正。
- 网络输入为：

```text
[x_hat, y_hat, t_hat]
```

- 输出仍为：

```text
u12, u13
```

该版本用于对照研究：分区网络本身是否有助于压力连续、裂缝连通和训练稳定。如果 v7 明显弱于 v5，说明分区网络对表达复杂多尺度储层具有实际价值；如果 v7 表现接近 v5，则说明当前问题可能更依赖损失设计和采样策略，而不是网络分区。

### `pinn_hf_srv_usrv_v8_no_hf_local_coord`

基于 v3 的 LHS 动态重采样对照版本。

注意：目录名保留了早期实验命名。当前 v8 不是完全取消 HF 局部坐标，而是将 HF 短轴局部坐标固定为 `0.5`，同时删除 HF 主裂缝/细裂缝连通增强 loss。

核心变化：

- 保留 v3 的分区 MLP、几何范围、物理参数、PDE、边界条件和界面连续损失。
- HF 物理 PDE 点仍分布在完整裂缝矩形内，不只采中心线。
- HF 网络输入中的短轴局部坐标固定为 `0.5`，避免 0.01 m 厚度局部坐标放大自动微分导数。
- 删除 `hf_main_link`、`hf_secondary_link`、`hf_junction` 三类 HF 主裂缝/细裂缝连通增强 loss。
- HF/SRV/USRV 空间采样使用 Latin hypercube；SRV/USRV 通过 LHS 候选点加几何拒绝筛选得到。
- 时间采样在 `log(1+t)` 空间使用一维 Latin hypercube。
- 默认 `fixed_collocation_points=false`，每 100 个 epoch 重新生成一套 LHS 点。
- 默认采样规模为 HF `6000`、SRV `12000`、USRV `6000`，并取消额外近界面 PDE 环带加密。
- HF-SRV 和 SRV-USRV 界面点仍单独保留，用于压力连续和法向通量连续。
- 界面通量残差使用基于界面法向的尺度归一化：`sn=max(1, |kappa_x*n_x|+|kappa_y*n_y|)`，并取界面两侧材料尺度的较大值归一化通量跳跃，避免竖直次级裂缝界面被较大的 `kappa_y` 人为弱化。

该版本用于验证：在不使用 HF 连通增强 loss 的情况下，二维 LHS 采样、动态重采样、HF 短轴局部坐标固定和界面通量法向归一化是否能改善 PDE 覆盖、训练稳定性和压力场合理性。因为输入特征含义、采样分布和界面残差尺度都发生变化，v8 建议从头训练，不建议继续使用旧 checkpoint。详细说明见 `pinn_hf_srv_usrv_v8_no_hf_local_coord/README.md`。

### `mph_extract_simple`

辅助数据提取项目。

该目录保存与外部模型文件提取相关的辅助文件，例如 XML、JSON 和资源文件。它不是 PINN 主训练项目，但可用于保留外部模型或仿真数据处理过程。

## 运行方式

进入某个子项目目录后运行。

例如运行 v7：

```powershell
cd D:\CursorProject\pinn_hf_srv_usrv_v7_v5based_single_mlp
python main.py test
python main.py train
python main.py evaluate
python main.py plot
```

也可以直接调用模块：

```powershell
python -m src.train --config config/default.yaml
python -m src.evaluate --config config/default.yaml --checkpoint outputs/checkpoints/final.pt
python plot_cloud_maps.py
python plot_loss_history.py
```

## 训练输出

各子项目训练后通常生成：

```text
outputs/checkpoints/       # 模型 checkpoint
outputs/logs/              # loss_history.csv
outputs/figures/           # 压力云图、loss 曲线、剖面图
outputs/tables/            # 诊断表
```

这些输出目录不会上传到 GitHub。

原因：

- checkpoint 文件可能很大。
- loss 日志和图片属于实验产物，频繁变化。
- GitHub 不适合直接管理大量训练输出。
- 保持仓库轻量，便于代码版本管理。

如果需要保存重要训练结果，建议使用：

- GitHub Releases
- Git LFS
- OneDrive、网盘或外部实验数据目录

## Git 忽略规则

本仓库根目录包含 `.gitignore`，主要忽略：

- Python 缓存：`__pycache__/`、`.pytest_cache/`
- 虚拟环境：`.venv/`
- IDE 文件：`.vscode/`
- 训练输出：`outputs/`
- 模型文件：`*.pt`、`*.pth`、`*.ckpt`
- 大型数组文件：`*.npy`、`*.npz`、`*.mat`、`*.h5`
- 生成图片和临时文件

因此，GitHub 仓库主要保存源码、配置、测试和说明文档。

## 当前建议使用方式

如果目标是继续做实验对比，建议优先关注：

- v3：分区 MLP 基线版本。
- v5：基准场修正版本。
- v6：混合压力-通量形式版本。
- v7：单 MLP 对照版本。
- v8：HF 短轴局部坐标固定、删除 HF 连通增强、使用 LHS 动态重采样的 v3 对照版本。

如果目标是分析“为什么压力分布不合理”，建议对比：

- v3 与 v7：观察分区网络是否关键。
- v5 与 v3：观察 base-correction 是否改善传播。
- v6 与 v3：观察避免二阶导后训练稳定性是否改善。
- v8 与 v3：观察 LHS 动态重采样在无 HF 连通增强条件下是否改善训练稳定性和压力场合理性。

如果目标是调参，优先检查：

- `config/default.yaml` 中的采样点数量。
- `loss_weights` 中各损失项权重。
- 是否启用 LBFGS。
- HF 主裂缝和次级裂缝连通损失是否过强或过弱。
- 绘图色阶是否统一，避免误判界面不连续。

## 注意事项

- 本仓库是实验研究代码集合，不是最终工业级求解器。
- 多个版本之间存在大量相似代码，这是为了保留实验可复现性，而不是追求最少重复。
- 不同版本 checkpoint 通常不兼容，尤其是网络输出维度或网络结构发生变化时。
- 若需要继续训练某一版本，应使用同一版本生成的 checkpoint。
- 若 GitHub 上缺少 `outputs/`，这是预期行为，不代表训练结果丢失。
