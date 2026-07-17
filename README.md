# HF/SRV/USRV 分区储层压力场 PINN 实验项目

本仓库用于记录一组基于 Python/PyTorch 的 PINN（Physics-Informed Neural Network）实验项目，研究对象是带有主裂缝、次级细裂缝、SRV 区域和 USRV 区域的二维储层压力场求解问题。

仓库中的多个版本不是彼此独立无关的项目，而是一条连续的建模与调试路线：从原始 PINN 实现，到分区 MLP、生产端硬约束、基准场修正、混合通量形式，再到取消分区网络的单 MLP 对照实验、v10 的 RDFM/I-PINN 有限元离散残差路线、v11 的 E-PINN/EDFM 有限体积残差路线、v12 的线裂缝强形式 PINN 路线，以及 v13 的 mesh-free 单压力 P12 纯物理训练路线。保留这些版本的目的，是便于比较不同 PINN 结构、裂缝表示方式和约束方式对压力传播、裂缝连通、界面连续性和训练稳定性的影响。

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

v10 版本另开一条路线：不再用随机采样点上的强形式 PDE residual 训练，而是借鉴 RDFM/I-PINN 文献，把神经网络预测值放到规则 Q1 FEM 网格节点上，再用有限元离散残差作为 loss。对应的半离散形式为：

```text
M * du/dt + A * u = 0
```

其中 `M` 是质量矩阵，`A` 是刚度矩阵。时间推进使用隐式 Euler：

```text
r = M * (u_next - u_prev) / dt + A * u_next
loss = mean(abs(r_free))
```

这里的 `r_free` 只包含非 Dirichlet 自由节点残差。生产端 Dirichlet 值在计算残差前直接覆盖到节点压力上；外边界无流条件不再作为 soft loss，而是作为 FEM 弱形式的自然边界条件进入：由于外边界通量项为零，组装时不额外加入边界载荷即可。

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
├── pinn_hf_srv_usrv_v9_base_correction
├── pinn_hf_srv_usrv_v10_rdfm_ipinn
├── pinn_hf_srv_usrv_v11_epinn_edfm_singlephase
├── pinn_hf_srv_usrv_v12_line_hf_random
├── pinn_hf_srv_usrv_v13_meshfree_p12_fvm
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

v10 因为采用 RDFM/FEM 离散残差，额外包含：

```text
src/rdfm_mesh.py          # 规则矩形 Q1 FEM 网格、节点、单元、Dirichlet/free 节点
src/rdfm_fractures.py     # 将薄矩形 HF 转换为 RDFM 裂缝中心线、开度、切向和积分分段
src/rdfm_assembly.py      # 组装基质/裂缝质量矩阵和刚度矩阵
src/fem_solver.py         # 用同一套 RDFM/FEM 矩阵做 PCG 参考求解
src/plot_mesh.py          # 绘制规则 FEM 网格、RDFM 裂缝中心线和生产端节点
src/plot_fields.py        # 基于节点快照和三角剖分绘制压力云图
src/plot_sections.py      # 沿主裂缝和次级裂缝抽取压力剖面
outputs/snapshots.npz     # v10 的主要结果文件，保存各时间步节点压力快照；可来自 train 或 solve
```

v11 因为采用 E-PINN/EDFM/FVM 单相流路线，额外包含：

```text
main_POD.py              # POD-MLP 降阶代理模型的一键入口：生成快照、训练、预测、气量/同位素后处理和绘图
src/edfm_grid.py          # 规则 cell-centered FVM 网格、EDFM 裂缝段、mm/mf/ff 连接和 sparse graph edges
src/model.py              # sparse E-PINN 压力向量网络：anchoring、adaptive ReLU、skip/gate、message passing
src/losses.py             # 单相隐式 Euler FVM residual、双压力分量 residual 和 BHP hard constraint
src/fvm_solve.py          # 同一 EDFM/FVM 系统的直接隐式参考解，输出 direct_fvm_edfm 快照
src/pod_generate_data.py  # 复用 direct FVM/EDFM 求解器生成 POD 训练快照，不覆盖默认 snapshots.npz
src/pod_train.py          # 用训练快照拟合 POD 基并训练 time/BHP -> POD 系数的小型 MLP
src/pod_predict.py        # 在训练时间区间内对任意时间直接预测 P12/P13/Ptotal 全场
src/pod_evaluate.py       # 分开评估 POD 截断误差和 POD-MLP 系数预测误差
src/gas_metrics.py        # 基于 MPa 压力场计算 HF/SRV/USRV 区域累产气量、速率和甲烷碳同位素
src/pod_gas_postprocess.py # 调用 POD 预测或读取快照，输出 Q_HF/Q_SRV/Q_USRV/Q_cum 和 delta_prod 曲线
src/plot_mesh.py          # 绘制 EDFM 网格、裂缝段和 BHP cell
outputs/snapshots.npz     # v11 的 cell-centered pressure_mpa 快照，形状可为 [time, cell, component]
outputs/pod/              # POD 专用快照、POD 基、预测 archive 和气量后处理 archive
outputs/figures/pod/      # POD 误差图、预测压力图、累产气量图和 delta_prod 图
POD_MLP_README.md         # POD-MLP 降阶代理模型说明
GAS_POSTPROCESSING_CN_NOTES.md   # 气量与甲烷碳同位素后处理中文详解
```

v12 因为采用 v3 分区 MLP + 线裂缝 + 随机非固定 collocation 路线，重点变化在：

```text
src/geometry.py           # 将原 HF 薄矩形转换为裂缝中心线；HF 只在中心线上作为 REGION_HF
src/sampler.py            # HF PDE 点、HF-SRV 耦合点、主/次裂缝 link 点全部在线段上随机采样
src/physics.py            # line_hf_srv_residual：裂缝线点与法向偏移 SRV 点的压力耦合
src/losses.py             # 保留 v3 强形式 PDE 主路线，但 HF-SRV interface loss 改为线-面耦合
src/model.py              # 分区 MLP；默认每个子网输入 [x_hat, y_hat, t_hat]，可选 5D local feature
outputs/checkpoints/      # v12 训练 checkpoint；默认不上传
```

v13 因为采用 mesh-free 单压力 P12 + 纯 PDE/边界训练路线，重点变化在：

```text
src/sampler.py            # 随机非固定采样；HF 只在线裂缝上采样，并支持残差自适应 PDE 候选点
src/physics.py            # SRV/USRV 二维扩散残差；HF 一维切向裂缝扩散残差
src/losses.py             # 单压力 P12 PDE、边界、界面、线裂缝和 causal time weighting
src/model.py              # 分区 MLP；ic_base_correction + 生产端 hard Dirichlet
src/fvm_reference.py      # 仅用于离线 FVM 对照；不进入训练 loss
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
- 时间采样改为“连续时间采样 + 固定时间切片”混合方式：50% 点仍使用 `log(1+t)` 空间的一维 LHS，10% 固定在 `t=1 d`，10% 固定在 `t=100 d`，10% 固定在 `t=500 d`，20% 固定在 `t=1000 d`。这样保留早期时间加密，同时保证晚期 1000 d 有足够训练点。
- 默认 `fixed_collocation_points=false`，每 100 个 epoch 重新生成一套 LHS 点。
- 默认采样规模为 HF `6000`、SRV `12000`、USRV `6000`，并取消额外近界面 PDE 环带加密。
- HF-SRV 和 SRV-USRV 界面点仍单独保留，用于压力连续和法向通量连续；界面点先保证每条线段最低点数，再把剩余点按线段长度分配，避免 0.01 m 裂缝尖端和主裂缝端部没有采样点。
- 界面通量残差使用基于界面法向的尺度归一化：`sn=max(1, |kappa_x*n_x|+|kappa_y*n_y|)`，并取界面两侧材料尺度的较大值归一化通量跳跃，避免竖直次级裂缝界面被较大的 `kappa_y` 人为弱化。

该版本用于验证：在不使用 HF 连通增强 loss 的情况下，二维 LHS 空间采样、混合时间采样、动态重采样、HF 短轴局部坐标固定、界面线段最低点数分配和界面通量法向归一化是否能改善 PDE 覆盖、训练稳定性和压力场合理性。因为输入特征含义、采样分布和界面残差尺度都发生变化，v8 建议从头训练，不建议继续使用旧 checkpoint。详细说明见 `pinn_hf_srv_usrv_v8_no_hf_local_coord/README.md`。

### `pinn_hf_srv_usrv_v9_base_correction`

基于当前 v8 的基准场修正版本。

核心变化：

- 保留 v8 的几何、分区 MLP、空间 LHS、混合时间采样、界面线段最低点数分配和界面通量法向归一化。
- 网络不再直接学习完整压力场，而是学习相对基准场的修正：

```text
u = u_base(x, y, t) + envelope(x, y, t) * correction_NN(x, y, t)
```

- 默认 `u_base=1`，即初始均匀无量纲压力场。
- `envelope(t) = (1-exp(-decay_rate*t))**correction_envelope_power`，因此 `t=0` 时修正项为 0，初始条件仍被硬满足。
- 可选 `model.base_checkpoint`，用于加载冻结 checkpoint 作为基准模型；此时基准场可在 `max(t-base_time_lag_days,t_min)` 处评价。
- 默认使用 `float32`，用于降低内存占用，并为后续 GPU 加速测试做准备。
- 默认 `constraint_mode="ic_base_correction"`，checkpoint 版本号为 `partition_mlp_v9_base_correction`，不与 v8 checkpoint 直接混用。

该版本用于验证：在 v8 的采样和界面约束改进基础上，base-correction 输出结构是否能降低 PINN 从零学习压力传播的难度，并改善生产端扰动向主裂缝远端、次级裂缝和 SRV 区域传播的学习效果。

### `pinn_hf_srv_usrv_v10_rdfm_ipinn`

基于 RDFM（Reinterpreted Discrete Fracture Model）和 I-PINN 思想的有限元离散残差版本。

这个版本和 v1-v9 的差别最大。v1-v9 基本都是“连续坐标输入神经网络 + 自动微分强形式 PDE + 边界/界面 soft loss”的 PINN 路线；v10 改成“神经网络预测 FEM 网格节点压力 + 离散弱形式残差”的路线。它更接近文献中的 I-PINN：神经网络仍然参与表达未知场，但物理约束不再依赖二阶自动微分和大量随机 collocation 点，而是来自已经组装好的 FEM/RDFM 稀疏矩阵。

核心建模方式：

- 基质区域为完整矩形储层 `0~360 m` × `0~150 m`，使用代码生成的规则 Q1 矩形 FEM 网格。
- 默认网格为 `mesh.nx=180`、`mesh.ny=150`，因此 `y=75 m` 主裂缝中心线正好落在节点线上。
- 基质单元按 cell center 判定为 `SRV` 或 `USRV`，不再把 HF 当成二维薄矩形区域参与 PDE 采样。
- 原来的主裂缝和 5 条次级裂缝从薄矩形转换为 RDFM 中心线段。
- 裂缝开度 `epsilon` 取原薄矩形短边长度；裂缝切向取长轴方向。
- HF 储量和导流能力通过裂缝线积分加入全局质量矩阵和刚度矩阵，而不是通过 HF 区域强形式 PDE 点加入。

v10 组装的矩阵包括：

```text
matrix mass:
  ∫Ω Fai * N_i * N_j dx

matrix stiffness:
  ∫Ω D * grad(N_i) · grad(N_j) dx

fracture mass:
  Σ_l ∫Γ_l epsilon_l * Fai_HF * N_i * N_j ds

fracture stiffness:
  Σ_l ∫Γ_l epsilon_l * D_HF * d_tau(N_i) * d_tau(N_j) ds
```

对 `u12` 和 `u13` 分别组装刚度矩阵；质量矩阵当前共用同一套 `Fai`。裂缝贡献通过 Q1 形函数映射到裂缝穿过的基质单元节点，因此裂缝和基质的耦合由全局矩阵自然完成。

时间推进方式：

- `time_grid.times_days` 默认是 `[0, 1, 10, 100, 500, 1000]`。
- 这些不是随机采样时间，而是实际离散时间步快照。
- 训练从 `t=0` 的均匀初始压力开始，依次推进 `0->1`、`1->10`、`10->100`、`100->500`、`500->1000 day`。
- 每个时间步都训练同一个全局 MLP；进入下一个时间步时继承上一步权重，等价于文献中的 transfer initialization 思路。
- 每个时间步保存当前步最小 loss 对应的节点压力快照。

边界条件处理：

- 生产端 Dirichlet 边界位于主裂缝右端附近，即 `x=360 m`、`y≈75 m` 的短竖直线段。
- v10 会在网格节点中找到该线段上的节点；如果线段比网格更细，则退化为选择最接近线段中点的节点。
- Dirichlet 节点在计算残差前被硬覆盖为目标生产压力，不再依赖 `loss_dirichlet`。
- 其他外边界默认是零通量边界。由于使用弱形式，齐次 Neumann 边界是自然边界；代码中不额外加入边界通量载荷，也不再使用 `loss_neumann`。
- SRV/USRV 和裂缝-基质之间的交换由弱形式矩阵和共享节点/形函数耦合表达，不再单独保留 v9 的 `interface_pressure`、`interface_flux`、`hf_link` 主损失。

神经网络与输出：

- v10 使用单个全局 MLP，输入恢复为纯二维归一化坐标 `[x_hat, y_hat]`。
- 输出为两个无量纲压力分量 `[u12, u13]`。
- 网络不直接输入时间；时间依赖来自逐时间步训练和继承权重。
- 默认激活函数为 `relu`，贴近文献设置。
- 最后一层零初始化，初始网络输出接近均匀初始压力 `u=1`。
- 后处理时通过已有 affine 归一化工具恢复为 `P12/P13/Ptotal`，单位 MPa。

运行入口：

```powershell
cd D:\CursorProject\pinn_hf_srv_usrv_v10_rdfm_ipinn
python main.py test
python main.py train
python main.py solve
python main.py evaluate
python main.py plot
python main.py mesh
```

快速冒烟训练可以减少每步 epoch：

```powershell
python main.py train --epochs-per-step 2
```

默认完整训练预算为：

```yaml
training:
  epochs_per_step: 200
```

默认完整训练会进行 5 个时间步、共 1000 个 local epoch 记录。主要输出包括：

```text
outputs/snapshots.npz             # 节点坐标、三角剖分、各时间步 u12/u13 快照
outputs/checkpoints/final.pt      # 最终模型权重和训练状态
outputs/checkpoints/step_*.pt     # 各时间步 checkpoint
outputs/logs/loss_history.csv     # 每个 local epoch 的残差 loss
outputs/tables/diagnostics.csv    # 各快照压力范围和非有限值统计
outputs/figures/field_*.png       # P12/P13/Ptotal rainbow 云图
outputs/figures/profile_*.png     # 主裂缝和次级裂缝压力剖面
outputs/figures/loss_history.png  # loss 曲线
outputs/figures/mesh_*.png        # FEM 网格、RDFM 裂缝和生产端节点图
```

当前默认配置下已经通过的基本验收包括：

- `python main.py test`：11 个测试通过。
- `python main.py train`：默认 200 epochs/step 可完成 I-PINN 训练。
- `python main.py solve`：可用同一套 RDFM/FEM 矩阵生成 PCG 参考快照，并备份已有 `outputs/snapshots.npz`。
- `python main.py evaluate`：可生成 `diagnostics.csv`，无 NaN/Inf 压力点。
- `python main.py plot`：可生成 rainbow 压力云图、裂缝剖面和 loss 曲线。

当前需要区分两类输出：`train` 是 I-PINN 残差训练结果，适合研究神经网络是否能逼近离散方程解；`solve` 是同一套 RDFM/FEM 离散系统的 PCG 参考解，适合检查裂缝附近压力传播是否符合高导流裂缝预期。最近一次调试表明，单靠全局 MLP 训练可能在主裂缝附近欠收敛；因此判断 v10 物理合理性时，应优先把 `train` 快照与 `solve` 参考快照对比，再继续做时间步加密、网格加密和 COMSOL/FEM 参考解对比。

v10 的适用场景：

- 需要避免强形式 PINN 对二阶导数和复杂界面 soft loss 的依赖。
- 希望把裂缝从 0.01 m 薄矩形 PDE 区域改为低维中心线模型。
- 希望更忠实地复现 RDFM/I-PINN 文献中的“离散残差训练”思路。
- 接受方法对代码生成 FEM 网格有依赖，而不是完全无网格。

v10 当前的限制：

- 首版只支持代码生成的规则矩形 Q1 网格，不导入 COMSOL 网格。
- 当前实现只覆盖压力扩散问题，不实现文献中的污染物输运和 bound-preserving 浓度约束。
- 时间网格默认较粗，适合先看趋势；若要做严肃对比，应加密 `time_grid.times_days`。
- RDFM 裂缝中心线积分依赖网格和线段切分，网格尺寸变化可能影响数值结果，需要做网格无关性检查。
- 当前 loss 是节点残差平均绝对值，尚未加入自适应权重、局部误差指示器或多网格训练。

### `pinn_hf_srv_usrv_v11_epinn_edfm_singlephase`

基于 2024 年两相流 E-PINN/EDFM 文献思想的单相流版本。v11 不再使用“坐标到压力”的 MLP，也不使用 v10 的 RDFM/FEM 节点残差；它把上一时间步全场压力向量输入 E-PINN，输出下一时间步全场压力向量，并用 cell-centered FVM residual 训练。
在此基础上，当前 v11 又增加了一条独立的 POD-MLP 降阶代理模型路线：先用 direct FVM/EDFM 生成压力快照，再用 POD 提取低维模态，最后训练一个小型 MLP 从时间/BHP 特征直接预测 POD 系数。因此 v11 现在同时包含三类能力：原始 sparse E-PINN 训练、direct FVM/EDFM 数值参考解、POD-MLP 任意时间代理预测。

当前 v11 已扩展为同时求解两个压力分量 `P12/P13`。这里仍然是单相压力物理，不是两相流：没有饱和度、相渗、毛管力或 IMPES；`P12/P13` 是两个压力-like 分量，后处理时相加得到 `Ptotal`。

核心建模方式：

- 基质采用规则 cell-centered FVM 网格，裂缝采用 EDFM 中心线段并作为额外 cell。
- 连接表包含 `T_mm`、`T_mf` 和 `T_ff`，外边界无流通过不添加外边界连接自然实现。
- 默认网格已经加密到 `grid.nx=360`、`grid.ny=150`，并把主裂缝和 5 条次级裂缝切分为 EDFM 段。
- v11 现在只保留与 v3/v10 一致量级的 `Fai/D1/D2` 扩散参数路线。其中 `Fai` 进入有限体积 storage，`D1/D2` 分别组装 `P12/P13` 的连接 transmissibility。
- 连接 transmissibility 为 `T_ij,c = seconds_per_day * D_c,face * A/L`，因此旧版本按秒理解的扩散系数可以和 v11 的 day 时间步保持一致；旧的 `k/mu` 传导率计算分支已经删除。
- 最初的 dense adjacency E-PINN 已改为 sparse graph/message-passing 版本，避免 `N x N` 稠密邻接矩阵限制。`edfm.max_dense_elements` 只控制小网格 dense adjacency 检查，不再限制训练规模。
- E-PINN 输入/输出张量形状为 `[num_cells, 2]`，两个通道对应 `P12/P13` 的归一化压力。
- 网络保留 E-PINN 风格的 anchoring、adaptive ReLU、skip connection、gated updating，并通过 sparse `edge_index` 在 EDFM/FVM 图上聚合邻居信息。
- 单相隐式 Euler residual 对每个压力分量分别计算：

```text
R_i,c = phi_i * c_t * V_i * (p_i,c* - p_i,c^t) / dt
        - sum_j T_ij,c * (p_j,c* - p_i,c*)
loss = mean((R_i,c / row_scale_i,c)^2)
```

- BHP 生产 cell 和连通的裂缝端点 cell 在 residual 计算前被硬覆盖为目标压力，目标压力也按 `P12/P13` 比例拆分。
- `pressure.C13_C12` 控制 `P13/P12` 的默认分量比例；两个压力分量的传导率分别由 `physics.diffusivity_keys` 指向的 `D1/D2` 提供。
- `python main.py train` 训练 sparse E-PINN；`python main.py solve` 用同一套 EDFM/FVM 方程做直接隐式参考解。两者都会写 `outputs/snapshots.npz`，需要通过快照中的 `solver` 字段区分 `sparse_epinn_train` 和 `direct_fvm_edfm`。
- 绘图输出 `field_P12_t*.png`、`field_P13_t*.png` 和 `field_Ptotal_t*.png`，主裂缝剖面也同时显示三个量。

v11 中三条计算路线的定位不同：

- `main.py train` 是 sparse E-PINN 训练路线，输出 `outputs/snapshots.npz`，时间点来自 `time_grid.times_days`。它用于研究压力向量网络能否在 EDFM/FVM 离散图上学会隐式时间推进。
- `main.py solve` 是 direct FVM/EDFM 数值参考路线，也会输出 `outputs/snapshots.npz`，但 `solver` 字段为 `direct_fvm_edfm`。它不是 PINN 训练结果，而是检查离散物理和压力传播趋势的参考解。
- `main_POD.py all` 是 POD-MLP 降阶代理路线，默认先生成 `outputs/pod/direct_snapshots.npz`，再训练 `outputs/pod/pod_basis.npz` 和 `outputs/checkpoints/pod/pod_mlp.pt`，最后输出 `outputs/pod/pod_predictions.npz`。这条路线不改 E-PINN loss，也不加入 FVM residual loss；它是数据驱动的低维代理模型。

POD-MLP 的核心假设是：在固定几何、固定网格、固定 `Fai/D1/D2` 和固定 BHP 制度下，压力场可以由少量 POD 模态近似。默认会排除 BHP hard-constrained cells，只对 free cells 拟合 POD；预测完成后再把 BHP cells 精确覆盖为对应时间的分量 BHP。这样做避免了时间变化的 Dirichlet cell 主导 POD 模态。

POD 输入特征只有两个：

```text
[normalized log time, normalized total BHP]
```

输出是 POD modal coefficients。POD-MLP 不做递推时间步进，也不使用坐标、GNN 或 residual loss；在训练时间区间内给定任意时间，就能一次 forward 预测 `P12/P13/Ptotal` 全场。改变几何、网格、`Fai`、`D1/D2`、BHP 递减制度或 cell ordering 后，通常需要重新生成快照并重新训练 POD-MLP。

v11 还包含区域累产气量和甲烷碳同位素后处理。该后处理只读取 MPa 单位的 `P12/P13` 压力场，按 COMSOL 等价的理想气体公式计算：

- `Q_HF_m3`、`Q_SRV_m3`、`Q_USRV_m3`：HF/SRV/USRV 区域库存亏空量。
- `Q_cum_m3`：三大区域累加的总累产气量。
- `Q1_*` 和 `Q2_*`：分别对应 C12 甲烷和 C13 甲烷分量。
- `delta_prod_permil`：用 `d(Q2_cum)/dt` 和 `d(Q1_cum)/dt` 得到的生产端甲烷碳同位素曲线。

这些区域 Q 是“区域初始库存量 - 区域当前剩余库存量”，不是严格的气源示踪贡献量；因此程序不会把负值裁剪为 0，也不会强制单调。`pod_isotope_delta.png` 只绘制 `delta_prod_permil` 一条曲线，`delta_cum_permil` 和 `delta_remaining_permil` 仍保留在 CSV/NPZ 结果中供数值检查。

该版本适合研究：压力向量网络和 EDFM/FVM residual 是否比坐标型 PINN 更容易捕捉沿主裂缝、次级裂缝和基质之间的压力传播；以及在固定物理设置下，POD-MLP 能否用更低成本复现 direct FVM/EDFM 压力快照，并进一步输出区域气量和同位素后处理曲线。需要特别注意，`solve` 的结果是数值参考解，不是 PINN 训练结果；判断 E-PINN 是否学好时应对比 `train` 和 `solve` 的快照。判断 POD-MLP 是否可靠时，应优先查看 `pod_metrics_by_time.csv` 中的 test error，以及 `pod_vs_reference_gas_metrics.csv` 中的气量/同位素对比误差。

### `pinn_hf_srv_usrv_v12_line_hf_random`

基于 v3 分区 MLP 的线裂缝强形式 PINN 版本。v12 的目标是保留 v3 的“坐标输入 + 自动微分 PDE residual + 分区子网”路线，但把 HF 从 `0.01 m` 厚的二维薄矩形区域改成嵌入 SRV 内部的一维裂缝线。

核心建模方式：

- `config/default.yaml` 中仍保留主裂缝和次级裂缝的薄矩形坐标，但这些矩形只作为提取中心线和裂缝开度的源数据。
- `geometry.hf_representation="line"` 时，HF 只在裂缝中心线上被判定为 `REGION_HF`；原来薄矩形占据的二维空间不再是 HF 面区域，而是 SRV。
- 主裂缝中心线为 `y=75 m` 的水平线，5 条次级裂缝为竖直中心线。
- HF PDE collocation 点直接在线段上采样，不再在 0.01 m 厚矩形内部采样。
- 生产端 Dirichlet 从 v3 的短竖线退化为主裂缝右端点 `(360, 75)`。
- HF-SRV 耦合不再使用薄矩形两侧 offset 的 interface residual，而是采用“线点 + 法向偏移 SRV 点”的压力耦合：

```text
sample x_f on fracture centerline
x_srv = x_f + eps_hf_srv * n
loss_hf_srv = mean((u_HF(x_f,t) - u_SRV(x_srv,t))^2)
```

- SRV-USRV interface 仍保留 v3 的压力连续和法向通量连续损失。
- 主裂缝 link、次级裂缝 link 和 junction coupling 仍保留，但采样点都来自裂缝线。
- 默认每个子网接收 `[x_hat, y_hat, t_hat]` 三维归一化坐标，即 `model.subnet_input_dim=3`；旧的 `[x_local, y_local, x_hat, y_hat, t_hat]` 五维 local-feature 模式仍可通过 `subnet_input_dim=5` 手动启用。
- 默认使用随机采样且不固定 collocation 点：

```yaml
sampler:
  sampling_mode: "random"
  time_sampling_mode: "random"
training:
  fixed_collocation_points: false
  resample_every: 1
  use_lbfgs: false
```

- 默认关闭 LBFGS，使用 Adam。原因是随机 collocation 下每个 epoch 的训练点都会变化，而 LBFGS closure 会在同一外层步内多次评估，默认与这种随机重采样策略不匹配。
- 云图绘制时只对 SRV/USRV 作为二维区域填色，HF 作为黑色线叠加；裂缝线上的压力通过剖面图输出。

v12 适合回答的问题是：如果不想采用 v10/v11 的离散网格 residual，但又不希望 HF 被 0.01 m 厚度造成的二维薄区域和短轴二阶导控制，能否在强形式 PINN 中把 HF 显式降维为裂缝线，并通过随机非固定 collocation 提高训练覆盖。它仍然是强形式 PINN，依然依赖自动微分二阶导和 soft loss 权重调节；与 v10/v11 相比，它不是离散守恒残差方法。

### `pinn_hf_srv_usrv_v13_meshfree_p12_fvm`

v13 是面向单个压力分量 `P12` 的 mesh-free PINN 版本。它保留 v12 的“连续坐标输入 + 随机非固定 collocation + 线裂缝采样”思想，但不再同时求 `P13/Ptotal`；主裂缝和次级裂缝都作为一维线段处理，HF/SRV/USRV 使用不同孔隙度、扩散系数和渗透率参数。

v13 的关键点是：PINN 主模型仍然是 mesh-free 的坐标网络，输入不是网格编号，也不在固定 FVM 网格点上训练；训练时禁止使用 FVM teacher loss，只依赖 PDE residual、生产端 Dirichlet、外边界无流、HF-SRV/SRV-USRV 界面约束和裂缝连通约束。为了改善时变 PINN 的传播稳定性，v13 增加了 causal time weighting、残差自适应 collocation、生产端 hard Dirichlet、一维 HF 切向 PDE residual、解析 erfc 扩散基准场和最大值原理输出约束。

核心建模方式：

- `pressure.components=["P12"]` 且 `model.output_dim=1`，只求一个压力场。
- `physics.parameterization="legacy_diffusion"`，默认使用 `Fai.HF/SRV/USRV` 和 `D.HF/SRV/USRV` 组装强形式 PDE 系数；`permeability_mD` 同时保留，用于物理含义和可选参考模型扩展。
- HF 不再是 `0.01 m` 厚二维区域；HF PDE 点、主裂缝 link 点、次级裂缝 link 点和 junction 点都在线段上随机采样。
- `model.constraint_mode="ic_base_correction"` 时，网络学习相对基准场的修正，避免从零直接学习完整高压场。
- `fvm_reference.enabled=false` 是默认设置；FVM 相关代码只保留为离线对照工具，不会进入训练 loss。

需要注意：v13 判断训练质量时应优先看 `loss_history.csv` 中 PDE、边界、界面和裂缝项是否下降，`diagnostics.csv` 中是否存在负压/非有限压力点，以及 `field_P12_t*.png` 和裂缝剖面是否呈现从生产端沿主裂缝向 SRV/USRV 传播的合理趋势。

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

例如运行 v10：

```powershell
cd D:\CursorProject\pinn_hf_srv_usrv_v10_rdfm_ipinn
python main.py test
python main.py train
python main.py evaluate
python main.py plot
python main.py mesh
```

例如运行 v11：

```powershell
cd D:\CursorProject\pinn_hf_srv_usrv_v11_epinn_edfm_singlephase

# 原始 v11 E-PINN / direct FVM 流程
python main.py test
python main.py train --time-steps 2 --epochs-per-step 2 --grid-nx 8 --grid-ny 6
python main.py solve
python main.py evaluate
python main.py plot
python main.py mesh

# POD-MLP 降阶代理模型与气量/同位素后处理流程
python main_POD.py all
python main_POD.py gas
```

`main_POD.py all` 会自动执行：生成 POD direct FVM 快照、训练 POD basis、训练 POD-MLP、评估、预测、绘制压力图、计算 `Q_HF/Q_SRV/Q_USRV/Q_cum`、绘制 `delta_prod_permil`。当前 `main_POD.py` 顶部 IDE 默认参数是较快的中等规模 smoke 设置；如果要跑完整默认网格和完整 `0~1000 day`，可以改掉顶部 `IDE_*` override，或直接使用模块命令：

```powershell
python -m src.pod_generate_data --config config/default.yaml
python -m src.pod_train --config config/default.yaml
python -m src.pod_evaluate --config config/default.yaml
python -m src.pod_predict --config config/default.yaml --times 50 123.4 437 800 --output-name pod_predictions.npz
python -m src.pod_gas_postprocess --config config/default.yaml --prediction-file outputs/pod/pod_predictions.npz
```

例如运行 v12：

```powershell
cd D:\CursorProject\pinn_hf_srv_usrv_v12_line_hf_random
python main.py test
python main.py train --epochs 3500 --no-resume
python main.py evaluate
python main.py plot
```

例如运行 v13：

```powershell
cd D:\CursorProject\pinn_hf_srv_usrv_v13_meshfree_p12_fvm
python main.py test
python main.py train --epochs 3500 --no-resume
python main.py evaluate
python main.py plot
```

也可以直接调用模块：

```powershell
python -m src.train --config config/default.yaml
python -m src.evaluate --config config/default.yaml --checkpoint outputs/checkpoints/final.pt
python plot_cloud_maps.py
python -m src.plot_loss_history --config config/default.yaml
```

## 训练输出

各子项目训练后通常生成：

```text
outputs/checkpoints/       # 模型 checkpoint
outputs/logs/              # loss_history.csv
outputs/figures/           # 压力云图、loss 曲线、剖面图
outputs/tables/            # 诊断表；v10 solve 会写 fem_solver_history.csv，v11 会写 well_history.csv，v12 会写 section_profiles.csv
outputs/snapshots.npz      # v10 节点压力快照；v11 cell-centered 压力快照；v12 不使用该快照格式作为主输出
outputs/pod/               # v11 POD source snapshots、POD basis、POD predictions 和 gas production archive
outputs/figures/pod/       # v11 POD 误差图、预测压力图、累产气量图、产气速率图和 delta_prod 图
outputs/tables/pod/        # v11 POD 训练/评估表、gas production CSV、summary CSV 和 POD-vs-reference 对比表
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
- v8：HF 短轴局部坐标固定、删除 HF 连通增强、使用空间 LHS、混合时间采样和界面线段最低点数分配的 v3 对照版本。
- v9：基于 v8 的 base-correction 输出结构版本。
- v10：RDFM 低维裂缝 + FEM 离散残差 + I-PINN 训练版本，用于对照强形式 PINN 路线。
- v11：EDFM 显式裂缝 + FVM 离散残差 + E-PINN 压力向量网络的单相流版本，并已扩展 direct FVM 快照驱动的 POD-MLP 降阶代理模型和区域累产气量/甲烷碳同位素后处理。
- v12：基于 v3 的线裂缝强形式 PINN 版本，HF 不再是 0.01 m 厚二维区域，默认随机非固定 collocation。
- v13：mesh-free 单压力 P12 + 线裂缝 + 纯 PDE/边界训练版本，用于检验不使用 FVM teacher 时连续坐标 PINN 的压力传播能力。

如果目标是分析“为什么压力分布不合理”，建议对比：

- v3 与 v7：观察分区网络是否关键。
- v5 与 v3：观察 base-correction 是否改善传播。
- v6 与 v3：观察避免二阶导后训练稳定性是否改善。
- v8 与 v3：观察空间 LHS、混合时间采样、界面线段最低点数分配和动态重采样在无 HF 连通增强条件下是否改善训练稳定性和压力场合理性。
- v9 与 v8：观察学习相对基准场修正是否比直接学习完整压力场更容易形成合理压力传播。
- v10 与 v3/v8/v9：观察把强形式 PDE loss 和界面 soft loss 替换为 RDFM/FEM 离散残差后，压力传播、裂缝连通和训练稳定性是否更合理。
- v11 与 v10：观察压力向量网络和 EDFM/FVM residual 是否比坐标 MLP + RDFM/FEM residual 更容易学习裂缝导流。
- v12 与 v3：观察把 HF 从薄矩形 PDE 区域降维为裂缝线后，是否能减少 0.01 m 厚度方向带来的训练困难。
- v12 与 v10/v11：观察在仍使用强形式 PINN 的前提下，线裂缝采样和随机重采样能否接近离散 residual 方法的裂缝传播效果。
- v13 与 v12：观察加入 causal time weighting、残差自适应采样、hard Dirichlet 和一维 HF PDE 后，mesh-free 坐标 PINN 是否能更稳定地沿一维主裂缝和次级裂缝传播 P12 压降。

如果目标是调参，优先检查：

- `config/default.yaml` 中的采样点数量。
- `loss_weights` 中各损失项权重。
- 是否启用 LBFGS。
- HF 主裂缝和次级裂缝连通损失是否过强或过弱。
- 绘图色阶是否统一，避免误判界面不连续。

如果目标是调 v10，优先检查：

- `mesh.nx`、`mesh.ny` 是否能捕捉主裂缝中心线和生产端节点。
- `time_grid.times_days` 是否足够密，尤其是 `10~500 day` 的中期压降阶段。
- `physics.Fai`、`physics.D1`、`physics.D2` 中 HF/SRV/USRV 的量级是否符合预期。
- `training.epochs_per_step` 是否足够让每个隐式时间步残差收敛。
- `outputs/tables/diagnostics.csv` 中是否存在非有限压力点、负压力点或异常压力范围。

如果目标是调 v11，优先检查：

- `grid.nx`、`grid.ny` 是否足够解析主裂缝和次级裂缝附近的基质压力梯度。
- `time_grid.times_days` 是否足够密，过大的时间步会让直接 FVM 参考解和 E-PINN 训练都呈现较强隐式平滑。
- v11 调参时只检查 `physics.Fai`、`physics.D1`、`physics.D2` 和 `physics.seconds_per_day`。
- `edfm.fracture_tangential_multiplier` 默认应为 `1.0`；legacy 模式已经通过 `D_HF=1.0` 和裂缝开度体现高导流，不应再无意中额外放大 1000 倍。
- `training.fracture_residual_weight` 和 `training.fracture_flux_weight` 是否过强或过弱。
- 当前 `outputs/snapshots.npz` 来自 `train` 还是 `solve`。若 `solver=direct_fvm_edfm`，那是直接数值参考解，不是 PINN 训练结果。
- `loss_history.csv` 中 `P12/P13` 的 residual 是否同时下降，避免只让大尺度的 `P12` 看起来收敛。
- 若使用 POD-MLP，先确认 `outputs/pod/direct_snapshots.npz` 来自当前几何、网格、BHP 和 `Fai/D1/D2`；这些设置变化后必须重新生成快照并重训。
- `pod_basis.npz` 中的 `selected_rank` 和 `retained_energy` 是否足够；若 test error 偏大，优先加密 POD snapshot 时间点或提高 `max_rank/fixed_rank`，再增加 MLP epochs。
- `pod_metrics_by_time.csv` 中应分开查看 `pod_projection` 和 `pod_mlp`：前者代表 POD 截断误差，后者还包含 MLP 系数预测误差。
- 气量后处理只使用 MPa 物理压力场，不使用归一化压力或 POD 系数；如果 `Q_HF/Q_SRV/Q_USRV` 出现负值或非单调，应先判断是否为区域库存补给效应，而不是直接裁剪。
- `pod_isotope_delta.png` 只绘制 `delta_prod_permil`；`delta_cum_permil` 和 `delta_remaining_permil` 在 `pod_gas_production.csv` 中保留用于诊断。

如果目标是调 v12，优先检查：

- `geometry.hf_representation` 是否保持为 `"line"`，确认 HF 没有退回薄矩形面区域。
- `model.subnet_input_dim` 当前默认为 `3`，即每个子网只接收 `[x_hat,y_hat,t_hat]`；若要恢复局部坐标特征，可手动改为 `5`。
- `training.fixed_collocation_points=false` 和 `training.resample_every=1` 是否符合随机非固定 collocation 设计。
- `sampler.eps_hf_srv` 是否足够把裂缝线点偏移到 SRV 中，但又不要大到跨过局部几何细节。
- `loss_weights.interface_pressure`、`hf_main_link`、`hf_secondary_link` 和 `hf_junction` 是否平衡；线裂缝版本仍然依赖这些 soft loss 引导裂缝连通。

如果目标是调 v13，优先检查：

- `pressure.components` 是否保持为 `["P12"]`，`model.output_dim` 是否保持为 `1`。
- `loss_weights` 中是否不存在 `fvm_teacher`；v13 纯物理训练禁止使用该项。
- `training.causal_time_weighting` 是否开启；时变问题早期传播不稳时，优先调 `bins` 和 `epsilon`。
- `training.adaptive_resampling` 是否开启；主裂缝或界面附近残差高时，优先增加 `candidate_multiplier` 和 `keep_hf/keep_srv/keep_usrv`。
- `model.constraint_mode` 是否为 `ic_base_correction`；该模式让网络学习相对基准场的修正，比直接学习完整压力场更稳。
- `model.hard_dirichlet` 是否开启；生产端压力不准时，优先检查 `radius_m` 和 `power`。
- `model.analytic_base` 是否开启；该项用扩散方程的 erfc 型解析近似给出初始传播形态，不使用 FVM 数值解。
- `model.enforce_maximum_principle` 是否开启；无源压力扩散问题中压力不应高于初始压力。

## 注意事项

- 本仓库是实验研究代码集合，不是最终工业级求解器。
- 多个版本之间存在大量相似代码，这是为了保留实验可复现性，而不是追求最少重复。
- 不同版本 checkpoint 通常不兼容，尤其是网络输出维度或网络结构发生变化时。
- 若需要继续训练某一版本，应使用同一版本生成的 checkpoint。
- 若 GitHub 上缺少 `outputs/`，这是预期行为，不代表训练结果丢失。
