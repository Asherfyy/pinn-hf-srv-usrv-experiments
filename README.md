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

该版本用于保存最早的建模思路和代码基础。

### `pinn_hf_srv_usrv_v2`

独立重写版本。

主要作用：

- 在不覆盖原始项目的前提下，重新组织代码结构。
- 增加更清晰的训练、评估和绘图入口。
- 使用更规范的 `src/`、`tests/`、`config/` 结构。

该版本是后续多个实验版本的结构基础之一。

### `pinn_hf_srv_usrv_v3_partition_mlp`

分区 MLP 版本。

核心变化：

- 为 HF、SRV、USRV 分别建立独立子网络。
- 每个区域使用自己的 MLP 表达压力场。
- 保留统一的物理坐标输入，但在模型内部按区域选择子网络。
- 对 HF 极薄裂缝引入局部坐标处理，避免厚度方向极小尺度导致二阶导异常放大。
- 增加主裂缝和次级裂缝连通损失，使裂缝内部更接近高导流通道。

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

如果目标是分析“为什么压力分布不合理”，建议对比：

- v3 与 v7：观察分区网络是否关键。
- v5 与 v3：观察 base-correction 是否改善传播。
- v6 与 v3：观察避免二阶导后训练稳定性是否改善。

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
