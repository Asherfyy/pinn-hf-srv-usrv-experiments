# pinn_hf_srv_usrv_v8_no_hf_local_coord

这是一个基于 `pinn_hf_srv_usrv_v3_partition_mlp` 新建的独立对照项目。v8 保留 v3 的分区 MLP、几何范围、物理参数、PDE、边界条件和 HF-SRV/SRV-USRV 界面连续损失，但同时删除两类 v3 特殊处理：

- 删除 HF 极薄裂缝局部坐标处理。
- 删除 HF 主裂缝/细裂缝连通增强 loss。

## 与 v3 的核心差异

### 1. 删除 HF 极薄裂缝局部坐标处理

v3 对 HF 区域做了特殊处理：

- 水平主裂缝保留长度方向 `x_local`，固定厚度方向 `y_local=0.5`。
- 竖直细裂缝保留长度方向 `y_local`，固定厚度方向 `x_local=0.5`。

v8 删除上述处理。HF 主裂缝和细裂缝都直接使用完整矩形局部坐标：

```text
x_local = (x - rect.x_min) / rect.width
y_local = (y - rect.y_min) / rect.height
```

因此：

- 水平主裂缝的厚度方向 `y_local` 会在 `[0,1]` 内变化。
- 竖直细裂缝的厚度方向 `x_local` 会在 `[0,1]` 内变化。
- HF 子网可以表达裂缝厚度方向上的压力变化。
- 代价是 0.01 m 厚度方向会通过局部坐标被直接放大，强形式 PDE 中的二阶导项可能更容易变大。

### 2. 删除 HF 连通增强 loss

v3 中为了模拟高导流裂缝快速等压，加入了三类额外损失：

- `hf_main_link`：主裂缝中心线拉向生产端目标压力。
- `hf_secondary_link`：细裂缝中心线与其主裂缝交汇点同压。
- `hf_junction`：主裂缝和细裂缝交汇处强连通。

v8 已删除这些采样项、loss 权重、总损失项、训练日志项和评估诊断项。也就是说，v8 中 HF 压力传播只依赖：

- HF/SRV/USRV 内部 PDE 残差。
- 生产端 Dirichlet soft loss。
- 外边界 Neumann soft loss。
- HF-SRV 与 SRV-USRV 的压力连续和法向通量连续。
- 压力范围 soft penalty。

## 当前默认 PINN 设置

- 网络结构：`HF`、`SRV`、`USRV` 三个区域分别使用独立 MLP。
- 公共输入：外部输入物理坐标 `x,y,t`，内部归一化为 `z=[x_hat,y_hat,t_hat]`。
- 子网输入：默认 `subnet_input_dim=5`，即 `[x_local, y_local, x_hat, y_hat, t_hat]`。
- 输出变量：`u12` 和 `u13` 两个无量纲压力分量。
- 子网规模：HF 为 4 个隐藏层、每层 96 个神经元；SRV 和 USRV 为 5 个隐藏层、每层 128 个神经元。
- 激活函数：默认 `tanh`。
- 初始条件硬约束：`u = 1 + (1 - exp(-decay_rate*t)) * N_region(x,y,t)`，保证 `t=0` 时严格满足 `u=1`。
- PDE 形式：强形式有效扩散方程 `u_t - kappa_x*u_xx - kappa_y*u_yy = 0`。
- 边界条件：生产端 Dirichlet 为 soft loss，外边界 Neumann 为 soft loss。
- 界面条件：HF-SRV 和 SRV-USRV 均施加压力连续和法向通量连续 loss。
- 训练方式：默认 `float64`、CPU、固定 collocation 点，并启用 LBFGS。

## 建议用途

v8 是一个更纯粹的反向对照实验：它同时拿掉 v3 中对 HF 的两个“人为帮助”机制。与 v3 对比时重点看：

- `loss_pde` 是否明显升高或震荡。
- HF 区域压力是否出现厚度方向振荡。
- 生产端压力影响是否更难传播到主裂缝远端。
- 细裂缝是否失去快速等压特征。
- HF-SRV 界面压力连续和通量连续是否恶化。

## 运行

```bash
python main.py test
python main.py train
python main.py evaluate
python main.py plot
```

也可以分别运行：

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
