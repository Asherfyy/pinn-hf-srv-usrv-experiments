# 基于 PINN 的 HF/SRV/USRV 分区渗流 PDE 求解

本项目实现一个纯 PyTorch、CPU 训练的 PINN 框架，用于在二维复杂几何储层中求解两个无量纲压力变量 `u12,u13`，并还原为 `P12/P13` MPa。几何来自总矩形储层、SRV 改造区、水平主裂缝和 5 条竖向次级裂缝。

## 当前默认 PINN 设置

当前 `pinn_hf_srv_usrv` 是原始版单网络 PINN。它不是 v3/v5 那种分区 MLP，而是用一个全域 MLP 同时表达 HF、SRV 和 USRV 中的压力场。

### 网络输入与输出

网络输入维度为 `9`：

```text
[x_hat, y_hat, t_hat, region_HF, region_SRV, region_USRV, d_outer, d_HF, d_Dirichlet]
```

含义如下：

- `x_hat, y_hat, t_hat`：物理坐标 `x,y,t` 归一化后的坐标。
- `region_HF, region_SRV, region_USRV`：当前点所属区域的 one-hot 编码，区域优先级为 `HF > SRV > USRV`。
- `d_outer`：到全域外边界的归一化距离特征。
- `d_HF`：到最近 HF 裂缝边界的归一化距离特征。
- `d_Dirichlet`：到生产端 Dirichlet 边界线段的归一化距离特征。

网络输出维度为 `2`：

```text
[u12, u13]
```

其中 `u12/u13` 是 `P12/P13` 两个压力分量的无量纲形式。后处理阶段通过压力归一化参数还原为 `P12/P13`，并可相加得到总压力 `Ptotal`。

### MLP 结构

默认配置为：

```yaml
input_dim: 9
output_dim: 2
hidden_layers: 6
hidden_units: 64
activation: "tanh"
```

也就是 6 层隐藏层、每层 64 个神经元的全连接网络。权重使用 Xavier 初始化。当前代码支持 `tanh`、`silu` 和 `relu`，但二阶 PDE 残差需要稳定计算二阶导，默认使用更平滑的 `tanh`。

### 硬约束形式

默认约束模式为：

```yaml
constraint_mode: "dirichlet_hard"
```

模型输出不是直接等于神经网络原始输出，而是：

```text
u(x,y,t) = u_ref(x,y,t) + B_D(x,y) * N_theta(x,y,t)
```

其中：

- `N_theta` 是 MLP 原始输出。
- `B_D(x,y)` 是生产端 Dirichlet 边界的 ADF 距离因子。
- 在生产端 Dirichlet 线段上，`B_D=0`，因此 `u=u_ref=g_D(t)`，生产端压力边界被结构精确满足。
- 当前 `dirichlet_adf_type="tanh"`，`dirichlet_adf_length_m=1.0`，表示网络扰动项在离开生产端约数米后恢复自由度。

默认参考场为：

```yaml
hard_constraint_reference: "local_initial"
```

也就是生产端附近使用低压边界参考值，远离生产端逐渐回到初始高压场。这样可以避免硬约束参考场把整个区域都拉向生产端低压。

代码还保留两个可切换模式：

```text
direct:  u = N_theta
ic_hard: u = u0(x,y) + t_hat * N_theta
```

`direct` 表示所有边界/初始条件都作为 soft loss；`ic_hard` 表示初始条件硬满足，生产端 Dirichlet 通过 soft loss 约束。

### PDE 设置

当前求解的是两变量强形式扩散 PDE。对 `u12/u13` 分别计算：

```text
Fai_r * u_t - D_x,r * u_xx - D_y,r * u_yy = 0
```

其中 `r` 表示 HF/SRV/USRV 分区。代码在归一化坐标下求导，并根据配置中的物理尺度把 COMSOL 风格参数转换成无量纲 PDE 系数。

默认启用：

```yaml
use_physical_scaling: true
pde_time_scale_days: 1.0
```

默认物性量级为：

```text
HF:   Fai=0.1,  D1=D2=1.0
SRV:  Fai=0.05, D1=1.0e-7, D2=9.95e-8
USRV: Fai=0.05, D1=1.0e-9, D2=9.9e-10
```

这意味着 HF 的导流/扩散能力远高于 SRV 和 USRV，也是训练困难的主要来源之一。

### 损失函数

总损失由以下部分组成：

```text
loss_total =
  w_pde       * loss_pde
+ w_initial   * loss_ic
+ w_dirichlet * loss_dirichlet
+ w_neumann   * loss_neumann
+ w_if_p      * loss_interface_pressure
+ w_if_q      * loss_interface_flux
+ w_data      * loss_data
```

默认权重为：

```yaml
pde: 1.0
initial: 1.0
dirichlet: 5.0
neumann: 5.0
interface_pressure: 5
interface_flux: 2
data: 0.0
```

说明：

- `loss_pde`：HF/SRV/USRV 内部 PDE 残差。
- `loss_ic`：`t=0` 初始压力场约束。
- `loss_dirichlet`：生产端压力边界误差；在 `dirichlet_hard` 下主要作为诊断项。
- `loss_neumann`：外边界无流条件。
- `loss_interface_pressure`：HF-SRV 与 SRV-USRV 界面压力连续。
- `loss_interface_flux`：HF-SRV 与 SRV-USRV 界面法向通量连续。
- `loss_data`：可选 COMSOL 快照监督项，默认关闭。

### PDE 残差归一化

默认开启：

```yaml
pde_residual_normalization:
  enabled: true
```

代码会把 `u12/u13` 在 `HF/SRV/USRV` 中的 6 个 PDE 分量拆开计算，并分别按固定尺度归一化。这样做是为了避免 HF 区域巨大扩散系数直接主导总 loss，使 SRV/USRV 的 PDE 约束失去作用。

### 采样设置

每个 epoch 都重新随机采样训练点。默认数量为：

```yaml
n_pde_hf: 3000
n_pde_srv: 3000
n_pde_usrv: 3000
n_initial_hf: 2000
n_initial_srv: 2000
n_initial_usrv: 2000
n_dirichlet: 2000
n_neumann: 2000
n_interface_hf_srv: 3000
n_interface_srv_usrv: 3000
```

采样方式：

- HF 点直接在主裂缝和 5 条次级裂缝矩形内采样，避免 0.01 m 极薄裂缝在全域随机采样中几乎采不到。
- SRV 点在 SRV 背景矩形内采样，并排除 HF。
- USRV 点在全域内采样，并排除 SRV/HF。
- HF-SRV、SRV-USRV 界面点按线段长度分配采样数。
- 界面 loss 中会沿法向做 `eps_hf_srv` 或 `eps_srv_usrv` 偏移，分别取界面两侧点计算压力跳跃和通量跳跃。

时间采样默认：

```yaml
t_min: 0.0
t_max: 1000.0
time_strategy: "log1p_uniform"
```

即在 `log1p(t)` 空间均匀采样，再映射回物理时间，用于增强早期和中期压力衰减阶段的采样覆盖。

### 训练设置

默认训练参数为：

```yaml
device: "cpu"
dtype: "float64"
epochs: 1000
learning_rate: 1.0e-3
grad_clip_norm: 10.0
use_lbfgs: false
```

训练流程使用 Adam 优化器，每个 epoch 重新采样一批 collocation points。LBFGS 接口保留，但默认关闭，因为在 CPU 和二阶导 PDE 下计算成本较高。

## PDE 模型

第一版求解两变量扩散型简化 PDE：

```text
A12,r du12/dt_hat - D12,r (d2u12/dx_hat2 + alpha_y,12 d2u12/dy_hat2) = 0
A13,r du13/dt_hat - D13,r (d2u13/dx_hat2 + alpha_y,13 d2u13/dy_hat2) = 0
```

其中 `r` 表示 HF/SRV/USRV 分区。当前 `D` 是归一化坐标上的无量纲有效扩散系数，后续可以根据 COMSOL 物理系数、长度尺度和时间尺度换算后替换为 COMSOL 冻结系数。本版不复现 COMSOL 的 Pg1/Pg2/Pw 三变量非线性耦合，也不调用 COMSOL API。

## 几何说明

区域优先级固定为：

1. 点在任意裂缝矩形内，属于 HF。
2. 否则点在 SRV 背景矩形内，属于 SRV。
3. 否则点在总计算区域内，属于 USRV。
4. 否则属于区域外。

模型输入包含归一化坐标、区域 one-hot、到外边界距离、到 HF 距离和到 Dirichlet 边界距离。Dirichlet 边界采用 hard constraint：

```text
u_i(x,y,t) = g_i(t) + B_D(x,y) N_i(x,y,t)
```

因此在 Dirichlet 线段上 `B_D=0`，边界值由结构精确满足。

## CPU 训练

训练入口会设置：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")
torch.set_num_threads(cpu_threads)
```

项目中不使用 `.cuda()`，也不自动选择 GPU。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行训练

```bash
python -m src.train --config config/default.yaml
```

调试时可短跑：

```bash
python -m src.train --config config/default.yaml --epochs 1
```

## IDE 一键运行

打开项目根目录下的 `main.py`，在 IDE 中直接点击运行即可。默认 `quick` 模式会依次执行测试、1 个 epoch 训练、评价和云图绘制。

也可以在命令行中指定模式：

```bash
python main.py quick
python main.py test
python main.py train
python main.py evaluate
python main.py plot_fields
```

## 运行评价

```bash
python -m src.evaluate --config config/default.yaml --checkpoint outputs/checkpoints/final.pt
```

## 绘制云图

```bash
python -m src.plot_fields --config config/default.yaml --checkpoint outputs/checkpoints/final.pt
```

## 运行测试

```bash
pytest tests
```

## 可选 COMSOL 数据

`data/comsol_snapshots.csv` 是可选文件。若存在有效数据，评价脚本会自动计算 PINN 与 COMSOL 的相对 L2 误差。格式如下：

```text
x,y,t,Pg1,Pg2,region
```

其中 `region` 可省略，程序会用解析几何自动补充 HF/SRV/USRV 分区。
