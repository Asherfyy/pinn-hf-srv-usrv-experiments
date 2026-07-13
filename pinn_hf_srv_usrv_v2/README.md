# pinn_hf_srv_usrv_v2

这是一个独立的、简化版 HF/SRV/USRV 分区 PINN 项目。它不覆盖也不依赖原
`pinn_hf_srv_usrv` 项目的代码，只复制当前实际使用的几何参数和真实物理系数。

## 项目目标

使用纯 PyTorch 和 CPU，在二维 HF/SRV/USRV 分区储层几何中同时求解两个无量纲压力
变量 `u12/u13`，并还原为物理压力 `P12/P13`，单位 MPa。

## 当前默认 PINN 设置

v2 是简化输入的单网络 PINN。它使用一个全域 MLP 表达 HF、SRV、USRV 三个区域中的压力场，几何只用于采样、分区物性选择、界面 loss 和绘图，不作为 one-hot 或距离特征输入网络。

### 网络输入与输出

网络输入只有 3 维归一化坐标：

```text
[x_hat, y_hat, t_hat]
```

其中：

- `x_hat` 由 `x = 0~360 m` 归一化得到。
- `y_hat` 由 `y = 0~150 m` 归一化得到。
- `t_hat` 由 `t = 0~1000 day` 归一化得到。

v2 明确不使用以下输入：

- HF/SRV/USRV 区域 one-hot。
- 到裂缝、外边界、生产端边界的距离特征。
- SDF/ADF 几何嵌入。

网络输出为：

```text
[u12, u13]
```

`u12/u13` 分别对应 `P12/P13` 的无量纲压力。当前压力归一化方式为 `component_affine`，使初始状态下两个分量都等于 `1`，生产端长期目标接近 `0`。

### MLP 结构

默认配置为：

```yaml
input_dim: 3
output_dim: 2
hidden_layers: 4
hidden_units: 128
activation: "tanh"
```

代码中 `MLP` 使用全连接层堆叠，权重采用 Xavier 初始化。当前支持的激活函数是 `tanh` 和 `silu`。默认使用 `tanh`，因为当前 PDE loss 需要计算二阶导，平滑激活函数更适合强形式扩散 PDE。

### 初始条件硬约束

v2 只实现一种约束模式：

```yaml
constraint_mode: "ic_hard"
```

模型输出形式为：

```text
u(x,y,t) = 1 + t_hat * N_theta(x_hat,y_hat,t_hat)
```

其中 `N_theta` 是 MLP 原始输出。由于 `t=0` 时 `t_hat=0`，因此：

```text
u12(x,y,0) = 1
u13(x,y,0) = 1
```

也就是说初始条件被结构严格满足，不需要再额外采样初始条件点，也没有单独的 initial loss。

### 生产端边界条件

v2 不使用生产端 Dirichlet 硬约束。生产端边界通过 soft loss 约束：

```text
loss_dirichlet = MSE(u_pred - u_D(t))
```

生产端目标压力由以下参数给出：

```yaml
P_t0: 25.0
P_out: 3.0
C13_C12: 0.010900084
decay_rate: 0.018
```

代码会先构造随时间衰减的总压力，再按 `C13_C12` 拆分为 `P12/P13`，最后映射为无量纲目标 `u12/u13`。

### PDE 形式

v2 使用有效扩散形式，不再保留原始 `Fai*u_t-D*Laplace(u)` 写法。对每个区域 `r` 和每个变量 `i`，PDE 为：

```text
du_i/dt_hat
- kappa_x_i,r * d2u_i/dx_hat2
- kappa_y_i,r * d2u_i/dy_hat2 = 0
```

其中：

```text
K_i,r = D_i,r / Fai_r
kappa_x_i,r = K_i,r * T / Lx^2
kappa_y_i,r = alpha_y_i * K_i,r * T / Ly^2
```

默认物性参数为：

```text
HF:   Fai=0.1,  D1=1.0,    D2=1.0
SRV:  Fai=0.05, D1=1.0e-7, D2=9.95e-8
USRV: Fai=0.05, D1=1.0e-9, D2=9.9e-10
```

因此 HF 的有效扩散能力远高于 SRV 和 USRV，这也是该问题训练困难的主要来源之一。

### 损失函数

v2 的总损失为：

```text
loss_total =
  w_pde       * loss_pde
+ w_dirichlet * loss_dirichlet
+ w_neumann   * loss_neumann
+ w_if_p      * loss_interface_pressure
+ w_if_q      * loss_interface_flux
```

默认权重为：

```yaml
pde: 1.0
dirichlet: 10.0
neumann: 1.0
interface_pressure: 1.0
interface_flux: 0.1
```

各项含义：

- `loss_pde`：HF/SRV/USRV 内部 PDE 残差。代码分别计算 `u12/u13` 在三个区域内的 6 个 PDE 分量，并用残差尺度归一化后取平均。
- `loss_dirichlet`：生产端边界压力 soft loss。
- `loss_neumann`：外边界无流 soft loss。当前对归一化坐标的法向导数施加零约束。
- `loss_interface_pressure`：HF-SRV 与 SRV-USRV 界面两侧压力连续。
- `loss_interface_flux`：HF-SRV 与 SRV-USRV 界面两侧有效扩散通量连续。

### 界面连续处理

界面点先按几何线段长度采样，随后在 `physics.py` 中沿界面法向做两侧偏移：

```text
x_minus = x_interface - eps * n
x_plus  = x_interface + eps * n
```

然后用区域判断过滤，只保留真正跨越目标材料界面的点。默认偏移距离为：

```yaml
eps_hf_srv: 0.001
eps_srv_usrv: 0.05
```

压力连续 loss 使用：

```text
u_minus - u_plus
```

通量连续 loss 使用与 PDE 一致的有效扩散通量：

```text
kappa_x * u_x * n_x + kappa_y * u_y * n_y
```

并按两侧残差尺度归一化，避免 HF 的极大 `kappa` 直接支配优化。

### 采样设置

默认采样数量为：

```yaml
n_pde_hf: 5000
n_pde_srv: 5000
n_pde_usrv: 5000
n_near_hf_srv: 5000
n_near_srv_usrv: 5000
n_dirichlet: 5000
n_neumann: 5000
n_interface_hf_srv: 5000
n_interface_srv_usrv: 5000
```

采样逻辑：

- HF PDE 点直接在主裂缝和 5 条次级裂缝矩形内采样。
- SRV PDE 点在 SRV 背景区域采样，并排除 HF。
- USRV PDE 点在全域采样，并排除 SRV/HF。
- `n_near_hf_srv` 会在 HF 外侧窄带中采样属于 SRV 的 PDE 加密点，并合并进 SRV PDE 点。
- `n_near_srv_usrv` 会在 SRV-USRV 内部界面附近采样窄带 PDE 点，一半并入 SRV，一半并入 USRV。

因此默认每套 collocation 中，实际 PDE 点数量约为：

```text
HF:   5000
SRV:  5000 + 5000 + 2500 = 12500
USRV: 5000 + 2500 = 7500
```

默认训练设置为：

```yaml
fixed_collocation_points: true
resample_every: 500
```

也就是训练开始时采样一次并固定复用。只有把 `fixed_collocation_points` 改为 `false` 后，才会按 `resample_every` 重新采样。

### 时间采样

v2 只支持：

```yaml
time_strategy: "log1p_uniform"
```

采样时先在 `log1p(t)` 空间随机均匀采样，再通过 `expm1` 映射回物理时间 `0~1000 day`。这样可以增强早期和中期压力衰减阶段的采样密度，同时仍覆盖长时间范围。

### 训练设置

默认训练参数：

```yaml
device: "cpu"
dtype: "float64"
epochs: 3500
learning_rate: 1.0e-3
grad_clip_norm: 10.0
```

训练主流程使用 Adam 优化器。配置文件中保留了 `use_lbfgs` 字段，但当前 v2 的 `src/train.py` 主流程实际没有执行 LBFGS 阶段。

## 与原复杂项目的区别

本版本保留简化输入结构，同时加入了可配置的界面连续训练 loss：

- 不使用距离输入特征；
- 不使用区域 one-hot 作为网络输入；
- 不使用 SDF/ADF；
- 不使用 Dirichlet hard constraint；
- 使用 HF-SRV 和 SRV-USRV 两侧偏移点的压力连续 loss；
- 使用有效扩散 `kappa_x/kappa_y` 对应的界面通量连续 loss；
- 网络输入只有 `x_hat/y_hat/t_hat`；
- 初始条件使用结构硬约束 `u=1+t_hat*N_theta`；
- 生产边界和 Neumann 外边界使用 soft loss；
- PDE 使用 `D/Fai` 有效扩散系数。

几何仍然用于分区采样、PDE 系数选择、区域统计、绘图 mask 和近界面 PDE 加密点生成。

## 物理形式

原始线性扩散形式为：

```text
Fai_r dP_i/dt - D_i,r Laplace(P_i) = 0
```

定义有效扩散系数：

```text
K_i,r = D_i,r / Fai_r
```

归一化坐标下使用：

```text
du_i/dt_hat
- kappa_x_i,r d2u_i/dx_hat2
- kappa_y_i,r d2u_i/dy_hat2 = 0
```

其中：

```text
kappa_x_i,r = K_i,r * T / Lx^2
kappa_y_i,r = alpha_y_i * K_i,r * T / Ly^2
```

本项目继续使用真实极端物理系数：

- HF: `Fai=0.1`, `D1=1.0`, `D2=1.0`
- SRV: `Fai=0.05`, `D1=1.0e-7`, `D2=9.95e-8`
- USRV: `Fai=0.05`, `D1=1.0e-9`, `D2=9.9e-10`

## 压力无量纲化

P12 和 P13 使用独立仿射尺度，使初始状态下两个变量都为 1，长期出口状态都趋近 0。
所有压力转换统一在 `src/utils.py` 中实现。

## 局限

该模型通过 soft loss 约束界面压力和有效扩散通量连续，但它仍是 PINN 近似解，不像
有限体积/有限元离散那样天然守恒。结果判断应结合剖面图，并最好和同一简化 PDE 的
传统数值基准对比。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行测试

```bash
pytest tests
```

## 训练

```bash
python -m src.train --config config/default.yaml
```

短跑冒烟：

```bash
python -m src.train --config config/default.yaml --epochs 5
```

继续训练：

```bash
python -m src.train --config config/default.yaml --resume outputs/checkpoints/epoch_3000.pt
```

## 评价

```bash
python -m src.evaluate --config config/default.yaml --checkpoint outputs/checkpoints/final.pt
```

诊断表输出到：

```text
outputs/tables/diagnostics.csv
```

## 绘图

云图：

```bash
python -m src.plot_fields --config config/default.yaml --checkpoint outputs/checkpoints/final.pt
```

剖面：

```bash
python -m src.plot_sections --config config/default.yaml --checkpoint outputs/checkpoints/final.pt
```

loss 曲线：

```bash
python plot_loss_history.py
```

## 统一入口

```bash
python main.py test
python main.py train
python main.py evaluate
python main.py plot
python main.py all
```

IDE 直接运行 `main.py` 时默认执行训练，不绘图也不保存图片。
