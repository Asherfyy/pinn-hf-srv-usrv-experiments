# pinn_hf_srv_usrv_v2

这是一个独立的、简化版 HF/SRV/USRV 分区 PINN 项目。它不覆盖也不依赖原
`pinn_hf_srv_usrv` 项目的代码，只复制当前实际使用的几何参数和真实物理系数。

## 项目目标

使用纯 PyTorch 和 CPU，在二维 HF/SRV/USRV 分区储层几何中同时求解两个无量纲压力
变量 `u12/u13`，并还原为物理压力 `P12/P13`，单位 MPa。

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
