# 基于 PINN 的 HF/SRV/USRV 分区渗流 PDE 求解

本项目实现一个纯 PyTorch、CPU 训练的 PINN 框架，用于在二维复杂几何储层中求解两个无量纲压力变量 `u12,u13`，并还原为 `P12/P13` MPa。几何来自总矩形储层、SRV 改造区、水平主裂缝和 5 条竖向次级裂缝。

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
