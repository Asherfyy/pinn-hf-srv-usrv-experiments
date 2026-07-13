# pinn_hf_srv_usrv_v7_v5based_single_mlp

本项目基于 `pinn_hf_srv_usrv_v5_base_correction` 创建，保留 v5 的几何范围、采样方式、PDE/边界/界面损失、裂缝连通损失、绘图脚本和训练入口。

核心变化：v7 不再为 HF/SRV/USRV 分别建立子网络，而是使用一个共享 MLP 表达整个区域的压力场修正。

```text
input  = [x_hat, y_hat, t_hat]
output = [u12, u13]
u      = u_base(x, y, t) + envelope(t) * correction_NN(x_hat, y_hat, t_hat)
```

PDE、界面通量和材料系数仍然按 HF/SRV/USRV 分区分别计算，但神经网络参数是全域共享的。这样便于和 v5 的分区 MLP 结果做对照，观察“分区网络”本身对压力连续性、裂缝传播和训练稳定性的影响。

默认 `model.base_checkpoint: null`，因此 `u_base` 是初始均匀压力场。如果后续指定 v7 自己训练得到的 checkpoint，`u_base` 会取该冻结模型在 `max(t - base_time_lag_days, t_min)` 的压力场。

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
