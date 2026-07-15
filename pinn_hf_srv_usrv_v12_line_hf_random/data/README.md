# data 目录说明

本项目是独立的简化 PINN 版本，不依赖 COMSOL API。`data/` 目录预留给后续放置
外部基准数据、剖面采样点或同一简化 PDE 的数值解。当前训练、评价和绘图只依赖
`config/default.yaml` 中的解析矩形几何和物理参数。
