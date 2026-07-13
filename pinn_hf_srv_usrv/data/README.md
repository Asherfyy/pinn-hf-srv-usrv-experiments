# data 目录说明

本目录用于放置可选的 COMSOL 导出数据和外部系数表。项目第一版不依赖 COMSOL API，
没有这些文件也可以完成基础 PINN 训练。

- `dirichlet_boundary.csv`：可选 Dirichlet 边界点云，至少包含 `x,y` 两列。当前默认仍使用解析线段。
- `comsol_snapshots.csv`：可选 COMSOL 快照，格式为 `x,y,t,Pg1,Pg2,region`，其中 `region` 可省略。
- `coefficients.csv`：可选外部系数记录。第一版主要从 `config/default.yaml` 读取系数。
