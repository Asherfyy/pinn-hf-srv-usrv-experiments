# POD-MLP 降阶代理模型中文详解

## 1. 这条路线解决什么问题

v11 的主线是 E-PINN/EDFM/FVM：用神经网络从上一时刻压力向量预测下一时刻压力向量，并用 EDFM/FVM 离散残差训练。  
POD-MLP 是在这条主线旁边增加的一条降阶代理模型路线。它的目标不是重新定义物理方程，而是在固定几何、固定网格、固定裂缝、固定物性参数和固定井底压力制度下，用更低成本快速复现 direct FVM/EDFM 生成的压力快照。

因此需要明确：

- POD-MLP 不是 PINN。
- POD-MLP 不计算 PDE 自动微分残差。
- POD-MLP 不在训练时强制局部守恒。
- POD-MLP 依赖已有压力快照，默认来源是 direct FVM/EDFM 数值解。
- POD-MLP 学到的是“时间/BHP 到低维压力模态系数”的映射。

它更接近一个数据驱动的降阶代理模型：先用高维数值快照提取主要变化模式，再训练一个小网络预测这些模式随时间如何变化。

## 2. 总体流程

完整流程分五步：

1. 使用 direct FVM/EDFM 生成一组压力快照。
2. 把每个时刻的 `P12/P13` 压力场展平成高维向量。
3. 对训练快照做 POD 分解，得到少量正交模态。
4. 把每个快照投影到 POD 模态上，得到对应的 POD 系数。
5. 训练 MLP，用时间特征和 BHP 特征预测 POD 系数。

预测新时刻时：

```text
时间 t、BHP(t)
    -> MLP 预测 POD 系数
    -> POD 基重构高维压力向量
    -> 对 BHP 控制 cell 做 hard 覆盖
    -> 得到 P12/P13/Ptotal 压力场
```

这条路线不做递推时间步进。也就是说，预测 `500 day` 时不需要先预测 `1 day`、`10 day`、`100 day`。只要 `500 day` 落在训练时间范围内，就可以直接 forward 得到压力场。

## 3. 为什么要用 POD

原始压力场维度很高。例如 `360 x 150` 的基质网格已经有 `54000` 个 matrix cell，再加 EDFM 裂缝 cell，并且每个 cell 有 `P12/P13` 两个分量。直接让 MLP 输出完整压力向量，输出维度会非常大，训练和泛化都不稳定。

POD 的作用是把高维压力场写成少量模态的线性组合：

```text
p(t) ≈ p_mean + a1(t) * phi1 + a2(t) * phi2 + ... + ar(t) * phir
```

其中：

- `p(t)` 是某个时刻的完整压力向量。
- `p_mean` 是训练快照的平均压力场。
- `phi1...phir` 是 POD 模态。
- `a1...ar` 是该时刻的 POD 系数。
- `r` 是保留阶数，通常远小于原始自由度数量。

这样 MLP 不需要输出几万维压力场，只需要输出几十个 POD 系数。压力场的大尺度空间结构由 POD 模态承载，MLP 只学习时间演化。

## 4. 数据来源

POD 训练数据默认来自：

```text
outputs/pod/direct_snapshots.npz
```

这个文件由 direct FVM/EDFM 求解器生成，和 v11 的 EDFM 网格、连接、Fai/D1/D2、井控条件一致。

生成命令：

```powershell
python -m src.pod_generate_data --config config/default.yaml
```

快速 smoke 数据集可以用更小网格和更短时间：

```powershell
python -m src.pod_generate_data --config config/default.yaml --grid-nx 36 --grid-ny 15 --early-points 9 --middle-points 5 --late-points 5 --final-time 20
```

注意：如果几何、网格、裂缝、物性参数、BHP 制度或 cell 排序变化，旧的 POD 快照通常不能继续使用，需要重新生成数据并重新训练 POD-MLP。

## 5. 时间采样

POD 数据生成不是简单均匀时间采样，而是默认分成早期、中期、晚期三段。

这样做的原因是压力扩散问题早期变化快、后期变化慢。如果只用均匀时间点，早期压降和裂缝附近快速传播容易采样不足；如果全部用对数采样，后期长期响应又可能不够。

配置中常见参数含义：

- `early_points`：早期时间点数量。
- `middle_points`：中期时间点数量。
- `late_points`：后期时间点数量。
- `final_time_days`：POD 数据覆盖的最终时间。

训练后，POD-MLP 默认只允许在训练时间范围内预测。超出范围属于外推，默认禁止。

## 6. BHP cell 为什么要从 POD 中排除

生产端 BHP 是 hard constraint。它的压力由井底压力公式直接给定，不应该交给 POD 模态和 MLP 自由拟合。

因此默认做法是：

1. 训练 POD 时排除 BHP hard-constrained cells。
2. MLP 只预测自由 cell 的 POD 系数。
3. 重构完成后，再把 BHP cells 的 `P12/P13` 精确覆盖为目标值。

这样可以避免一个问题：如果把 BHP cell 也放进 POD，它们的强制低压变化会在 POD 模态中占据很大权重，反而削弱对裂缝和基质压力传播形态的表达。

## 7. MLP 输入和输出

当前 POD-MLP 的输入不是坐标，也不是上一时刻压力场，而是低维时间特征：

```text
[t_norm, log_time_norm, bhp_total_hat]
```

其中：

- `t_norm` 表示线性归一化时间。
- `log_time_norm` 表示 `log1p(t)` 归一化时间，用于增强早期变化表达。
- `bhp_total_hat` 表示归一化后的总井底压力。

输出是 POD 系数：

```text
[a1, a2, ..., ar]
```

然后通过 POD 基重构为完整压力场。

## 8. 训练命令

训练 POD 基和 POD-MLP：

```powershell
python -m src.pod_train --config config/default.yaml
```

快速训练示例：

```powershell
python -m src.pod_train --config config/default.yaml --epochs 20 --batch-size 8 --fixed-rank 5
```

输出文件：

```text
outputs/pod/pod_basis.npz
outputs/checkpoints/pod/pod_mlp.pt
outputs/logs/pod/pod_training_history.csv
```

其中：

- `pod_basis.npz` 保存平均场、POD 模态、奇异值、保留能量、训练/验证/测试划分等。
- `pod_mlp.pt` 保存 MLP 权重、归一化信息和必要元数据。
- `pod_training_history.csv` 保存训练和验证 loss。

## 9. 评估方法

评估命令：

```powershell
python -m src.pod_evaluate --config config/default.yaml
```

评估时会区分两类误差：

- POD projection error：只看 POD 截断带来的误差。也就是用真实快照投影到 POD 模态后再重构，检查保留阶数够不够。
- POD-MLP error：看 MLP 预测系数后重构出来的误差。这个误差包含 POD 截断误差和 MLP 系数预测误差。

如果 projection error 已经很大，说明 POD 模态数量不够，应该提高 `max_rank` 或使用 `fixed_rank`。  
如果 projection error 很小但 MLP error 很大，说明模态足够，但 MLP 没学好，应该增加训练 epoch、调整学习率或改进时间采样。

评估输出包括：

```text
outputs/tables/pod/pod_metrics_by_time.csv
outputs/tables/pod/pod_metrics_summary.csv
outputs/tables/pod/pod_energy.csv
outputs/figures/pod/pod_cumulative_energy.png
outputs/figures/pod/pod_error_vs_time.png
outputs/figures/pod/pod_coefficients_vs_time.png
outputs/figures/pod/pod_training_history.png
```

## 10. 任意时间预测

在训练时间范围内预测指定时刻：

```powershell
python -m src.pod_predict --config config/default.yaml --times 50 123.4 437 800 --output-name custom_predictions.npz
```

输出：

```text
outputs/pod/custom_predictions.npz
```

如果确实要外推到训练时间范围之外，需要显式打开：

```powershell
python -m src.pod_predict --config config/default.yaml --times 1100 --allow-extrapolation
```

外推结果只能作为探索，不建议作为可靠物理结果。

## 11. 绘图

普通压力云图和剖面图可以直接读取 POD 预测 archive：

```powershell
python -m src.plot_fields --config config/default.yaml --snapshot-file outputs/pod/custom_predictions.npz
python -m src.plot_sections --config config/default.yaml --snapshot-file outputs/pod/custom_predictions.npz
```

如果要对比 POD/PINN 与 FVM 的误差，可以使用三联图脚本：

```powershell
python plot_cloud_maps.py --source pod --time 999 --component Ptotal --fvm-file outputs/direct_snapshots.npz
```

三联图含义：

- 左图：POD 或 PINN 压力场。
- 中图：FVM 参考压力场。
- 右图：误差场，即 `POD/PINN - FVM`。

注意：三联图要求两份 archive 的 matrix grid 完全一致，不能把不同 `nx/ny` 的结果直接相减。

## 12. main_POD.py 一键流程

`main_POD.py` 是 IDE 友好的流程入口。

完整流程：

```powershell
python main_POD.py all
```

它会依次执行：

1. 生成 POD 训练快照。
2. 训练 POD basis 和 POD-MLP。
3. 评估 POD projection 和 POD-MLP 误差。
4. 预测默认输出 `outputs/pod/pod_predictions.npz`。
5. 做气量和甲烷碳同位素后处理。
6. 绘制压力云图、剖面图、误差图和气量/同位素曲线。
7. 对预测 archive 做基础 diagnostics。

只重新计算气量和同位素：

```powershell
python main_POD.py gas
```

## 13. 输出文件汇总

核心 POD 文件：

```text
outputs/pod/direct_snapshots.npz
outputs/pod/pod_basis.npz
outputs/pod/pod_predictions.npz
outputs/pod/pod_test_predictions.npz
outputs/checkpoints/pod/pod_mlp.pt
outputs/logs/pod/pod_training_history.csv
```

评估表和图：

```text
outputs/tables/pod/source_well_history.csv
outputs/tables/pod/pod_metrics_by_time.csv
outputs/tables/pod/pod_metrics_summary.csv
outputs/tables/pod/pod_energy.csv
outputs/figures/pod/pod_cumulative_energy.png
outputs/figures/pod/pod_error_vs_time.png
outputs/figures/pod/pod_coefficients_vs_time.png
outputs/figures/pod/pod_training_history.png
```

气量和同位素后处理文件见 `GAS_POSTPROCESSING_CN_NOTES.md`。

## 14. 局限性

POD-MLP 的局限性很明确：

- 它强依赖训练快照，训练快照来自哪个物理设置，它就只适用于哪个物理设置。
- 它不保证局部守恒。
- 它不适合直接迁移到新裂缝几何、新网格或新物性参数。
- 它的可靠预测范围主要在训练时间区间内。
- 它不会自动发现新的物理机制，只是在已有快照低维空间中插值。

因此判断 POD-MLP 是否可信时，应优先检查：

- `pod_energy.csv` 中 POD 保留能量是否足够。
- `pod_metrics_by_time.csv` 中 test 时间点误差是否可接受。
- `pod_error_vs_time.png` 中误差是否集中在早期、晚期或某些 BHP 快速变化阶段。
- 如果用于气量和同位素，应继续检查 `pod_vs_reference_gas_metrics.csv`。

## 15. 和 v11 E-PINN 主线的关系

v11 目前有三类结果来源：

- `train`：E-PINN 训练结果，solver 字段通常为 `sparse_epinn_train`。
- `solve`：direct FVM/EDFM 数值参考解，solver 字段通常为 `direct_fvm_edfm`。
- `main_POD.py`：POD-MLP 降阶代理结果，solver 字段通常为 `pod_mlp`。

三者不能混淆。  
如果要检验 E-PINN 是否学好，应比较 `train` 和 `solve`。  
如果要检验 POD-MLP 是否可用，应比较 `pod_mlp` 和 direct FVM/EDFM 快照。
