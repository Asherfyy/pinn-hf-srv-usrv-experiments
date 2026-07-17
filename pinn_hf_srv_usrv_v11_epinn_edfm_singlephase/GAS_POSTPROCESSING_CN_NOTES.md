# 区域累产气量与甲烷碳同位素后处理中文详解

本文件是 v11 中气量和甲烷碳同位素后处理的唯一说明文档。英文版 `GAS_POSTPROCESSING_README.md` 已删除，后续只维护这份中文说明。

相关代码：

- `src/gas_metrics.py`：负责从压力快照计算区域库存、累产气量、产气速率和同位素指标。
- `src/pod_gas_postprocess.py`：负责读取 POD/FVM/PINN 快照，调用 `gas_metrics.py`，保存 CSV/NPZ 和曲线图。
- `main_POD.py`：在 POD-MLP 一键流程中自动调用气量和同位素后处理。

## 1. 后处理使用什么数据

后处理读取的是已经恢复到 MPa 单位的物理压力场：

```text
pressure_mpa[time, cell, component]
```

当前 v11 默认有两个压力分量：

- `P12`：代表含 C12 的甲烷压力分量。
- `P13`：代表含 C13 的甲烷压力分量。

后处理不直接读取归一化压力，也不直接读取 POD 系数。  
如果输入来自 POD-MLP，程序会先把 POD 系数重构为完整 `P12/P13` 压力场，再计算气量和同位素。

## 2. 区域累产气量的物理含义

`Q_HF`、`Q_SRV`、`Q_USRV` 表示对应区域的库存亏空量：

```text
区域累产气量 = 初始气体库存量 - 当前剩余气体库存量
```

这里的“区域”是 HF、SRV、USRV，而不是气体真实来源示踪。  
因此这些量更准确地说是区域库存变化量，不是严格的流线追踪贡献。

这带来几个重要结果：

- 单个区域的 `Q` 可以短时为负。
- 单个区域的 `Q` 可以局部非单调。
- 程序不会把负值强制截断成 0。
- 程序不会对区域 `Q` 做累计最大值修正。

总累产气量定义为：

```text
Q_cum = Q_HF + Q_SRV + Q_USRV
```

每个区域还保留两个同位素分量：

```text
Q_HF   = Q1_HF   + Q2_HF
Q_SRV  = Q1_SRV  + Q2_SRV
Q_USRV = Q1_USRV + Q2_USRV
```

其中：

- `Q1` 对应 C12 甲烷。
- `Q2` 对应 C13 甲烷。

## 3. 离散积分公式

对压力分量 `c` 和区域 `r`，累产气量计算为：

```text
Q_c_r(t) =
    gas_conversion_factor
    * sum_i(
        Fai_i
        * regional_integration_weights_m3[i, r]
        * (P_c_initial - P_c_i(t))
    )
```

各项含义：

- `i`：cell 编号，包括 matrix cell 和 EDFM fracture cell。
- `r`：区域编号，取 `HF/SRV/USRV`。
- `Fai_i`：cell 的储集/孔隙参数，来自 `grid.cell_phi`。
- `regional_integration_weights_m3[i, r]`：cell 对区域 `r` 的体积积分权重。
- `P_c_initial`：分量 `c` 的初始压力。
- `P_c_i(t)`：时刻 `t`、cell `i`、分量 `c` 的压力，单位 MPa。
- `gas_conversion_factor`：把 `Fai * volume * pressure` 转换到标准立方米的系数。

这个公式本质上是对每个区域做库存差积分。压力降低越多，库存亏空越大，累产气量越大。

## 4. 气体换算因子

气体换算因子采用理想气体关系：

```text
gas_conversion_factor =
    pressure_MPa_to_Pa
    * Vm_std_m3_per_mol
    * model_count
    /
    (R_J_per_mol_K * temperature_K * production_scale_divisor)
```

默认常数来自配置 `gas_postprocessing.constants`。

当前默认值对应：

```text
gas_conversion_factor = 0.016173427226937653
```

注意：

- 压力输入单位是 MPa，所以需要乘 `pressure_MPa_to_Pa`。
- 输出单位是标准立方米。
- 当前实现不额外引入气体压缩因子 `Z`。
- 如果要和 COMSOL 或其它软件对齐，优先检查这些常数是否一致。

## 5. 厚度修正和 HF 裂缝体积

FVM/EDFM 求解使用的 `cell_volume` 已经包含源网格厚度。  
为了和参考模型的储层厚度一致，程序引入厚度修正：

```text
thickness_correction =
    reference_reservoir_thickness_m / source_grid_thickness_m
```

当前默认：

```text
source_grid_thickness_m = 1.0
reference_reservoir_thickness_m = 8.0
thickness_correction = 8.0
```

matrix cell 的有效积分体积会乘这个厚度修正。

HF EDFM 裂缝 cell 的原始体积为：

```text
fracture_segment_length * fracture_aperture * source_grid_thickness_m
```

程序不会再次乘以 aperture。  
这样避免把裂缝体积重复缩小。

## 6. 区域积分权重

`regional_integration_weights_m3` 是一个二维数组：

```text
[cell, region]
```

它告诉程序：每个 cell 有多少体积应该计入 HF、SRV 或 USRV。

默认逻辑：

- HF EDFM fracture cell 计入 HF。
- matrix cell 按 cell center 所属区域计入 SRV 或 USRV。
- BHP well cell 是否参与库存统计由配置控制。

如果启用：

```yaml
gas_postprocessing:
  behavior:
    exclude_well_cells_from_inventory: true
```

则 BHP 控制 cell 不参与区域库存积分，避免井控强制压力点对区域库存造成过强影响。

## 7. 甲烷碳同位素 delta 的计算

同位素标准比值为：

```text
Rst = isotope_standard_ratio
```

产出端瞬时同位素值使用累产分量对时间的导数：

```text
rate1 = d(Q1_cum) / dt
rate2 = d(Q2_cum) / dt
delta_prod = ((rate2 / rate1) / Rst - 1) * 1000
```

输出字段：

```text
delta_prod_permil
```

单位是 per mil，也就是千分值。

程序使用：

```text
np.gradient(values, times_days, edge_order=edge_order)
```

在非均匀时间点上计算导数。  
因为 `rate2/rate1` 是比值，所以时间单位使用 day 不会改变同位素比值。

如果 `rate1` 太小，`delta_prod_permil` 会保留为 `NaN`，不会强行替换成 0。

## 8. 另外两个同位素指标

除了产出端瞬时同位素，程序还输出：

```text
delta_cum_permil
delta_remaining_permil
```

含义：

- `delta_cum_permil`：使用 `Q2_cum / Q1_cum` 计算累积产出同位素。
- `delta_remaining_permil`：使用剩余总库存 `G2_remaining_total / G1_remaining_total` 计算储层剩余气体同位素。

剩余库存定义为：

```text
G1_remaining = G1_initial_total - Q1_cum
G2_remaining = G2_initial_total - Q2_cum
```

如果分母过小或无效，对应 delta 保留为 `NaN`。

## 9. 为什么不用 well_history 直接算气量

`well_history` 中的 rate 是求解器侧边界通量诊断量。  
当前 gas 后处理要对齐的是基于库存差的 COMSOL 等价公式，所以气量来自区域库存积分，而不是直接来自 `well_history`。

也就是说：

```text
当前气量 = 区域压力亏空积分
不是
当前气量 = 井口通量历史积分
```

这两种方法在离散误差、边界处理和区域归因上可能不完全相同。

## 10. 常用命令

使用 POD-MLP 在指定时间范围内预测并计算气量：

```powershell
python -m src.pod_gas_postprocess --config config/default.yaml --start-time 0 --end-time 1000 --num-times 1001
```

处理已有 POD 预测文件：

```powershell
python -m src.pod_gas_postprocess --config config/default.yaml --prediction-file outputs/pod/pod_predictions.npz
```

处理 direct FVM 快照：

```powershell
python -m src.pod_gas_postprocess --config config/default.yaml --source-snapshot-file outputs/pod/direct_snapshots.npz --output-name direct_gas_production.csv
```

对比 POD 与 direct FVM 的气量和同位素：

```powershell
python -m src.pod_gas_postprocess --config config/default.yaml --reference-snapshot-file outputs/pod/direct_snapshots.npz --start-time 0 --end-time 1000 --num-times 1001
```

IDE 一键流程：

```powershell
python main_POD.py all
python main_POD.py gas
```

其中：

- `main_POD.py all` 会生成快照、训练 POD-MLP、预测、绘图并计算气量/同位素。
- `main_POD.py gas` 只读取已有 `outputs/pod/pod_predictions.npz` 并重新生成气量和同位素结果。

## 11. 输出文件

CSV 表格：

```text
outputs/tables/pod/pod_gas_production.csv
outputs/tables/pod/pod_gas_production_summary.csv
outputs/tables/pod/pod_vs_reference_gas_metrics.csv
```

NPZ archive：

```text
outputs/pod/pod_gas_production.npz
```

图像：

```text
outputs/figures/pod/pod_cumulative_gas.png
outputs/figures/pod/pod_component_cumulative_gas.png
outputs/figures/pod/pod_gas_rate.png
outputs/figures/pod/pod_isotope_delta.png
outputs/figures/pod/pod_regional_fraction.png
outputs/figures/pod/pod_vs_reference_cumulative_gas.png
outputs/figures/pod/pod_vs_reference_isotope.png
```

其中 `pod_isotope_delta.png` 默认重点绘制：

```text
delta_prod_permil
```

`delta_cum_permil` 和 `delta_remaining_permil` 仍会写入 CSV/NPZ，方便数值检查。

## 12. 结果检查建议

检查气量和同位素时建议按以下顺序：

1. 先确认输入压力 archive 的 `pressure_mpa` 没有 NaN/Inf。
2. 确认 `times_days` 严格递增，至少有 3 个时间点，否则无法稳定计算导数。
3. 确认 archive 的 cell ordering 和当前配置网格一致。
4. 查看 `Q1_cum/Q2_cum/Q_cum` 是否在量级上合理。
5. 查看 `delta_prod_permil` 中 NaN 是否出现在早期极小产量或分母接近 0 的位置。
6. 如果是 POD 结果，先看 POD pressure error，再判断 gas/isotope error。

## 13. 局限性

当前后处理有以下限制：

- 它依赖 MPa 物理压力场，不接受归一化压力。
- 它默认当前模型只有 `P12/P13` 两个压力分量。
- 它是库存差后处理，不是严格的组分运移模拟。
- 它没有求解吸附/解吸、真实气体状态方程或多相流。
- POD 后处理的可靠性受 POD-MLP 压力预测误差限制。
- 如果几何、网格、厚度、Fai、BHP 制度改变，应重新生成快照和后处理结果。

## 14. 与 POD-MLP 的关系

POD-MLP 只负责给出压力场：

```text
t -> POD coefficients -> pressure_mpa
```

gas 后处理负责把压力场转换为区域库存和同位素：

```text
pressure_mpa -> Q1/Q2/Q_HF/Q_SRV/Q_USRV/Q_cum -> delta
```

两者是串联关系。  
如果 POD-MLP 压力场在裂缝附近或生产端附近误差较大，气量和同位素误差会被放大。因此用 POD 结果做气量分析时，必须同时保留 direct FVM 参考对比。
