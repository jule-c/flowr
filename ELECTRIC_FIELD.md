# 电场引导的扩散分子生成（基于 FLOWR 的增量实现）

本目录在 **不改动的 FLOWR 主体代码** 的基础上，按照国家自然科学基金申请书
《考虑靶蛋白活性口袋电场效应的扩散分子生成模型及其应用研究》第 15–16 页"生成模型"部分，
以**纯增量（只增加、不删除）**的方式实现了：

> **将基于结构的分子生成转换成原子在蛋白质电场引导下的扩散过程。**

所有新功能都由新增的命令行开关控制，**默认关闭**；关闭时模型结构、参数和数值行为与原 FLOWR 完全一致（旧 checkpoint 可以照常加载和推理）。

---

## 一、与申请书公式的对应关系

| 申请书内容 | 实现位置 | 说明 |
|---|---|---|
| 公式(5)：口袋附近任意位置 i 的电场强度叠加 `E_i = (1/4πε) Σ_j q_j·(r_i−r_j)/\|r_i−r_j\|³` | `flowr/models/electric_field.py` → `electrostatic_field()` | 常数 1/(4πε) 折入下游可学习的投影 / 引导强度；带 `softening` 软化距离与可选 `cutoff` 截断 |
| 公式(5) 融入消息传递与注意力 | `electric_field.py` → `ElectricFieldEncoder`；注入点在 `flowr/models/pocket.py` 的 `LigandDecoder.forward` | 电场以三种方式注入：① 等变特征（E 是极矢量，逐分量线性映射，保持 SE(3) 等变）；② 不变节点特征（\|E\|、电势 φ 及其 log 压缩，经 MLP 加入注意力节点特征）；③ 配体–口袋逐对静电特征（高斯展开后加到交叉注意力的边特征上） |
| 公式(6)：原子类型的 VP（方差保留）扩散 | `flowr/data/vp_diffusion.py` → `vp_signal_coeff / vp_mean_std / vp_noisy_types`；插值分支在 `flowr/data/interpolate.py`（`--categorical_strategy vp`） | 信号系数 s(t) = 1 − exp(−∫β)，x_t = √s·x₁ + √(1−s)·ε，满足 VP 性质 s + (1−s) = 1 |
| 公式(7)–(8)：VP 扩散向量场 u_t | `flowr/data/vp_diffusion.py` → `vp_type_vector_field()` | 推导得 u_t = ½β(t)·(x̂₁/√s(t) − x_t)，从噪声指向数据 |
| 公式(9)：原子位置的最优传输路径 x_t = (1−t)x₀ + t·x₁ | 原代码已实现（`GeometricInterpolant` 的 OT 匹配 + 线性插值）；参考实现见 `vp_diffusion.py` → `ot_coord_path / ot_coord_vector_field` | FLOWR 本身即 OT 流匹配 |
| 公式(10)：位置向量场 dx/dt = x₁ − x₀ | 原代码已实现：`flowr/models/fm_pocket.py` → `Integrator.step` 中的 `coord_velocity = (x̂₁ − x_t)/(1−t)` | 不需要改动 |
| 公式(11)：给定口袋 C 和电场 E 条件下的流匹配损失 | 损失函数不变；条件中新增电场 E（`use_electric_field`） | 网络以 (C, E) 为条件预测向量场，损失自动成为 L_E |
| **原子在电场引导下的扩散（核心思想）** | ① 条件注入（训练期，见上）；② 采样期引导：`electric_field.py` → `electric_field_drift()`，`fm_pocket.py` → `Integrator.step`（`--efield_guidance`） | 采样时每个积分步给坐标更新加上电场对原子的力 F_i = q_i·E_i（q_i 取模型预测电荷分布的期望），使逆扩散成为原子在蛋白电场中的受引导扩散过程 |

## 二、新增 / 修改的文件

**新增（2 个模块 + 1 个测试 + 1 个文档）：**
- `flowr/models/electric_field.py` —— 公式(5)电场叠加、电场编码器（等变/不变/逐对特征）、采样期电场漂移
- `flowr/data/vp_diffusion.py` —— VP 扩散调度与前向加噪、VP 与 OT 向量场
- `scripts/test_electric_field.py` —— 冒烟测试（数值对拍、等变性、前后向一致性）
- `ELECTRIC_FIELD.md` —— 本文档

**增量修改（只在原有代码中追加，未删除任何原有逻辑）：**
- `flowr/models/pocket.py`：`LigandDecoder` / `LigandGenerator` 新增 `use_electric_field`、`efield_cutoff`、`efield_softening`、`efield_num_rbf`、`pocket_charge_values`（全部有默认值）
- `flowr/models/fm_pocket.py`：`Integrator` 新增 `efield_guidance`（采样步中的电场漂移项）；`LigandPocketCFM` 新增电场选项、在 `forward` 中自动把口袋 one-hot 电荷转成数值电荷并传给网络、在 `_generate` 采样循环中逐步计算电场漂移
- `flowr/data/interpolate.py`：`type/bond_interpolation` 新增 `"vp"` 选项（公式(6)的 VP 前向加噪分支）
- `flowr/train.py`：新增命令行参数并在建模型 / 积分器 / CFM 时透传；`--categorical_strategy vp`
- `flowr/gen/generate_from_pdb.py`：从 checkpoint 超参恢复电场相关配置（旧 checkpoint 自动回退到关闭状态）

## 三、电荷来源

- 默认：口袋原子电荷使用数据集中已有的 7 类 one-hot 电荷（对应 `flowr/util/rdkit.py` 的 `IDX_CHARGE_MAP`，数值 0、±1、±2、+3、−3），由 `charges_to_values()` 自动转换。
  **注意**：这些电荷来自 PDB 文件第 79–80 列的形式电荷标注，绝大多数晶体结构该列为空，即蛋白原子电荷几乎全为 0——此时电场恒为 0，引导不会起任何作用。
- **推荐（与申请书一致，OPLS4 偏电荷）**：用 `scripts/add_opls4_charges.py`（Schrödinger prepwizard + structconvert）或你们前期工作已有的 OPLS4 加电荷管线，给靶蛋白生成每原子偏电荷的 PQR 文件，再通过推理命令的 `--pocket_charge_file` 传入。代码按（链, 残基号, 残基名, 原子名）自动把电荷对齐到口袋原子（`flowr/util/partial_charges.py`），偏电荷优先级高于数据管线的形式电荷。

```bash
# 1) 生成带 OPLS4 电荷的 PQR（需要 $SCHRODINGER 和许可证）
python scripts/add_opls4_charges.py \
    --pdb_file gen_gpr119/GPR119_protein.pdb \
    --out_pqr gen_gpr119/GPR119_opls4.pqr

# 2) 生成时挂上电荷文件（与 --efield_guidance 搭配使用）
python -m flowr.gen.generate_from_pdb ... \
    --pocket_charge_file gen_gpr119/GPR119_opls4.pqr \
    --efield_guidance --efield_guidance_scale 0.05
```

## 四、使用方法

训练（电场条件 + 采样期电场引导）：

```bash
python -m flowr.train \
  --dataset kinodata --data_path <DATA_PATH> --arch pocket --pocket_noise fix \
  --use_electric_field \
  --efield_guidance --efield_guidance_scale 0.05 \
  --efield_cutoff 10.0 --efield_softening 0.5 --efield_drift_mode force
```

若需要按申请书的 VP 方式对原子类型做前向加噪（公式(6)），加上：

```bash
  --categorical_strategy vp
```

推理（`flowr/gen/generate_from_pdb.py`）无需额外参数：电场配置保存在 checkpoint 的超参中，会自动恢复。

**参数说明：**

| 参数 | 默认 | 含义 |
|---|---|---|
| `--use_electric_field` | 关 | 网络条件中加入蛋白电场 E（公式(5)），注入消息传递与注意力 |
| `--efield_guidance` | 关 | 采样时原子沿电场受力漂移（"原子在电场引导下的扩散"） |
| `--efield_guidance_scale` | 0.0 | 漂移强度；每个积分步坐标更新 += step_size × scale × drift |
| `--efield_cutoff` | 10.0 Å | 电场计算的距离截断（余弦窗） |
| `--efield_softening` | 0.5 Å | 距离软化，避免近邻原子电场发散 |
| `--efield_drift_mode` | force | `force`：原始静电力 q·E；`unit`：q·E 方向（仅作用于带电原子）；**`field`：全原子沿局部电场方向（推荐）** |
| `--categorical_strategy vp` | — | 原子类型/键采用 VP 方差保留加噪训练（公式(6)–(8)） |

**强度标定（重要）**：`scripts/demo_charge_guidance.py` 在真实 GPR119 口袋上测到，原始静电力 q·E 即使配 OPLS4 级偏电荷、scale=5 时全程累计位移仍 <0.01 Å（相对扩散总位移 ~2 Å），基本无效；而 `field` 模式的累计位移 ≈ scale Å，推荐扫描 **0.05–0.3**（对应 2.5%–15% 的结构扰动）。且模型预测的是形式电荷，中性配体原子 q≈0，`force`/`unit` 模式几乎不作用——请优先使用 `field` 模式。

**建议**：先只开 `--use_electric_field` 训练（网络学会"感知"电场）；引导强度 `efield_guidance_scale` 按上表从 0.05–0.3 起调，过大可能把原子推出口袋或破坏键长。

## 五、验证

```bash
python scripts/test_electric_field.py      # 冒烟测试
python scripts/demo_charge_guidance.py --data_dir gen_gpr119   # 真实口袋上的电场/漂移机制验证
```

覆盖：公式(5)与手算值对拍、电场及编码特征的 SE(3) 等变性、`use_electric_field=False` 时参数集与原模型完全一致、开启后前向传播（含与相互作用分支共存）、积分器电场漂移的开/关一致性、VP 调度端点与方差保留性质、VP 向量场。
