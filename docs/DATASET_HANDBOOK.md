# FLOE 浮冰尺寸分类数据集 — 说明手册

> 面向后续训练环节的交付文档。数据集为**多源合并的 Sentinel-1 SAR + 浮冰尺寸(FLOE)逐像素标签**，
> 目标任务：**海冰浮冰尺寸(floe size)语义分割 / 逐像素分类**。
> 版本日期：2026-07-04。

---

## 1. 一句话概述
每个样本是一景 Sentinel-1 EW GRD SAR 影像，经 SNAP 预处理得到 **HH/HV(dB) + 入射角 + 3 个 GLCM 纹理**共 6 个输入特征通道，
配套一张**逐像素 FLOE 标签**（浮冰尺寸类别）。共 **250 景**，其中海冰四类(Small/Medium/Big/Vast)已做过**跨来源均衡**。

---

## 2. 文件清单与位置

| 目录 | 内容 | 数量 | 说明 |
|---|---|---|---|
| `.../CapeFarewell/data_nc/` | **未归一化** NC | **250** | 合并后的原始数据集 |
| `.../CapeFarewell/data_nc_zscore/` | **已归一化** NC（训练直接用这个） | **248** | 全局 z-score；另 2 个因内存不足未生成，见 §9 |
| `.../data_nc_zscore/zscore_stats.json` | 归一化统计量 | 1 | 每个特征的 mean/std（见 §8/附录） |

- 单文件命名沿用 Sentinel-1 场景名，如 `S1A_EW_GRDM_1SDH_20240103T200401_..._FC74.nc`。
- 格式：**NetCDF4（h5netcdf 引擎，zlib 压缩）**。逐景独立文件，**尺寸不固定**（各景 y×x 不同，如 6009×12738），像元约 **80 m**。
- 目录名沿用了 "CapeFarewell"，但**内容是多源合集**（见 §6），非仅 CapeFarewell。

---

## 3. 变量（每个 NC 内的 data_vars）

| 变量 | dtype | 含义 | 是否归一化 | 备注 |
|---|---|---|---|---|
| `nersc_sar_primary` | float32 | HH 后向散射 σ0 (dB) | 是(z-score) | 主偏振 |
| `nersc_sar_secondary` | float32 | HV 后向散射 σ0 (dB) | 是 | 副偏振 |
| `sar_incidenceangle` | float32 | 入射角 (度) | 是 | |
| `glcm_sigma0_hh_contrast` | float32 | GLCM 对比度(基于 HH) | 是 | 纹理 |
| `glcm_sigma0_hh_dissimilarity` | float32 | GLCM 相异度 | 是 | 纹理 |
| `glcm_sigma0_hh_homogeneity` | float32 | GLCM 同质度 | 是 | 纹理 |
| **`FLOE`** | **uint8** | **浮冰尺寸标签**（见 §4） | 否 | 训练目标 |
| `global_valid_mask` | float32(0/1) | 有效像元掩膜 | — | 1=有效, 0=无效(见 §8) |

**输入特征通道顺序建议**：`[nersc_sar_primary, nersc_sar_secondary, sar_incidenceangle, glcm_contrast, glcm_dissimilarity, glcm_homogeneity]`（6 通道）。

⚠️ 少数来自加拿大北极的文件(见 §6)**额外带 `SOD`(可能还有 `SIC`)变量**——那是旧数据集遗留、**未归一化**、与本任务无关。**训练时请按变量名显式选取上述 6 个特征 + `FLOE`，忽略 SOD/SIC。**

---

## 4. 标签定义（`FLOE` 变量）

uint8，取值 **0–6 为类别，255 为忽略(ignore)**：

| 值 | 类别 | 本数据集是否有 |
|---|---|---|
| 0 | Open water 开阔水 | 有（背景，占绝大多数） |
| 1 | Cake ice 饼状冰 | **有**（2026-07-16 扩充新增；此前为 0） |
| 2 | Small floe 小浮冰 | 有 |
| 3 | Medium floe 中浮冰 | 有 |
| 4 | Big floe 大浮冰 | 有 |
| 5 | Vast floe 巨浮冰 | 有 |
| 6 | Bergs 冰山类 | **无(0)** |
| **255** | **mask / 忽略**（陆地、SAR 黑边、模糊多边形、快冰等） | 有（占全图约 2/3） |

- **255 必须在 loss 中忽略**（`ignore_index=255`）。它包含陆地、雷达黑边、以及被判为模糊/非目标的像元。
- 本数据集实际出现的类别为 **{0, 1, 2, 3, 4, 5, 255}**（2026-07-16 扩充后 Cake(1) 有像素；此前为 {0,2,3,4,5,255}）；Bergs(6) 仍全为 0（来源里就没有），按约定不纳入。
- 标签来自 SIGRID-3 冰图多边形栅格化 + 官方 floe-size 编码映射；2/3 类的判定阈值做过放宽(CT≥6 且主导浮冰占比≥0.65)以提升 MIZ 小/中浮冰召回。

---

## 5. 类别分布与均衡

### 5.1 现行版本（2026-07-16 扩充后，train.json 356 景 / val.json 40 景）

**train（356 景）**，Mask(255) 占全图 69.9%，Bergs(6) 为 0：

| 标签 | 像素 | 占有效像元 | 占冰区(1-5) | 含该类景数 |
|---|---|---|---|---|
| Water(0) | 85.0 亿 | 89.4% | — | — |
| Cake(1) | 3,051 万 | 0.32% | 3.0% | 27 |
| Small(2) | 2.41 亿 | 2.53% | 23.9% | 227 |
| Medium(3) | 2.39 亿 | 2.51% | 23.7% | 200 |
| Big(4) | 2.43 亿 | 2.56% | 24.1% | 7 |
| Vast(5) | 2.56 亿 | 2.70% | 25.4% | 5 |

- 原四冰类依然近乎均衡（各占冰区 23.7–25.4%）；新增 Cake 像素上是显著少数类（占冰区 3.0%），但景级覆盖 27/356，多样性反而好于 Big(7 景)/Vast(5 景)。
- **val（40 景）冰区构成**：Cake 0.8%（4 景，四区各一、含 Shuga 日）、Small 3.9%（19 景）、Medium 5.8%（24 景）、Big 28.7%（5 景）、Vast 60.8%（4 景）。val 冰区被 Big/Vast 主导是历史遗留（8 个北极大场景像素量大）：逐类 IoU/F1 不受影响，但像素加权的总体指标会偏向这两类。

### 5.2 历史版本（250 景时代，无 Cake，供旧实验对照）

**排除 Mask 掩膜后**（有效像元 = 水+冰，合计约 4497 万 km²）：

| 标签 | 面积(万 km²) | 占有效像元 | 占冰区(2-5) |
|---|---|---|---|
| Water 水 | 4033 | 89.67% | — |
| Small(2) | 117 | 2.60% | 25% |
| Medium(3) | 100 | 2.23% | 22% |
| Big(4) | 119 | 2.65% | 26% |
| Vast(5) | 128 | 2.84% | 28% |

- **海冰四类基本均衡(22–28%)** —— 这是本数据集的核心设计目标（Small/Medium 来自东格陵兰，Big/Vast 来自北极合并补入）。
- 但**水 vs 冰极度不平衡**：有效像元里 ~90% 是水、仅 ~10% 是海冰。**训练务必用类权重/focal loss，并以“含冰”区域为中心裁 patch**，否则随机 patch 几乎全是水/mask。
- 上表为**全 250 景**的量；当前**归一化子集 248 景**因缺 2 个北极文件(见 §9)，Vast/Medium 略低，建议补齐后再定稿。

---

## 6. 数据来源与构成（重要：存在“类别-地域”相关）

共 250 景 = **242 东格陵兰 + 8 加拿大北极**：

| 来源 | 景数 | 提供的主要类别 | 冰图源 |
|---|---|---|---|
| 东格陵兰 CapeFarewell / CentralEast / SouthEast | 242 | **Small(2) / Medium(3)** | DMI 格陵兰区域冰图(RIC) |
| 加拿大北极 Beaufort(BS) / CAA / Hudson Bay(HB) | 8 | **Big(4) / Vast(5)**（少量 Medium） | 旧 SOD 数据集(AI4Arctic 谱系) |

- SAR 传感器统一为 **Sentinel-1 EW 模式 GRD 中分辨率(GRDM, 1SDH)**；两来源同一套 SNAP 预处理(§7)、变量/编码/dB 值域一致，可直接合并。
- ⚠️ **关键局限（务必告知训练/评估）：类别与地域强相关** —— Small/Medium **只**来自东格陵兰、Big/Vast **只**来自加拿大北极。
  模型可能学到“**某地域的 SAR 纹理**”而非真正的浮冰尺寸特征，**跨区域泛化存疑**。
  - 建议：**train/val/test 划分要避免同一景/相邻日期泄漏**；如条件允许，测试集应包含“同一地域内不同尺寸”的样本来检验模型是否真的在学尺寸；报告指标时注明此偏置。

---

## 7. SAR 预处理流程（SNAP，两来源一致）

`Sentinel-1 EW GRDM (.zip)` → 经 SNAP `gpt` 图处理：
1. Apply-Orbit-File（精密轨道）
2. ThermalNoiseRemoval（HH,HV）
3. Remove-GRD-Border-Noise
4. Calibration → σ0（HH,HV）
5. Speckle-Filter：**Refined Lee**（窗口 7×7，目标 3×3）
6. Terrain-Correction：**Copernicus 90m DEM，像元 80 m**，输出 incidenceAngleFromEllipsoid
7. LinearToFromdB（σ0 → dB）
8. 无效填充：HH==0 的出界/黑边像元置 **-9999**（nodata）
9. GLCM(基于 Sigma0_HH)：angleStr=ALL，量化 64 级，**窗口 11×11**，位移 2 → 输出 Contrast / Dissimilarity / Homogeneity

标签侧：SIGRID-3 冰图多边形按 SAR 网格栅格化并映射为 FLOE 类；陆地(POLY_TYPE='L')与 SAR 无效区统一置 **255**。

---

## 8. 归一化（全局 z-score，训练用 `data_nc_zscore`）

- 对上述 **6 个特征**在**全体样本**上统计全局 mean/std 后做 `(x-mean)/std`；标签 `FLOE` 不归一化。
- SAR dB 通道(`nersc_sar_primary/secondary`)先按 **[-35, 5] dB 截断**再归一化。
- **无效像元处理(training-safe)**：nodata=-9999；归一化后**无效像元的特征值统一映射为 0**，并额外写出 `global_valid_mask`（1=有效, 0=无效）。训练时可用该 mask 或 `FLOE==255` 排除无效区。
- 每个变量的属性里带 `zscore_mean`/`zscore_std`（便于反归一化/核对）。
- 全局 mean/std 见 `zscore_stats.json`（附录列出）。

---

## 9. 已知问题与注意事项

1. **归一化缺 2 景（内存不足，非数据损坏）**：`...20241104...`(CAA) 与 `...20250202...`(BS) 在 apply 阶段 `MemoryError` 被跳过，故 `data_nc_zscore` 只有 **248/250**。这 2 个恰是北极的 **Vast/Medium 贡献者**，缺失会让归一化子集的 **Vast/Medium 偏低**（相比 §5 的均衡量）。**建议**：腾出内存后重跑 step5(会跳过已好的 248、只补这 2 个)，即可恢复完整 250 与 §5 的均衡。
2. **类别-地域相关**（见 §6）——最重要的科学局限。
3. **水/冰极度不平衡**（§5）——需类权重/focal + 按冰裁 patch。
4. **255=忽略**——loss 必须 `ignore_index=255`。
5. **图幅尺寸不一**——各景 y×x 不同，训练需按 patch 裁切或 resize；无统一空间参考系(NC 内未存 CRS/geotransform，逐景像元 80m)。
6. **个别文件多余变量 SOD/SIC**（§3）——按名取 6 特征+FLOE，忽略其余。

---

## 10. 训练使用建议（速查）
- 输入：6 通道(顺序见 §3)，已 z-score；无效区已为 0，可信 `global_valid_mask`。
- 标签：`FLOE`，类别 {0,2,3,4,5}，**255 忽略**。若只做“浮冰尺寸 4 类”，可把 0(水)也并入 ignore 或单列背景类，按需。
- 采样：**以含冰像元为中心裁 patch**；配 **类权重 / focal loss** 对抗水主导与类间差异。
- 划分：按**景**划分 train/val/test，避免相邻日期/同景泄漏；正视类别-地域偏置(§6)。

---

## 附录：全局归一化统计量（`zscore_stats.json`，nodata=-9999）

| 特征 | mean | std |
|---|---|---|
| nersc_sar_primary (HH dB) | -13.9018 | 8.3349 |
| nersc_sar_secondary (HV dB) | -27.0105 | 9.4454 |
| sar_incidenceangle (deg) | 34.2622 | 7.7648 |
| glcm_sigma0_hh_contrast | 51.6718 | 75.3112 |
| glcm_sigma0_hh_dissimilarity | 6.4706 | 4.0463 |
| glcm_sigma0_hh_homogeneity | 0.6907 | 0.3150 |

（统计像元数 count ≈ 1.43×10¹⁰，为全体样本有效像元。）
