# 论文初稿补充材料 #1（给写作 AI）

> 配套 `docs/2026-08-12_PAPER_WRITING_MANUAL.md` 与初稿 `docs/main (1).tex`。
> 本文件只做三件事：**① 稿件信息补正**；**② 可直接填入的真实数字**（每条附来源，已逐条核对）；
> **③ 明确指出初稿里一处需要改写的事实陈述**。
> 排版体例**沿用初稿的 Elsevier `elsarticle`**，本轮不做任何体例改动。
> 手册的全部规则继续有效——尤其是"手册/本文件没有的数字不得写进论文""无 cv5 支撑的判断只能写成设计理由"。
> 本文件里凡标 `[仍为 TODO]` 的，**留空，不要猜**。

---

## 0. 两处补正

### 0.1 标题已定，作者仍未定

```latex
\title{Physics-Guided Scale Gating for Floe-Size Segmentation in
       Sentinel-1 EW SAR Imagery}
```

作者、单位、通讯作者、基金、致谢：`[仍为 TODO]`——**不得按惯例填充任何机构名**。

### 0.2 摘要仍不在本轮范围

初稿里的 abstract 占位说明保留。若后续要写，必须与 §2 的增益结构一致（整体温和提升、最大单类增益在大浮冰、中浮冰未改善）。

---

## 1. 可直接填入的数字

### 1.1 Table 1 —— 数据集构成（可整表替换）

来源：`docs/2026-08-07_dataset_distribution.md`（传感器/年份/季节）、`datalists/cv5/cv5_report.md`（逐类像素，五折求和）、本轮对源数据目录的逐景比对（区域）。

```latex
\begin{table}[!t]
\caption{Dataset composition. Class pixel shares are computed over the valid
labelled pixels evaluated under the protocol of Section~\ref{sec:splits};
the ignore class is excluded.}
\label{tab:dataset}
\centering
\begin{tabular}{llr}
\toprule
Attribute & Value & Count \\
\midrule
Scenes      & total                                  & 417 \\
\midrule
Sensor      & Sentinel-1A                            & 342 \\
            & Sentinel-1C                            & 75 \\
\midrule
Region      & East Greenland (DMI charts)            & 398 \\
            & Canadian Arctic (AI4Arctic lineage)    & 19 \\
            & \quad Hudson Bay                       & 10 \\
            & \quad Beaufort Sea                     & 6 \\
            & \quad Canadian Arctic Archipelago      & 3 \\
\midrule
Season      & winter (DJF)                           & 59 \\
            & spring (MAM)                           & 187 \\
            & summer (JJA)                           & 159 \\
            & autumn (SON)                           & 12 \\
\midrule
            &                                        & Share (\%) \\
Class share & open water                             & 86.38 \\
            & cake ice                               & 0.49 \\
            & small floe                             & 2.38 \\
            & medium floe                            & 2.41 \\
            & big floe                               & 3.42 \\
            & vast floe                              & 4.93 \\
\bottomrule
\end{tabular}
\end{table}
```

补充数字，正文可用（不必都进表）：

- **总标注像素**：11.25 Gpx（各类 Mpx：water 9715.9 / cake 54.7 / small 268.1 / medium 270.8 / big 384.2 / vast 554.8）。
- **只看冰区的份额**（正文描述类别不平衡时更直观）：cake 3.57%、small 17.48%、medium 17.66%、big 25.05%、vast 36.18%。
- ⚠️ **口径统一说明**：以上像元数与 §2.1 的区域表同源——都取自 cv5 逐景混淆矩阵的行和，即"协议下实际参与评估的有效标注像元"。本文件初版曾用 tile 索引口径（总 11.10 Gpx），两者差约 1.3%，**不要混用**，统一用这一份。
- **年份分布**（如需一句）：2022: 15、2023: 102、2024: 134、2025: 108、2026: 58。
- **采集时间跨度**：2022-04-15 至 2026-06-21。

⚠️ 用词纪律：这些份额是"保留 tile 内的有效标注像素"口径，**不要写成 "of all pixels in the archive"**；按上面表标题那样写即可。

### 1.2 Table 2 —— 五折均衡体检（可整表替换）

来源：`datalists/cv5/cv5_report.md`。表里的百分数是**该类全库像素中落在该折验证集的比例**（理想值 20%）。初稿原表里的 "Groups" 一列没有逐折记录，**删掉该列**，换成更有信息量的逐类份额。

```latex
\begin{table}[!t]
\caption{Balance of the frozen five-fold grouped split. Percentages give the
share of each class's archive pixels that falls in the validation set of that
fold; a perfectly balanced split would give 20\% everywhere.}
\label{tab:folds}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccccccccc@{}}
\toprule
 & & & \multicolumn{6}{c}{Share of class pixels (\%)} \\
\cmidrule(lr){4-9}
Fold & Scenes & S1C & water & cake & small & medium & big & vast \\
\midrule
0 & 83 & 15 & 20 & 20 & 20 & 20 & 20 & 19 \\
1 & 84 & 15 & 20 & 20 & 20 & 20 & 21 & 19 \\
2 & 83 & 15 & 20 & 20 & 20 & 20 & 22 & 19 \\
3 & 84 & 15 & 20 & 20 & 20 & 20 & 17 & 22 \\
4 & 83 & 15 & 20 & 20 & 20 & 20 & 20 & 21 \\
\bottomrule
\end{tabular}
\end{table}
```

正文可加一句（有数据支撑）：417 景经分组闭包后收敛为 **292 组**，其中多景组 114 个，最大组 4 景。
另一句可选、用于说明稀有类的测量脆弱性（Limitations 里有用）：承载 big floe 的景全库仅 **12** 个、vast floe 仅 **9** 个（阈值：该类 ≥10⁵ 像素），因此逐折只有 2–3 个 big 景、1–3 个 vast 景。

### 1.3 §3.5 增强段（可直接替换 `[TODO: augmentation list]`）

来源：`configs/FLOE/_sota_base.py` 的 `data_augmentations`，语义按 `loaders.py` 的实现核对。

> **Augmentation.** Training patches are augmented with horizontal and vertical
> flips (each applied with probability 0.5), a random rotation drawn uniformly
> from $[-90^\circ, 90^\circ]$ and applied with probability 0.5, a random
> isotropic rescaling drawn uniformly from $[0.9, 1.1]$ and applied with
> probability 0.5, and CutMix with $\beta = 1.0$ applied with probability 0.5.

⚠️ 三点不要写错：旋转是**连续角度**（−90°~90° 均匀），不是 90° 的整数倍；缩放**各向同性**；三个几何增强各自独立按 0.5 的概率触发。

### 1.4 §4.4 / §4.6 参数量（可填 `[TODO: measured parameter and FLOP increment]`）

来源：按 `models/sota/segformer.py` 的层定义逐层解析计算（`embed_dim=256`、`S=4` 个尺度、证据 96 通道 = variogram 64 + polarimetric 32、transformer `d=64`/4 头/MLP ratio 2、variogram `n_probe=16`、5 个 lag、`mid=64`）。

| 模块 | 参数量 |
|---|---|
| Variogram evidence（探针 + mix + proj） | 47,360 |
| Polarimetric prior | 4,848 |
| Evidence-conditioned scale gate | 109,060 |
| Scale-axis transformer | 72,832 |
| **合计新增** | **234,100（≈0.23 M）** |

正文写法建议：在 §4.4 只写尺度轴 transformer 的 **0.073 M**；在 §4.6 汇总"三个机制合计新增 0.23 M 参数"。

**全模型总参数量（2026-09-16 实测，`_paper_t0_params.py`）**：基线 **24,738,310（24.74 M）**，PGSG **24,972,410（24.97 M）**，增量 **234,100（+0.95%）**——与上表逐层解析的合计完全一致。

`[仍为 TODO]`：① **FLOPs 增量**（需一次前向，开卡时 `--flops`）；② **硬件与每折训练耗时**。

### 1.5 §3.3 预处理段的三处补正

初稿这一段的算子与参数都对，但有三处要改：

1. **软件版本写全**。现在只写了 "ESA SNAP 12.0.0"。正确表述：

   > Scenes were processed with the Microwave Toolbox 12.0.1 on the ESA SNAP 12.0.0 platform.

   理由：本文用到的全部 SAR 算子（轨道、热噪声、边界噪声、定标、Refined Lee、地形校正、GLCM）都属于 Microwave Toolbox，该工具箱在本机安装后经更新中心升级到 12.0.1；SNAP 12.0.0 只是平台版本。

2. **引用年份错了**。初稿的 `\citep{esa2024snap}` 键名写的是 2024；**SNAP 12 的发布日期是 2025 年 5 月 8 日**，引用年份应为 **2025**（不是我们的使用年份 2026，也不是 2024）。请把 bib 条目的 year 改为 2025 并重命名键（如 `esa2025snap`）。
   `[仍为 TODO]`：SNAP 的具体引用条目格式——按目标期刊体例与 ESA STEP 官方推荐写法填写。

3. **补一句版本一致性**（审稿人会问，而我们恰好有干净答案）：

   > All 417 scenes were processed in a single campaign in July 2026 with an
   > unchanged software configuration.

其余保持不变；"入射角用椭球入射角而非局地入射角"那句初稿已写对，保留。

---

## 2. 两处必须改写的事实陈述（重要）

### 2.1 区域与 big/vast 的关系（已实测，初稿说法**成立**且应写得更强）

> 更正：本节初版曾判断"big/vast 主要来自加拿大北极"这句话可能写错了。2026-09-16 的实测
> （`_paper_t0_region.py`，对全部 417 景逐景累加真实类像元）推翻了那个判断——初稿是对的，
> 而且实际情况比它写的更极端。以下为实测结果。

**区域 × 类别的真实像元（Mpx，覆盖全部 417 景）**

| 区域 | 景数 | Water | Cake | Small | Medium | Big | Vast |
|---|---|---|---|---|---|---|---|
| East Greenland (DMI) | 398 | 9676.8 | 54.7 | 268.1 | 251.4 | **0.0** | **0.0** |
| Canadian Arctic (AI4Arctic) | 19 | 39.1 | 0.0 | 0.0 | 19.4 | **384.2** | **554.8** |

**每一类里各区域的占比**：cake 与 small **100% 来自东格陵兰**；big 与 vast **100% 来自加拿大北极**；medium 是唯一两边都有的类（东格陵兰 92.8% / 加拿大北极 7.2%）；water 99.6% 东格陵兰。
加拿大北极内部：Hudson Bay 贡献 big 的主体（350.6 Mpx），Beaufort Sea 贡献 vast 的主体（496.8 Mpx），CAA 两者都有一些。

**结论：类别与地域不是"相关"，是对 big/vast 而言的【完全混同】**——没有任何一个东格陵兰场景含有 big 或 vast 像元。所以按 **(B) 精确版**写，并且要写得比初稿更直白：

§3.1 可写：

> …supplemented with 19 scenes from the Canadian Arctic (Hudson Bay, Beaufort Sea
> and the Canadian Arctic Archipelago) carrying ice charts of the AI4Arctic lineage.
> These 19 scenes contribute all of the big-floe and vast-floe pixels in the archive,
> while cake and small floe occur only in the East Greenland scenes; medium floe is
> the only class present in both (92.8\% East Greenland, 7.2\% Canadian Arctic).

Limitations 必须相应加强（**这是本文最硬的一条局限，不要轻描淡写**）：

> Class and region are completely confounded for the two largest classes: every
> big-floe and vast-floe pixel in this archive comes from the 19 Canadian Arctic
> scenes, and no East Greenland scene contains either class. Grouped
> cross-validation prevents leakage between a scene and its neighbouring frames,
> but it cannot decorrelate class from geography. Consequently, a per-class result
> for big or vast floe is also a statement about a single region, and on any one
> fold it rests on very few independent scenes (fold 0 contains two big-floe and
> one vast-floe scene).

⚠️ **并且必须约束 Results 的措辞**：本文最大的单类增益是 big floe $+0.026$，而 fold 0 里承载 big 的只有 **2 个场景**、全部来自同一区域。报告这个增益时要同时说明它的支撑有多薄，不能让它读起来像一个稳健的类级结论。这也是 T1 的五折配对最该优先做的理由之一。

### 2.2 零初始化的**点数**写错了（三处 → 两处）

初稿两处把零初始化说成三处：

- Introduction：“All three modules are zero-initialized…”
- §4.6：“The final projection of the scale-axis transformer, the final convolution of the gate **and the output projection of the evidence pathway into the gate** are all zero-initialized.”

**实现里只有两处零初始化**（`models/sota/segformer.py`）：尺度轴 transformer 的输出投影 `out_proj`，与门控的末层卷积（权重与偏置）。**证据通路的投影是刻意不做零初始化的**——PGSG 构造 variogram 模块时显式传了 `zero_proj=False`，代码注释写明了理由：该模块的输出不是主通路上的残差、而只是门控的条件输入，所以"单零纪律"不适用；门控自己的末层已经零初始化、保证初始 $g\equiv1$，证据再零初始化只会让梯度多迟滞一层。

**重要的是：等价性结论不受影响，反而更值得讲。** 证据只流向门控与尺度轴 transformer 两个模块，而这两个模块的输出在 $t=0$ 时都被零初始化抹平（$g\equiv1$ ⇒ 门控融合等于不加门控的拼接；残差为 0 ⇒ 尺度特征不变），所以**即使证据分支从随机权重出发，整网在 $t=0$ 的前向仍与基线逐位相同**。

改写建议（§4.6）：

> Two projections are zero-initialized: the output projection of the scale-axis
> transformer and the final convolution of the gate. The evidence pathway itself is
> not zero-initialized, and does not need to be: evidence reaches only these two
> decision modules, whose contributions vanish at initialisation ($g \equiv 1$ and a
> zero residual), so the forward pass at $t=0$ is bit-for-bit identical to the
> baseline regardless of how the evidence branch is initialised.

Introduction 里那句改成"the decision modules are zero-initialized, so that at the start of training the forward pass is bit-for-bit identical to the baseline"即可，不要数"三个模块"。

Fig. 3 的图注同理：标出的是**两个**零初始化点，不是三个（`docs/fig3_architecture.tex` 已按两处画）。

---

## 3. 仍为 TODO 的项（留空，不要填）

**等实验**（清单见 `docs/2026-08-12_PAPER_PENDING_EXPERIMENTS.md`）：

- §1 动机 B：中浮冰 train/val IoU 与 gap（**仍缺，需开卡**）。
  ~~+2 dB 平移的 Δhard-ice~~ ✅ 已测：**+2 dB → −0.039、−2 dB → −0.054**（补充材料 #2 §2a）。
  ~~暗景 medium→small 误判比例~~ ❌ **该分句被 fold 0 全量数据否掉，整句删除**（补充材料 #2 §2c）。
- §1 动机 C：基线混淆矩阵里 big→small / big→medium 的比例。
- Fig. 2 动机图素材。
- Table 4：五折配对结果。
- §4.4/§4.6：FLOPs 增量、总参数量、硬件与耗时。

**等作者**：作者/单位/通讯作者/基金/致谢；SNAP 引用条目格式；PA²-Swin 的正式引用（§2.3 末尾那条 lineage 语句在拿到引用前保持注释状态）。

**已完成（2026-09-16 无卡实测，结果见补充材料 #2）**：区域 × 类别像素构成、动机 C 的混淆比例、参数量、以及 Fig. 1/4/5/6 四张图。

---

## 4. 两条写作口径提醒（承手册）

1. **同一评估集**：所有消融数字都在 fold 0（83 景验证集）上；`Experiments` 首段必须写明 "all ablations are reported on fold 0"，并且**不要**把任何"五折汇总"的绝对值与 fold 0 的值并排比较。
2. **rationale 不是 finding**：§4.2 的最小 lag 选择、§4.5 的 softmax 预算守恒、§3.4 的全局归一化，初稿已经正确地写成了设计论证（"argued from first principles rather than from an experimental comparison"）——保持这个措辞，不要在修订中改成"experiments show"。
