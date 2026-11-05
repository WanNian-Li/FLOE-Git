# 论文初稿补充材料 #2 —— T0 实测结果与四张图（给写作 AI）

> 2026-09-16 在 AutoDL **无卡模式**（0.5 核 / 2 GB）上跑完，零 GPU。
> 图文件在 `docs/figs/`（每张都有 `.pdf` 用于排版、`.png` 用于预览）。
> 规则同前：**本文件与补充材料 #1、手册之外的数字一律不得写进论文**。

---

## 1. 动机 C 的数字（填 §1 段 3 的 `[TODO: big→small / big→medium confusion fractions]`）

fold 0 基线，行归一化混淆矩阵（真实 big 的像元都去哪了）：

| 真实 big → 预测 | water | cake | small | medium | **big** | vast |
|---|---|---|---|---|---|---|
| 基线 (a) | 0.1 | 0.3 | **1.5** | **16.7** | 65.8 | 15.5 |
| PGSG (c) | 0.1 | 0.3 | **0.2** | **16.4** | 68.7 | 14.3 |

**写法必须准确，不要夸大**：基线只有 **1.5%** 的 big 像元被判成 small、**16.7%** 被判成 medium（合计 18.3%）；同时还有 **15.5%** 被判成 vast。也就是说 big 的错误是**朝尺度轴两侧的相邻类溢出**，不是"大面积碎成小浮冰"。建议这样写：

> The baseline assigns only 65.8\% of big-floe pixels to the correct class, and the
> errors are spread over the two neighbouring classes on the scale axis: 16.7\% are
> read as medium floe and 15.5\% as vast floe, with a further 1.5\% read as small
> floe. Fine-scale evidence alone cannot tell the interior of a large floe from a
> field of smaller ones.

PGSG 的对应效果（可放 Results）：big→small 从 1.5% 降到 **0.2%**（降幅最干净、方向与机制一致），big→medium 基本不变（16.7→16.4），big 的召回率 65.8→68.7。

**顺带一个对动机 B 很有用的数**（真实 medium 的去向，基线）：water 8.2 / cake 10.2 / **small 38.1** / **medium 40.4** / big 0.7 / vast 2.5。中浮冰的召回只有 40.4%，而 **38.1% 被判成小浮冰**——几乎打平。这正是难类瓶颈的量化表达，比"medium IoU 只有 0.229"直观得多。

---

## 2. Fig. 5 混淆矩阵（新增图，`docs/figs/fig5_confusion.pdf`）

两联图：(a) 基线 / (c) PGSG，行归一化百分比。图注建议：

> Fig. 5. Row-normalised confusion matrices on fold 0 (percentage of each true
> class's pixels). Confusion is concentrated in the band adjacent to the diagonal:
> errors move a class to its neighbour on the scale axis rather than to an
> arbitrary class. Open water and vast floe are essentially solved (98--99\%),
> while cake, small and medium floe exchange pixels heavily.

可从图里直接读出、正文可引用的事实：

- 对角带结构明显——**尺度上的相邻类互混**，不是随机错误。这支持"任务是尺度推断"的表述（动机 A）。
- cake 只有 25%（基线）/ 22%（PGSG）判对，最大去向是 small（39% / 38%）。
- small 判对 43% / 46%，最大误判去向是 medium（29% / 30%）。
- water 98%、vast 98–99%，两类已解决。

---

## 2a. 动机 B 的辐射敏感度（填 §1 段 3 的 `[TODO: Δhard-ice]`）

fold 0 基线，83 景全量，`_eval_radiometric.py`（只动 HH/HV 的有效像元，极化比同步重算）：

| 扰动 | macro mIoU | hard-ice | Δhard-ice | water | cake | small | medium | big | vast |
|---|---|---|---|---|---|---|---|---|---|
| 无扰动 | 0.5031 | 0.2243 | — | .969 | .081 | .220 | .229 | .650 | .870 |
| $b{+}0.15$（≈$+1$ dB）| 0.4934 | 0.2247 | +0.000 | .961 | .082 | .228 | .221 | .641 | .827 |
| $b{-}0.15$ | 0.4902 | 0.2066 | −0.018 | .967 | .067 | .198 | .216 | .643 | .851 |
| **$b{+}0.30$（≈$+2$ dB）** | **0.4360** | **0.1857** | **−0.039** | .936 | .082 | .206 | .166 | .599 | .628 |
| **$b{-}0.30$（≈$-2$ dB）** | **0.4602** | **0.1700** | **−0.054** | .960 | .055 | .167 | .173 | .583 | .823 |
| $a=0.90$（对比度）| 0.5004 | 0.2132 | −0.011 | .967 | .087 | .215 | .212 | .655 | .867 |
| $a=1.10$ | 0.5040 | 0.2335 | +0.009 | .969 | .078 | .223 | .244 | .642 | .867 |

**正文用数**：整体平移 $+2$ dB 使 hard-ice 下降 **0.039**，$-2$ dB 下降 **0.054**；
作为对照，同一基线两个随机种子之间的 hard-ice 差只有 0.005。

两点写作时要注意：

1. **$\pm1$ dB 几乎无影响**（+0.000 / −0.018），效应到 $\pm2$ dB 才明显 ⇒ 不要写成"对辐射
   任意扰动都敏感"，准确说法是"a shift of about 2 dB"。
2. **macro 掉得比 hard-ice 更狠**，而且方向不同：$+2$ dB 时巨浮冰从 .870 塌到 **.628**（macro −0.067），
   $-2$ dB 时反而是难类受损更多。也就是说辐射扰动不是只伤难类——如果正文只讲 hard-ice，
   会漏掉"整体都在退化"这个更强的事实。

---

## 2b. Fig. 2 辐射动机图（`docs/figs/fig2_radiometric_*.pdf`）——动机 B 的视觉证据

场景 `20230612T080459`（fold 0 验证集）。四栏：参考标签 ｜ 输入整体 −0.7 z-unit ｜ 原样 ｜ +0.7 z-unit。
**灰度底图四栏完全相同**（都是未扰动的 HH），唯一变化是喂给模型的输入被加了一个常数。

| 操作 | 该景 medium IoU |
|---|---|
| $-0.7$ z-unit（约 $-4$ dB） | **0.03** |
| 原样 | **0.76** |
| $+0.7$ z-unit | 0.78 |

调暗之后，整条中浮冰带被读成**开阔水面**。冰一点没变，变的只是一个加性常数。

选景是扫描出来的（10 个中浮冰支撑 ≥2 Mpx 的场景 × 3 个操作点），翻转幅度排名前几位：

| swing | $-0.7$ / 原样 / $+0.7$ | 场景 |
|---|---|---|
| 0.761 | 0.095 / 0.334 / 0.856 | 20230612T080359 |
| **0.744** | **0.031 / 0.760 / 0.775** | **20230612T080459（出图用这个）** |
| 0.695 | 0.015 / 0.025 / 0.710 | 20230612T174743 |
| 0.615 | 0.000 / 0.615 / 0.083 | 20230709T181128 |

⚠️ 选 `080459` 的理由要说清楚：它**原样时本来就做得好（0.76）**，所以"崩掉"不能推给"这景本来就难"。
另外注意 `20230709T181128` 是两个方向都崩（0.615 → 0.000 / 0.083），说明并非"越亮越好"，
而是**模型只在训练时见过的辐射水平附近才工作**——这一点如果写进正文会让动机 B 更完整。

图注建议：

> Fig. 2. Scale reading is entangled with the absolute radiometric level. All four panels show
> the same Sentinel-1 scene and the same grey-scale backdrop; only a constant was added to the
> model input. (a) reference labels, dominated by a tongue of medium floe. (b) With the input
> darkened by 0.7 z-units (about 4 dB), the entire tongue is read as open water and the
> medium-floe IoU for this scene falls from 0.76 to 0.03. (c) unmodified. (d) Brightened by the
> same amount. Nothing about the ice has changed.

---

## 2c. ⚠️ 初稿动机 B 的第三个分句**被数据否掉，必须删**

初稿 §1 段 3 写道：

> …and in radiometrically dark scenes true medium floe is systematically read as small floe
> ([TODO: misclassification fraction]).

**这个 `[TODO]` 不要填，整个分句要删掉**。用 fold 0 全部 83 景做系统分组（按逐景 HH 均值分出
"最暗的一批"与其余，`_paper_t0_darkscenes.py`），结果与该说法相反：

| 暗端取多少 | 暗端景数 | 暗端 medium→small | 其余 medium→small | 暗端 medium 召回 | 其余 medium 召回 |
|---|---|---|---|---|---|
| 最暗 15% | 13 | **9.6%** | 40.8% | **76.0%** | 37.1% |
| 最暗 25% | 21 | 35.7% | 38.8% | **51.3%** | 37.5% |
| 最暗 35% | 29 | 36.7% | 38.9% | 44.4% | 38.1% |
| 最暗 50% | 41 | 36.5% | 40.2% | 45.6% | 33.7% |

**四个切分点一致**：暗景的 medium→small 比例**不高于**其余场景，中浮冰召回反而**更高**。
逐景看也没有单调关系——最暗的两景（HH 均值 −2.39 / −2.03）确实有 71%/82% 判成小浮冰，
但它们的支撑只有 0.16 / 0.11 Mpx；紧随其后的几景（−1.69 / −1.62 / −1.57）却是 0.1% / 12.7% / 12.6%，
召回 77% / 87% / 84%。medium→small 是**场景特异**的失效，不是按亮度排序的。

**为什么早期有过相反的印象**：那条观察来自 val_417 时代手工挑出的 3 个暗景，是选择性取样；
现在在 fold 0 上做全量系统分组就不成立了。

**这不影响动机 B 本身** —— 动机 B 的支撑是**受控扰动**（§2a 的 Δhard-ice 表 + Fig. 2 的逐景翻转），
那是干预证据，比"暗景表现差"这种观察性相关强得多。删掉这个分句反而让论证更干净：

> 建议改写：…and a uniform shift of about 2 dB applied to a scene lowers hard-ice mIoU by
> 0.04--0.05, an order of magnitude more than the 0.005 difference between two random seeds
> of the same baseline. Fig. 2 shows what this looks like on a single scene.

⚠️ 顺带：**不要**把"暗景更差"写进 Limitations 或 Discussion 的任何角落——同一份数据在这里也不支持。

---

## 3. 参数量与计算量（填 §4.4 / §4.6）

| 模块 | 参数量 |
|---|---|
| Variogram evidence | 47,360 |
| Polarimetric prior | 4,848 |
| Evidence-conditioned scale gate | 109,060 |
| Scale-axis transformer | 72,832 |
| **合计新增** | **234,100（0.234 M）** |
| 基线总参数 | 24,738,310（24.74 M） |
| PGSG 总参数 | 24,972,410（24.97 M，**+0.95%**） |

**计算量（fvcore 实测，单张 $1\times8\times256\times256$ patch）**：基线 **5.64 GMACs** → PGSG **7.56 GMACs**，
增量 **+1.92 GMACs（+34%）**。

⚠️ **初稿 §4.4 写的 "+~15% FLOPs、+0.07 M 参数" 两个数都要改**：0.07 M 只是尺度轴 transformer 一个模块
（全部新增是 0.234 M），而 +15% 低估了计算量——实测是 **+34%**。计算量的增幅远大于参数量的增幅（+0.95%），
原因是这三个模块都作用在 **stride-4 的大特征图**上（变异函数要对 5 个 lag 各做一次窗口平均，
尺度轴 transformer 是逐像素一条长度为 5 的序列），属于"参数少但算得多"的结构——这一点值得在正文里
直说一句，它是个诚实且有信息量的取舍说明。

正文建议：§4.4 写尺度轴 transformer 的 0.073 M；§4.6 汇总"三个机制合计 0.234 M 参数（+0.95%）、
+1.92 GMACs（+34%）"。

⚠️ 输入通道数在这里有个容易写错的地方：**网络实际吃 8 通道**（7 个 train_variables + 在线派生的
极化比），而论文口径的"7 通道"不计 `global_valid_mask`、但计极化比。两者算的是不同集合，
FLOPs 那一行如果要写张量形状，写 $8\times256\times256$ 才与实测一致；若不想引入这个矛盾，
正文就别写 FLOPs 的输入形状，只写增量百分比。

**顺带拿到的零初始化实证**：新建模型（未训练）第一次前向的读数是
`stage-mean g = [1.000, 1.000, 1.000, 1.000]`、选择熵 `1.386/1.386`（恰为 $\ln 4$ 的均匀上限）、
`‖resid‖/‖feat‖ = 0.0000`——即门控恒等、跨尺度残差为零，与 §4.6 声称的"$t=0$ 时前向与基线逐位相同"
完全一致。这条可以作为 §4.6 那句断言的直接佐证。

---

## 4. Fig. 6 组件读数（`docs/figs/fig6_readouts.pdf` + `.csv`）——**已升级为 (b) vs (c) 双配置**

2026-09-18 重画：原来只有主模型 (c)，现在把 **(b) + evidence gate** 一并解析进来（(b) 的训练
日志本就在服务器上，纯 CPU 解析，零成本）。四栏 = (b) 门控预算 ｜ (c) 门控预算 ｜ 两者的
选择熵+证据贡献 ｜ (c) 的跨尺度通信。⚠️ (b) 没有尺度轴 transformer，所以第四栏只有一条线，
这是事实不是缺数据。

| 读数 | (b) + evidence gate | (c) PGSG (full) |
|---|---|---|
| 门控 stage-mean $g_s$ | [1.00×4] → [**1.32**, 1.04, 1.03, **0.60**] | [1.00×4] → [1.14, 1.13, 1.02, **0.71**] |
| 选择熵（上限 1.386）| 1.386 → **1.289** | 1.386 → 1.316 |
| 证据/特征激活贡献比 | 0.445 → **0.098** | 0.394 → **0.045** |
| $\lVert\text{resid}\rVert/\lVert\text{feat}\rVert$ | —（无该模块）| 0.000 → **0.493** |
| 尺度→证据注意力份额 | — | 0.209 → 0.235（均匀基线 0.200）|

### ★ 双配置带来的新结论：证据贡献的衰减**不是跨尺度通信造成的**

原来只看 (c) 时，"证据贡献比掉到 0.045"很容易被读成"尺度轴 transformer 把证据的活抢了"。
现在看到 **(b) 在完全没有跨尺度通信的情况下也从 0.445 掉到 0.098** ⇒ 这是**门控自身训练动态
的性质**，与通信模块无关。两件事因此可以分开说：

- 通信模块**确实在工作**（(c) 的残差范数比长到 0.493）；
- 证据**被逐渐弃用**是门控本身的行为，在 (b)、(c) 上同向复现。

这让 Discussion 里那段"未来工作靶子"的指向更准确：要修的是**门控的优化动态**（如何让决策
持续以不变证据为条件），不是去掉或改造通信模块。

⚠️ 一个必须写明的口径差异：**(b) 早停于 e130（8 万次前向），(c) 跑满 200 epoch（16 万次前向）**，
所以两条曲线的终点在不同训练长度上。比较"末值"时要说明这一点；比较"趋势方向"不受影响。

图注建议：

> Fig. 6. Training-time readouts of the added modules for the two ablation rows that
> contain the gate (logarithmic axis; readouts are logged at geometrically spaced
> intervals, and row (b) stops earlier because early stopping triggered at epoch 130).
> Both configurations start from $g \equiv 1$, confirming the zero-initialisation, and
> settle into a mild preference against the coarsest scale. In (c) the cross-scale
> residual grows from zero to about half the norm of the scale features, so the
> communication path is used. The share of the gate's input activation contributed by
> the evidence decreases over training in both rows, that is, independently of whether
> cross-scale communication is present.

⚠️ **两处必须诚实处理，不能挑好看的说**：

1. **选择熵变化很小**（(b) 1.386→1.289、(c) 1.386→1.316，上限就是 1.386）。**不要写成 "the gate learns to select a scale per pixel"**；准确说法是 "a mild, systematic re-weighting away from the coarsest scale"。图里第三栏的纵轴是放大的，正文必须把绝对值和上限一起给出。
2. **证据贡献比在两个配置上都大幅下降**（(b) 0.445→0.098、(c) 0.394→0.045）。这是**对本文主张不利但必须报告**的读数，也正是 §4.7 所说"读数让正负结果都可判定"的实例。建议放 Discussion，写成一个开放问题：

> The evidence share at the gate input decreases over training, from 0.39 to 0.05.
> The gate therefore relies on the evidence mainly early in training; by convergence
> its decisions are driven predominantly by the main features. This is consistent
> with the modest size of the overall gain, and it identifies a concrete target for
> future work: keeping the decision conditioned on the invariant evidence for longer.

---

## 5. Fig. 1 任务示意（`docs/figs/fig1_task.pdf`）

三栏：(a) HH 灰度 ｜ (b) 标签叠色 ｜ (c) 六类在物理尺度轴上的划分（含 80 m 像元的虚线）。

场景与位置（图注用）：`S1A_EW_GRDM_1SDH_20240607T200401_20240607T200501_054222_06984F_A4E6`，tile 原点 (row 2048, col 1536)，512×512 像元 = **41.0 × 41.0 km**；该 tile 的类别构成为 water 33.2% / small 49.5% / medium 17.2%。

图注建议：

> Fig. 1. The task. (a) A $41\times41$ km Sentinel-1 EW HH subset of the marginal ice
> zone. (b) The corresponding floe size label; labels are constant within an
> ice-chart polygon, so the supervision is regional rather than per-floe. (c) The six
> classes partition a physical scale axis; the dashed line marks the 80 m pixel
> spacing, which lies inside the small-floe interval, so individual small and medium
> floes span only about one to six pixels.

---

## 6. Fig. 4 变异函数剖面（`docs/figs/fig4_variogram.pdf`）

每条曲线是一个类别的**中位剖面**，阴影是四分位带；每类取 10 个近纯 tile（vast 只有 9 个），且一景最多取一个，所以带宽反映的是**跨景**离散度。

(a) 原始 $\log_{10}\gamma_r$：四分位带很宽、且各类严重交叠——幅度主要由场景的辐射条件决定，不由浮冰尺寸决定；彩色虚线是施加 $f\mapsto1.3f+0.8$ 后的同一批剖面，整体被抬起。
(b) 归一化剖面 $v_r$：黑色虚线是仿射变换后的剖面，**严丝合缝压在彩色中位线上**，实测 $\max|v-v'| = 4\times10^{-5}$。

**新增可报告的发现：归一化剖面随类别单调有序**（各类中位数）：

| lag | 0.32 km | 0.64 km | 1.28 km | 2.56 km | 5.12 km |
|---|---|---|---|---|---|
| Small | −1.008 | −0.452 | −0.075 | 0.254 | **0.563** |
| Medium | −0.910 | −0.345 | 0.027 | 0.245 | 0.404 |
| Big | −0.757 | −0.218 | 0.073 | 0.219 | 0.331 |
| Vast | −0.759 | −0.227 | 0.067 | 0.245 | 0.333 |

剖面的**陡峭程度随浮冰尺寸单调下降**：小浮冰最陡（最细 lag 处最低、最粗 lag 处最高），中浮冰次之，大/巨浮冰最平。这与"剖面形状编码相关长度"的设计意图方向一致，可以作为一句实证支持。

⚠️ 但**三条边界必须守住**：

1. **四分位带大面积交叠** ⇒ 只能说 "the class medians order monotonically"，**不得**说 "the profiles separate the classes" 或暗示单凭剖面即可分类。
2. **big 与 vast 在这组 lag 上几乎完全重合**（0.32 km 处 −0.757 vs −0.759）——两者的粒度都远大于最粗 lag 5.12 km，剖面无从区分。如提及，要如实写。
3. 这是**输入 HH** 上的示意，不是模块实际输入的 encoder 特征；图注必须写明。

图注建议：

> Fig. 4. The construction of Section 4.2 applied to the $4\times$-pooled HH channel,
> which shares the 320 m grid of the stride-4 encoder features on which the module
> actually operates. Lines are class medians over ten near-pure tiles (nine for vast
> floe), each from a different scene; shading is the interquartile range. (a) The raw
> structure function. The spread within a class is large and the classes overlap,
> because the amplitude is set mainly by the radiometric level of the scene; an
> affine change $f \mapsto af + b$ (coloured dashed) displaces a profile bodily.
> (b) After normalisation across lags, the transformed profiles (black dashed)
> coincide with the originals to $4\times10^{-5}$, so only the shape survives. The
> median profiles then order monotonically with floe size --- steepest for small floe
> and flattest for big and vast floe, which are indistinguishable at these lags ---
> although the per-tile ranges still overlap.

---

## 7. 本轮暴露的一处口径统一（已在补充材料 #1 改好）

数据集的类别像元数有两种口径：tile 索引口径（保留 tile 内，总 11.10 Gpx）与评估口径（协议下实际参与评估的有效像元，总 11.25 Gpx），差约 1.3%。补充材料 #1 的 Table 1 已统一到**评估口径**，与 §2.1 的区域表同源。**论文里不要混用两种口径。**

---

## 7b. Fig. 3 架构图（`docs/figs/fig3_architecture.pdf`）

matplotlib 绘制（不是 TikZ，本地无 LaTeX）。上排蓝色主通路、下排红色虚线证据通路；证据只有
两条出边，分别进入尺度轴 transformer 的证据 token 与门控输入。**图上标了两个零初始化点**
（尺度轴 transformer 的输出投影、门控末层卷积），与 §2.2 的更正一致，不是初稿写的三个。

图注建议：

> Fig. 3. PGSG. The main feature path (blue) runs from the input through the MiT-B2 encoder
> to four scale features, which exchange information in a scale-axis transformer and are then
> weighted by a budget-conserving softmax gate before fusion. The evidence path (red, dashed)
> supplies the variogram profile and the polarimetric prior to the two decision modules only;
> it never joins the main feature path. Circled zeros mark the two zero-initialized
> projections, which make the forward pass at initialisation identical to the baseline.

---

## 7c. Fig. 7 定性对比（`docs/figs/fig7_qualitative_*.pdf`）——**两景，一成一败**

选景用逐景 CM 排的（`--rank`，纯 CPU）。

**(1) `20240607T200401`（成功案例，与 Fig. 1 同一景）**：裁剪窗口内像素一致率 **0.700 → 0.808**。
参考标签是"近岸一条中浮冰带、向外海过渡为小浮冰"；基线把中浮冰带大面积判成小浮冰，PGSG 把
这条带找回了相当一部分。这是本文增益最直观的一张。

**(2) `20231119T140527`（失效案例，正是动机 C 的画面）**：一致率 **0.093 → 0.170**。
参考标签整片是大浮冰，**两个模型都把它读成巨浮冰 + 成片的小浮冰碎块**——大浮冰内部纹理被
判成小浮冰，一眼可见。PGSG 的碎块明显少一些，但仍然错得很厉害。
⚠️ 这张图必须**作为失效模式呈现**，不能写成成功案例。它的价值是让读者直接看到动机 C 描述
的失败长什么样，以及本文只做到了部分修复。

### ⚠️ 由选景过程暴露的一条硬事实（必须进 Limitations 和 Results）

`--rank` 显示：**fold 0 的 83 个验证场景里，只有 2 个含有大浮冰**（≥10⁵ 像元）：

| 场景 | 大浮冰像元 | 基线 IoU | PGSG IoU |
|---|---|---|---|
| `20231119T140527` | 28.2 Mpx | 0.061 | 0.142 |
| `20240129T115017` | 49.4 Mpx | 0.999 | 0.998 |

也就是说，**本文最大的单类增益（big +0.026）几乎完全来自一个场景**——另一个场景两个模型
都已经做到 0.999、没有改进空间。加上 §2.1 的区域完全混同（所有大浮冰像元都来自那 19 个
加拿大北极场景），这个增益的支撑薄到必须明说：

> On fold 0 only two validation scenes contain big floe. One is already segmented almost
> perfectly by both models (IoU 0.999 and 0.998), so the aggregate improvement on this class
> is driven by the single remaining scene, where the IoU rises from 0.061 to 0.142 --- both
> values low in absolute terms. The big-floe number should therefore be read as an indication,
> not as an established class-level result.

这也是**五折配对（T1-1）不可省**的又一条理由：单折上这一类只有两个独立场景。

---

## 8. 仍然缺的（必须开卡）

- **Fig. 2** 辐射动机图、**Fig. 7** 定性预测对比图——都要模型前向。
- §1 段 3 动机 B 的三个数（train/val gap、$\Delta$hard-ice、暗景误判比例）。
- FLOPs、硬件与每折耗时。
- **Table 4 五折配对**——最关键的一项，见 `docs/2026-08-12_PAPER_PENDING_EXPERIMENTS.md`。

**硬件与单折耗时（2026-09-18 从链日志取到）**：单张 **NVIDIA RTX 5090**；单折训练 **4 h 01 m**（基线 fold3，早停较早）～ **10 h 32 m**（PGSG fold0，跑满 200 epoch 预算），逐景评估 83 景约 5.5 min。差异来自早停时机而非单 epoch 成本。
