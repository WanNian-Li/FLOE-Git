# 论文插图使用手册（给写作 AI）

> 配套 `docs/2026-08-12_PAPER_WRITING_MANUAL.md`、补充材料 `#1`（2026-09-15）与 `#2`（2026-09-16）。
> 规则照旧：本文件与手册/补充材料之外的数字一律不得写进论文。

## ⚠️ 最重要的一条：你看不到这些图

**图片文件不会提供给你。** 八张图都已画好并由作者保管，你的任务只是：

1. 在正文里插入 `figure` 浮动体，`\includegraphics` 引用**下面给定的确切文件名**；
2. 使用**下面给定的图注**；
3. 写出引用这些图的正文句子。

文件由作者在编译前放好，**你不需要、也不应该**去处理文件路径、目录或图片本身。

由此产生一条硬约束：

🚫 **禁止凭空描述画面。** 你没有见过这些图，任何"如图左上角所示""红色箭头指向……""可以看出
曲线在中段交叉"之类的话都是编造。**只允许使用本文件每张图下"图中有什么"一节列出的事实**，
以及手册/补充材料里的数字。若你想写的某个视觉细节在本文件中没有写明，**就不要写它**。

---

## 1. 文件名清单（写进 `\includegraphics` 的就是这些）

| 文件名 | 图 |
|---|---|
| `fig1_task.pdf` | Fig. 1 任务示意 |
| `fig2_radiometric.pdf` | Fig. 2 辐射动机 |
| `fig3_architecture.pdf` | Fig. 3 PGSG 架构 |
| `fig4_variogram.pdf` | Fig. 4 变异函数剖面 |
| `fig5_confusion.pdf` | Fig. 5 混淆矩阵 |
| `fig6_readouts.pdf` | Fig. 6 组件读数 |
| `fig7a_qualitative_hardclasses.pdf` | Fig. 7 上半：定性·难类改善 |
| `fig7b_qualitative_bigfloe.pdf` | Fig. 7 下半：定性·大浮冰失效 |

统一写成 `\includegraphics[width=\linewidth]{文件名}`，**不要加目录前缀**（作者会用
`\graphicspath` 指定目录）。八张图都是横向宽图，在初稿现有的单栏 `preprint` 模式下
`width=\linewidth` 正合适。

> **以下为作者侧事项，与你无关，列出只是让你知道为什么不用管路径**：作者会把 PDF 放进
> `figures/` 并在导言区加 `\graphicspath{{figures/}}`；若将来切到两栏 `5p` 模式，作者会把
> `figure` 改成 `figure*`。

---

## 2. 逐图：图中有什么 / 代码 / 图注 / 正文引用

每一条的 `\caption{}` 内容**可直接使用**（已按实测数据核对）；正文引用句是建议写法，
可调整措辞但**不得改变结论强度**，也不得加入"图中有什么"以外的视觉描述。

---

### Fig. 1 —— 任务示意（Introduction，动机 A 那一段）

**图中有什么**（你能依据的全部事实）：
- 三栏并列。(a) 一幅 41×41 km 的 Sentinel-1 EW HH 灰度切片，取自边缘冰区；
  (b) 同一切片叠上浮冰尺寸标签的彩色半透明覆盖；(c) 一条对数横轴，六个类别按米数区间
  依次排开，一条竖直虚线标出 80 m 像元。
- 该切片的标签构成：开阔水面 33%、小浮冰 50%、中浮冰 17%（无饼状冰、无大/巨浮冰）。
- 场景 `S1A_EW_GRDM_1SDH_20240607T200401`，tile 原点 (row 2048, col 1536)。

**代码**（替换初稿里 `\label{fig:task}` 那个 placeholder）：

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig1_task.pdf}
\caption{The task. (a) A $41\times41$~km Sentinel-1 EW HH subset of the marginal ice zone.
(b) The corresponding floe size label; labels are constant within an ice-chart polygon, so the
supervision is regional rather than per-floe. (c) The six classes partition a physical scale
axis; the dashed line marks the 80~m pixel spacing, which falls inside the small-floe interval,
so individual small and medium floes span only about one to six pixels.}
\label{fig:task}
\end{figure}
```

**正文引用**（动机 A 段末）：

> At the 80~m pixel spacing of Sentinel-1 EW, an individual small or medium floe spans only
> about one to six pixels (Fig.~\ref{fig:task}).

---

### Fig. 2 —— 辐射动机（Introduction，动机 B 那一段）

**图中有什么**：
- 四栏并列，**四栏的灰度底图完全相同**（都是未扰动的 HH）；变化的只是叠在上面的标签/预测。
- (a) 参考标签，主体是一条中浮冰带；(b)(c)(d) 分别是模型在输入被整体加上 $-0.7$ / $0$ / $+0.7$
  z-unit 之后的预测，每栏标题标出该栏的该景中浮冰 IoU。
- 三个操作点的该景中浮冰 IoU：$-0.7$ 时 **0.03**、原样 **0.76**、$+0.7$ 时 **0.78**。
- 在 $-0.7$ 那一栏，原本的中浮冰带被预测成**开阔水面**。
- 0.7 z-unit 约合 4 dB。该景来自 fold 0 验证集，是扫描出的翻转最剧烈者之一。

**代码**（初稿此处只有一行 `[TODO Fig. 2]` 注释，新建浮动体）：

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig2_radiometric.pdf}
\caption{Scale reading is entangled with the absolute radiometric level. All four panels show
the same scene and the same grey-scale backdrop; only a constant was added to the model input.
(a) Reference labels, dominated by a tongue of medium floe. (b) With the input darkened by
0.7~z-units (about 4~dB), the tongue is read as open water and the medium-floe IoU for this
scene falls from 0.76 to 0.03. (c) Unmodified input. (d) Brightened by the same amount.
Nothing about the ice has changed.}
\label{fig:radiometric}
\end{figure}
```

**正文引用**（动机 B）：

> A uniform shift of about 2~dB applied to a scene lowers hard-ice mIoU by 0.04--0.05, an order
> of magnitude more than the 0.005 difference between two random seeds of the same baseline;
> on individual scenes the effect is dramatic (Fig.~\ref{fig:radiometric}).

🚫 **不许写**：
- 不要说"暗景系统性地把中浮冰判成小浮冰"——**该说法已被 fold 0 全量数据否掉**（补充材料 #2 §2c）。
  这张图是**受控扰动**的证据，与"观察到暗景表现差"是两回事。
- 不要写成"±1 dB 就会崩"：±1 dB 的效应在噪声内，要写就写 "about 2 dB"。
- 不要把单景结果说成普遍规律；图注已注明是单景。

---

### Fig. 3 —— PGSG 架构（Method §4.1）

**图中有什么**：
- 单行流程图。上排是主特征通路（蓝色实线箭头）：输入 → MiT-B2 编码器 → 四个尺度特征
  （stride 4/8/16/32，虚线框内）→ 尺度轴 transformer → 证据条件化 softmax 门控 →
  concat 1024 + $1\times1$ conv/BN/ReLU → 分类器（6 类）。
- 下排是证据通路（红色虚线箭头）：编码器的 stride-4 特征 → 变异函数证据（64 通道）；
  输入的极化比通道 → 极化先验（32 通道）；两者汇入证据张量 $E$（96 通道）。
- $E$ **只有两条出边**，分别指向尺度轴 transformer 和门控；**没有任何一条边通向主通路**。
- 两个黄色圆圈标记"0"，位于尺度轴 transformer 与门控的右上角，表示这两处是零初始化投影。
- 底部有三项图例：主特征通路 / 证据（仅作条件）/ 零初始化投影。

**代码**（替换 `\label{fig:arch}` 那个 placeholder）：

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig3_architecture.pdf}
\caption{PGSG. The main feature path (blue) runs from the input through the MiT-B2 encoder to
four scale features, which exchange information in a scale-axis transformer and are then
weighted by a budget-conserving softmax gate before fusion. The evidence path (red, dashed)
supplies the variogram profile and the polarimetric prior to the two decision modules only;
it never joins the main feature path. Circled zeros mark the two zero-initialized projections,
which make the forward pass at initialisation identical to the baseline.}
\label{fig:arch}
\end{figure}
```

**正文引用**（§4.1 末）：

> Fig.~\ref{fig:arch} shows the two paths in separate colours, together with the two
> zero-initialized projections.

🚫 **不许写**：图上是**两个**零初始化点，不是三个。初稿 Introduction 与 §4.6 里凡说"三个模块都
零初始化"的地方都要一并改（补充材料 #1 §2.2 给了替换句）。

---

### Fig. 4 —— 变异函数剖面（Method §4.2）

**图中有什么**：
- 两栏。横轴都是 lag $r$（对数刻度，五个刻度：320 m / 640 m / 1.3 km / 2.6 km / 5.1 km）。
- (a) 纵轴 $\log_{10}\gamma_r$；(b) 纵轴归一化剖面 $v_r$。
- 每个类别一条**中位数**曲线加一条**四分位带**阴影，四个类别：小浮冰、中浮冰、大浮冰、巨浮冰；
  每类取 10 个近纯 tile（巨浮冰 9 个），一景最多取一个。
- (a) 里另有同色虚线，是对同一批 tile 施加 $f\mapsto 1.3f+0.8$ 之后的中位剖面，整体被抬起。
- (b) 里另有**黑色虚线**，是同一仿射变换后的归一化剖面，**与彩色中位线重合**；实测两者最大偏差
  $4\times10^{-5}$。
- (b) 中各类中位数在最细 lag（0.32 km）处依次为：小 −1.008、中 −0.910、大 −0.757、巨 −0.759；
  在最粗 lag（5.12 km）处为：小 0.563、中 0.404、大 0.331、巨 0.333。
  即剖面陡峭程度随浮冰尺寸单调下降，而**大浮冰与巨浮冰几乎重合**。
- 四个类别的四分位带彼此**大面积交叠**。

**代码**（替换 `\label{fig:variogram}` 那个 placeholder）：

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig4_variogram.pdf}
\caption{The construction of Section~\ref{sec:variogram} applied to the $4\times$-pooled HH
channel, which shares the 320~m grid of the stride-4 encoder features on which the module
actually operates. Lines are class medians over ten near-pure tiles (nine for vast floe), each
taken from a different scene; shading is the interquartile range. (a) The raw structure
function. The spread within a class is large and the classes overlap, because the amplitude is
set mainly by the radiometric level of the scene; an affine change $f \mapsto af+b$ (coloured
dashed) displaces a profile bodily. (b) After normalisation across lags, the transformed
profiles (black dashed) coincide with the originals to $4\times10^{-5}$, so only the shape
survives. The median profiles then order monotonically with floe size---steepest for small floe
and flattest for big and vast floe, which are indistinguishable at these lags---although the
per-tile ranges still overlap.}
\label{fig:variogram}
\end{figure}
```

**正文引用**（§4.2 的不变性命题之后）：

> Fig.~\ref{fig:variogram} illustrates both halves of this statement: the raw structure function
> of a given class is displaced by the radiometric level of the scene, while the normalised
> profile is unchanged by the same transformation.

🚫 **不许写**：
- **不得**说"归一化剖面把类别分开了"。中位数确实单调有序，但四分位带大面积交叠——最多只能写
  "the class medians order monotonically"。
- **不得**暗示大浮冰与巨浮冰可分：两者几乎重合。
- 图注里"作用在输入 HH 上、而模块实际作用于编码器特征"这句**必须保留**。

---

### Fig. 5 / 6 / 7 —— 属于 Results，本轮不写正文

⚠️ 这三张图的位置在 Experiments/Results/Ablation，而那几节**不在本轮撰写范围**内。
本轮只把浮动体和图注放进文件即可，正文叙述留给下一轮。若担心"图未被引用"的编译警告，
可先把这三个浮动体整体注释掉。

#### Fig. 5 混淆矩阵

**图中有什么**：两联热图，(a) 基线 / (c) PGSG（沿用消融表的行名），6×6，行=真实类、列=预测类，
按行归一化成百分比，格内标数字，共用一个色标。数值集中在对角线及其相邻带上。

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig5_confusion.pdf}
\caption{Row-normalised confusion matrices on fold~0 (percentage of each true class's pixels).
Confusion is concentrated in the band adjacent to the diagonal: errors move a class to its
neighbour on the scale axis rather than to an arbitrary class. Open water and vast floe are
essentially solved (98--99\%), while cake, small and medium floe exchange pixels heavily.}
\label{fig:confusion}
\end{figure}
```

写 Results 时可引用的数值：基线 true-big 判对 65.8\%，16.7\% 判成 medium、15.5\% 判成 vast、
1.5\% 判成 small；true-medium 召回 40.4\%，38.1\% 判成 small。

#### Fig. 6 组件读数

**图中有什么**：三栏，横轴都是前向次数（对数刻度，约六七个采样点）。
(a) 四条曲线，四个尺度的平均门值 $g_s$，从 1.000 出发，末期为 [1.14, 1.13, 1.02, 0.71]；
(b) 黑色实线为选择熵（1.386 → 1.316，虚线标出均匀上限 1.386），红色虚线为证据贡献比（右轴，
0.394 → 0.045）；(c) 蓝色为跨尺度残差范数比（0.000 → 0.493），浅绿为证据注意力份额
（0.209 → 0.235，点线标出均匀基线 0.200）。

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig6_readouts.pdf}
\caption{Training-time readouts of the added modules (logarithmic axis; readouts are logged at
geometrically spaced intervals). The gate starts from $g \equiv 1$, confirming the
zero-initialisation, and settles into a mild preference against the coarsest scale. The
cross-scale residual grows from zero to about half the norm of the scale features, so the
communication path is used. The share of the gate's input activation contributed by the
evidence decreases over training.}
\label{fig:readouts}
\end{figure}
```

🚫 **不许写**：
- 选择熵只从 1.386 降到 1.316，而均匀上限就是 1.386 ⇒ **不得**写成 "the gate learns to select a
  scale per pixel"；准确说法是 "a mild, systematic re-weighting away from the coarsest scale"，
  并把绝对值与上限一起给出（(b) 的纵轴是放大的，视觉落差被放大了）。
- 证据贡献比从 0.394 降到 0.045 是**对本文主张不利的读数**，必须如实报告；建议放 Discussion
  当作明确的未来工作靶子（措辞见补充材料 #2 §4）。

#### Fig. 7 定性对比（两景，上下两幅）

**图中有什么**：每幅四栏 [HH 灰度 | 参考标签 | 基线预测 | PGSG 预测]，标签与预测都以半透明彩色
叠在同一张 HH 底图上，且只画到参考标签有效的区域。
- 上幅（`fig7a`，边缘冰区场景）：参考是近岸一条中浮冰带、向外海过渡为小浮冰；基线把该带大面积
  判成小浮冰，PGSG 恢复了其中相当一部分。窗口内像素一致率 0.700 → 0.808。
- 下幅（`fig7b`）：参考整片是大浮冰；两个模型都把它判成巨浮冰并夹杂成片的小浮冰斑块，PGSG 的
  斑块较少。窗口内像素一致率 0.093 → 0.170。

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig7a_qualitative_hardclasses.pdf}\\[2pt]
\includegraphics[width=\linewidth]{fig7b_qualitative_bigfloe.pdf}
\caption{Qualitative comparison on two fold-0 validation scenes. Top: a marginal-ice-zone scene
in which the baseline reads most of the medium-floe band as small floe and PGSG recovers a
large part of it (pixel agreement over the displayed window 0.700 versus 0.808). Bottom: a
failure case in which the reference is uniformly big floe and both models read it as vast floe
interspersed with patches of small floe; PGSG produces fewer such patches, but both remain far
from the reference (0.093 versus 0.170).}
\label{fig:qualitative}
\end{figure}
```

🚫 **不许写**：下幅**必须作为失效案例呈现**。两个模型在该景上都错得很厉害，把它写成成功案例是
失实的；它的价值是让读者看到动机 C 所述的失败长什么样、以及本文只做到部分修复。

---

## 3. 一张总表：每张图服务哪个论点

| 图 | 所在节 | 支撑的论点 | 证据强度 |
|---|---|---|---|
| Fig. 1 | Intro（动机 A）| 类别由物理尺度定义；难类只有 1–6 像素 | 示意 |
| Fig. 2 | Intro（动机 B）| 尺度判读与绝对辐射水平纠缠 | **受控扰动，强** |
| Fig. 3 | Method §4.1 | 证据-决策解耦的结构 | 结构图 |
| Fig. 4 | Method §4.2 | 归一化精确去掉幅度、只留形状 | 不变性**强**；类别可分性**弱，不得据此立论** |
| Fig. 5 | Results | 错误集中在尺度轴相邻类 ⇒ 任务是尺度推断 | 强 |
| Fig. 6 | Results | 模块"是否被使用"可判定 | 机制读数，**不是**效果证据 |
| Fig. 7 | Results | 增益与残留失效的可视化 | 定性，各一景 |

---

## 4. 交稿前自检（你这一侧）

- [ ] 四个 `\includegraphics` 的文件名与 §1 表格逐字一致，且**不带目录前缀**；
- [ ] 初稿里 `\figplaceholder` 的**宏定义与全部调用**都已删除；
- [ ] 每个 `\label` 都有至少一处 `\ref`（Fig. 5/6/7 若暂不写 Results，就整体注释掉）；
- [ ] 全文零初始化点一律写**两处**；
- [ ] 动机 B 段落里**没有**"暗景系统性误判"这类说法；
- [ ] Fig. 4 相关正文**没有**"剖面把类别分开"这类说法；
- [ ] **全文没有任何一句描述了本文件"图中有什么"之外的画面细节**。
