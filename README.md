# 基于 MindSpore 的人体活动识别

本项目使用可穿戴设备记录的三轴加速度信号，识别一个人正在站立、步行、骑车等活动。项目包含数据预处理、传统机器学习基线、MindSpore 深度学习模型、留一被试评估和 MindIR 导出流程。

## 项目概况

- **数据集：** [UCI mHealth Dataset](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)，10 名参与者的胸戴传感器数据，采样率 50 Hz。
- **当前分类：** stand（站立）、sit（坐）、lie（躺）、walk（步行）、stairs（爬楼梯）、bike（骑车）、jog（慢跑）。
- **输入：** 每个样本是 3 秒的三轴加速度信号，即 `(3, 150)`。
- **评估方式：** 留一被试交叉验证（LOSO）。每轮留出一位参与者作为测试集，其余参与者用于训练，测试模型对新参与者的泛化能力。
- **主要模型：** 手工特征 + scikit-learn 分类器，以及 MindSpore 1D-CNN、MLP 和 BiLSTM。

## 实验结果

以下为仓库当前记录的 7 类实验结果。手工特征基线中，扩展特征 + SVM 表现最好；深度学习模型使用原始信号的结果也列在这里作比较。

| 模型 | 输入 | Accuracy | Macro-F1 |
|---|---|---:|---:|
| SVM | 26 维扩展特征 | **83.19%** | **82.87%** |
| Random Forest | 12 维基础特征 | 81.83% | 81.75% |
| BiLSTM | 原始信号 | 79.76% | 79.80% |
| 1D-CNN | 原始信号 | 79.11% | 78.14% |

这组结果来自 10 名参与者的留一被试交叉验证，数据集共切出 1,398 个窗口。另有 5 类实验将站立、坐和躺合并为“静态”活动；结果和分析见[项目报告](docs/报告.md)。不同类别定义下的分数不可直接比较。

## 快速开始

### 1. 获取代码和数据

```bash
git clone https://github.com/LanyueZhang-jiangtun/HAR_MindSpore.git
cd HAR_MindSpore
```

从 [UCI mHealth Dataset 页面](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)下载数据压缩包，将文件重命名为 `mhealth-uci.zip` 后放到：

```text
data/mhealth-uci.zip
```

数据集较大，未随代码仓库提交。`config.py` 默认从 `data/mhealth-uci.zip` 读取数据。
如果数据存放在其他位置，可在运行脚本前设置 `HAR_DATASET_ZIP`：

```powershell
$env:HAR_DATASET_ZIP = "D:\datasets\mhealth-uci.zip"
```

macOS / Linux：

```bash
export HAR_DATASET_ZIP="$HOME/datasets/mhealth-uci.zip"
```

### 2. 安装依赖

项目使用 Python 3.11。创建虚拟环境并安装 `requirements.txt` 中列出的依赖：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

依赖版本固定为项目当前验证环境：Python 3.11、MindSpore 2.11.0 CPU 和 NumPy 1.26.4。MindSpore 安装包与操作系统、Python 版本和硬件有关；如果安装失败，请按 [MindSpore 安装指南](https://www.mindspore.cn/install)确认当前平台支持该版本。

### 3. 运行分析和训练

建议先检查数据和生成基础图表，再运行分类基线与深度模型：

```bash
python scripts/01_eda.py
python scripts/02_baseline.py
python scripts/03_train_cnn.py
```

`03_train_cnn.py` 会比较 1D-CNN 与 MLP，并训练一个用于导出的全量 CNN。默认训练轮数为 80；快速检查流程时可减少轮数：

```bash
python scripts/03_train_cnn.py --epochs 4
```

可选实验：

```bash
python scripts/02_baseline.py --feature-set extended
python scripts/03b_train_bilstm.py --epochs 80
python scripts/03_train_cnn.py --augment
```

生成汇总报告图、逐被试误差分析和 MindIR 模型：

```bash
python scripts/04_report_figures.py
python scripts/08_error_analysis.py
python scripts/07_export_mindir.py
```

这些汇总和导出脚本依赖前面脚本生成的结果文件。`05_phone_test.py` 和 `06_cross_device_diag.py` 属于可选的手机跨设备实验，需要另外准备符合脚本要求的手机传感器 CSV 数据。`09_segment_timeline.py` 做长片段活动时间线识别：输入一段长 CSV（或用 `--files` 拼接多个 30 秒片段），输出"第 X 秒到第 Y 秒是什么活动"、逐段置信度和彩色时间线图。

## 输出文件

运行脚本后，生成文件会保存在 `outputs/`：

- `outputs/figures/`：波形、频谱、混淆矩阵和学习曲线。
- `outputs/results/`：模型指标、逐被试预测、训练曲线和导出的模型。

这些运行产物及数据压缩包不会提交到 GitHub。克隆仓库后需要先准备数据并运行相应脚本，才能生成它们。

## 目录结构

```text
HAR_MindSpore/
├── config.py                 # 数据路径、活动类别、窗口长度和训练参数
├── requirements.txt          # Python 依赖
├── data/
│   └── phone_labels.json     # 手机测试文件的标签覆盖配置
├── src/
│   ├── dataset.py            # 读取数据并切分窗口
│   ├── features.py           # 提取时域、频域和扩展特征
│   ├── baseline.py           # scikit-learn 基线与 LOSO 评估
│   ├── models.py             # CNN、MLP 和 BiLSTM
│   ├── train_mindspore.py    # MindSpore 训练与评估
│   ├── evaluate.py           # 结果读取和对比表
│   └── visualize.py          # 图表生成
├── scripts/                  # 可直接运行的分析、训练和导出脚本
└── docs/
    ├── 报告.md               # 实验过程、结果和局限
    └── 报告模板.md
```

## 当前限制与后续方向

- 当前使用 7 类活动，且只读取胸戴传感器的三轴加速度；其他传感器通道尚未融合。
- 手机跨设备测试样本较少（两个被试 P04/P02）。跨被试验证（P02 未参与训练）上限约 **60%**（26 维 RF），远低于同被试微调（93%）——"手机数据 + 重训"的收益是个性化的，抬高上限需多人覆盖全 4 类的数据。
- 位置是主导因素：口袋/背包可用，手持看屏是低幅活动（walk/run）的物理盲区（手机静止看屏时 IMU 看不到腿部运动），可由 GPS 速度补位（IMU+GPS 融合）；bike 最可迁移，stairs 最弱。
- 7 类任务中静态活动较容易混淆；5 类实验合并了站立、坐和躺，适合与 7 类细粒度结果对照。
- 手机测试目前没有开放集拒识能力，训练类别之外的活动（static/摇一摇）可能被自信地误判为已知类别。

后续可优先补充更多参与者和手机采集数据（覆盖 walk/stairs/bike/jog 全 4 类），再评估跨设备泛化；也可以加入多传感器融合、类别不平衡处理和开放集拒识实验。长片段活动时间线（`09_segment_timeline.py`）已实现，低置信度段可接人工复核流程。

## 相关资料

- [完整实验报告](docs/报告.md)
- [UCI mHealth Dataset](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)
- [MindSpore 安装指南](https://www.mindspore.cn/install)
