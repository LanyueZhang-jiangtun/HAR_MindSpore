# 基于 MindSpore 的 IMU 传感器人体活动识别（HAR）

四人小组作业骨架。使用华为 **MindSpore** 在 UCI mHealth 公开数据集上，
构建"IMU 原始信号 → 人体活动分类"的完整流水线：数据 → 特征 → 基线 → 深度学习 → 评估 → 部署。

## 环境

- Python 3.11 虚拟环境（已装 MindSpore 2.11.0 / scikit-learn / matplotlib，CPU 模式）：
  `E:/python_ziliao/.venv-ms311/Scripts/python.exe`（下文记为 `PY`）
- 数据集 `mhealth-uci.zip`（72MB，UCI mHealth，doi:10.24432/C5TW22）已存在
  `E:/python_ziliao/a机器学习1122/`，`config.py` 会自动定位，无需复制。

## 运行顺序（按依赖）

| 步骤 | 命令 | 产出 | 负责人 |
|---|---|---|---|
| 1 数据 EDA | `PY scripts/01_eda.py` | 逐被试逐类窗口数表、波形图、谱图 | 甲 |
| 2 基线 | `PY scripts/02_baseline.py` | 4 模型 LOSO 对比表、混淆矩阵、t-SNE（`--feature-set full/time/freq/extended` 特征消融） | 乙 |
| 3 MindSpore | `PY scripts/03_train_cnn.py --epochs 80` | CNN/MLP LOSO 结果、学习曲线、checkpoint（`--augment` 数据增强消融） | 丙 |
| 3b BiLSTM | `PY scripts/03b_train_bilstm.py --epochs 80` | BiLSTM 第二深度基线 LOSO | 丙 |
| 4 报告图 | `PY scripts/04_report_figures.py` | 全部报告图 + 最终对比表 | 丁 |
| 5 手机验证 | `PY scripts/05_phone_test.py` | 真机数据推理、逐文件准确率、phone_test.json（`--no-align` 轴对齐消融） | 丁 |
| 6 跨设备诊断 | `PY scripts/06_cross_device_diag.py` | 强度对比 + RF 特征通路跨设备测试 | 丁 |
| 7 MindIR | `PY scripts/07_export_mindir.py` | cnn_full.mindir（标准化烤入，带冒烟测试） | 丁 |
| 8 误差分析 | `PY scripts/08_error_analysis.py` | 逐被试精度表 + 易混类别对（error_analysis.md） | 乙 |

所有结果输出到 `outputs/`（figures/ 图、results/ JSON 与对比表），可整目录进附录。

## 目录结构与分工

```
HAR_MindSpore/
├── config.py                # 全局配置：路径/窗口/活动映射/训练超参
├── data/phone_labels.json   # 手机文件真实标签（CSV label 列有误时的覆盖映射）
├── src/
│   ├── dataset.py           # [甲] 数据加载、3s 切窗
│   ├── features.py          # [甲] 特征集：full(12)/time(9)/freq(3)/extended(26)
│   ├── baseline.py          # [乙] LR/SVM/RF/GBDT + LOSO 评估
│   ├── models.py            # [丙] MotionMLP（对照）/ MotionCNN（主模型）/ MotionBiLSTM
│   ├── train_mindspore.py   # [丙] LOSO 训练框架（--augment 增强）、学习曲线、checkpoint
│   ├── evaluate.py          # [丁] 对比表（md/csv）、结果读取
│   └── visualize.py         # [丁] 波形/谱图/混淆矩阵/t-SNE/学习曲线
├── scripts/
│   ├── 01_eda.py            # [甲] 入口
│   ├── 02_baseline.py       # [乙] 入口（--feature-set 消融）
│   ├── 03_train_cnn.py      # [丙] 入口（--epochs / --lr / --augment）
│   ├── 03b_train_bilstm.py  # [丙] BiLSTM 基线
│   ├── 04_report_figures.py # [丁] 入口
│   ├── 05_phone_test.py     # [丁] 手机真机验证（--no-align 消融）
│   ├── 06_cross_device_diag.py  # [丁] 跨设备差异定量诊断
│   ├── 07_export_mindir.py  # [丁] MindIR 导出（标准化烤入）
│   └── 08_error_analysis.py # [乙] 逐被试误差分析
└── docs/
    ├── 报告.md              # 正式报告（含全部实测数字）
    └── 报告模板.md
```

## 评估协议（全项目统一，报告的核心卖点）

**留一被试交叉验证（Leave-One-Subject-Out）**：10 个被试，每次用 9 人训练、
剩下 1 人测试。衡量"没见过的被试"上的泛化能力，比随机切分严格得多。

## 手机真机验证（scripts/05_phone_test.py）

模型是在胸戴传感器（mHealth）上训练的，用手机（口袋里）录的数据验证，
检验"跨设备"泛化。处理链：60Hz→50Hz 重采样 → 重力轴对齐（手机重力轴不定，
自动取 |直流| 最大的轴映射到 x 并翻负号）→ 3s 切窗 → 训练集 mean/std 标准化 → 推理。

录制规范（每段一个 CSV，列名固定）：

- 列名：`time_s, ax_m_s2, ay_m_s2, az_m_s2, label, participant`（单位 m/s²）
- **每文件只录一种活动、一个被试**，活动名必须和 `config.USE_LABELS` 的英文名一致
  （当前 7 类：stand / sit / lie / walk / stairs / bike / jog）
- 每类每被试 ≥1 分钟（原协议），手机位置全程固定（建议后裤袋）
- 现有数据：`a机器学习1122/P01_walk_*.csv`（P01 三段，各约 30s；真实活动为
  shake/walk/jog，见 `data/phone_labels.json`）
- **CSV 标签列标错了就在 `data/phone_labels.json` 写 文件名→真实标签 映射**，
  05 会自动覆盖（类外活动如 shake 会计入"类外"而不计准确率）
- 跑法：`PY scripts/05_phone_test.py`（默认扫该目录 P01_*.csv），
  结果在 `outputs/results/phone_test.json`（逐文件准确率 + 预测分布）

已知局限（写报告）：训练数据重力在 x 轴（胸戴），手机重力轴不定（P01 在 z 轴），
轴对齐只能恢复"哪个轴是重力轴"，y/z 相对方向无法恢复；口袋信号还叠加腿部运动。

## 扩展任务状态（工作量的来源）

- **甲**：✅ 特征扩充 26 维（extended）+ 消融（`02 --feature-set`）：extended+SVM
  **83.2%** 为新最佳基线；类别集 7 类（含 stairs）。待做：多传感器通道融合
  （数据集仅胸戴 IMU，需自录）。
- **乙**：✅ 特征消融、✅ 逐被试误差分析（`08` → `error_analysis.md`：S8 全模型最差、
  静态类互混、walk↔stairs 最难）。待做：超参搜索记录表。
- **丙**：✅ GRAPH vs PYNATIVE 对比（6.5s vs 6.0s，精度一致）；✅ BiLSTM 基线
  （`03b`）：**79.76%**，精度≈CNN 但慢 15 倍（5566s vs 362s）；✅ 数据增强消融
  （`03 --augment`）：**64.5%，负效应**（部分折崩溃到随机水平，全批量下增强需配
  更小 lr/更多 epoch，见报告 §7）。待做：AMP 对比、超参搜索。
- **丁**：✅ MindIR 导出（`cnn_full.mindir`，标准化烤入 + 冒烟测试）；✅ 手机验证
  （见下节）；✅ 轴对齐消融（影响可忽略）。待做：补录 7 类手机数据后微调、
  开放集拒绝类。

## 已跑通结果（2026-11-23 实测，LOSO）

**7 类 stand/sit/lie/walk/stairs/bike/jog（1398 窗口，当前默认）：**

| 模型 | Accuracy | Macro-F1 | 备注 |
|---|---|---|---|
| LogisticRegression | 0.7196 | 0.7191 | sklearn，12 维手工特征 |
| SVM | 0.7239 | 0.7213 | sklearn |
| RandomForest | 0.8183 | 0.8175 | sklearn |
| GradientBoosting | 0.7940 | 0.7954 | sklearn |
| **SVM（26 维 extended 特征）** | **0.8319** | **0.8287** | sklearn，**当前最佳** |
| 1D-CNN（原始信号） | 0.7911 | 0.7814 | MindSpore，端到端（80 epoch） |
| MLP（手工特征） | 0.5172 | 0.4406 | MindSpore 对照组 |

特征消融（4 模型最好值）：full(12)=81.83 / time(9)=79.11 / freq(3)=70.03 /
extended(26)=83.19——时频互补，扩充特征有增量。
逐被试：S8 全模型最差（CNN 54%/RF 64%/GBDT 58%），详见 `error_analysis.md`。

**类别粒度消融（5 类 = static/walk/stairs/bike/jog，stand/sit/lie 合并）：**

| 模型 | 7 类 | 5 类 |
|---|---|---|
| sklearn 基线最好值 | 83.19（SVM 26 维） | **99.36**（GBDT 26 维） |
| 1D-CNN | 79.11 | **90.77** |
| MLP（26 维） | 51.72 | 82.62 |

结论：静态类互混是 7 类的主要误差源（标准化抹掉了重力方向线索）；
"静止 vs 运动"场景建议以 5 类为主交付、7 类作细粒度对照（详见报告 §7）。

**3 类 walk/bike/jog（600 窗口，历史对比）：** 基线 0.9983~1.0000，CNN 0.9917，MLP 0.9950。
类别从 3 → 7 后整体难度上升，手工特征基线与 CNN 的差距明显缩小——报告"深度 vs 手工特征"
章节的核心素材。checkpoint 导出 + 恢复一致性验证通过。
结果文件在 `outputs/results/`，全部图在 `outputs/figures/`。

## 常见问题

- **图里中文变方块**：图内标签一律用英文（见 `src/visualize.py` 说明），中文写进报告正文。
- **MindSpore 报 device 错误**：本项目只在 CPU 上跑，确认用的是 `.venv-ms311` 环境的解释器。
- **跑得慢**：LOSO×10 折 + CNN，CPU 上 7 类 `--epochs 80` 约 6 分钟；调试可先 `--epochs 4`。
- **导出 MindIR 报 file_format 错误**：本版本 MindSpore 要求大写 `MINDIR`（不是 "MindIR"）。
- **导出后模型是随机权重（静默失败）**：`load_checkpoint(net=子Cell)` 会因参数名
  前缀不匹配而全不加载、只打 WARNING（坑 4）。`07` 已用"裸 net 预加载 + 冒烟对比"规避，
  改动导出逻辑后务必保留冒烟测试。
- **全量深训发散（NaN）**：300ep/lr=0.02 曾把部署模型训坏（logits NaN，样本内 14.3%）；
  全量训练用默认 80ep/lr=1e-3（样本内 94.4%）。
- **跨设备验证首结果（2026-11-23，P01 三段）**：3 个 CSV 都误标为 walk，真实标签见
  `data/phone_labels.json`（05 会自动用它覆盖 CSV 的 label 列）：
  ① 手拿着摇一摇（类外）② 走 ③ 慢跑。

  | 文件 | 真实 | 运动强度 vs 胸戴同类 | CNN（部署模型 80ep） | RF（模值特征） |
  |---|---|---|---|---|
  | ① 09:56 | shake（类外） | 1.75 m/s²（无参照） | sit×4 stairs×4 stand×1 walk×1 | stand×2 sit×1 walk×1 stairs×4 bike×2 |
  | ② 11:10 | walk | 2.61 = 1.18x 胸戴 walk | jog×6 lie×2 stairs×2（0/10） | walk×3 stairs×7（3/10） |
  | ③ 11:11 | jog | 8.54 = **1.04x** 胸戴 jog | jog×7 stairs×3（**7/10**） | **jog×10（10/10）** |

  结论（报告"跨设备"章节直接用）：
  1. 运动强度跨设备对齐良好（jog 1.04x、walk 1.18x）——口袋与胸戴信号幅度同量级；
  2. **jog 跨设备：CNN 70%、特征通路 100%**——强周期、高幅度活动可迁移；
  3. walk 是弱项（CNN 0% / RF 30%）：弱信号 + 与 stairs/jog 特征重叠；
  4. 类外（摇一摇）被吸收进邻近类，无开放集拒绝——已知局限；
  5. 训练稳定性案例：全量 300ep/lr=0.02 深训**发散**（logits 变 NaN，样本内 14.3%=随机水平），
     部署模型回退默认 80ep/lr=1e-3（样本内 94.4%）——全量长训要控制学习率与步数。
  改进方向：补录手机数据（见上节录制规范）微调 / 特征通路推理 / 增加"拒绝"类。
