# scDPCL

**Disentangled Private-Shared Collaborative Learning for Matched Single-Cell Multi-Omics Clustering**

![scDPCL framework](assets/framework.png)

scDPCL integrates matched RNA and ATAC profiles for single-cell clustering. This repository contains the standalone reproduction package and the current two-group configuration.

## 当前版本

本版本更新于 2026-10-01，**只保留最新正在使用的 `two_group` 参数**：PBMC-10k 单独一组，PBMC-3k 和 BMNC 共用另一组，仅输入维度和学习率不同。

包含完整模型代码、三个数据集的预处理输入、所需预训练权重、全部 88 项命令行参数、运行环境记录、对应实验日志与聚类结果。整个文件夹可以单独复制使用，不依赖上一级项目。

**权重说明：**原程序只保存预训练权重和聚类标签/嵌入，没有保存正式训练后的 best/final 模型权重。本包已归档现有预训练文件，可重新完成训练；此次整理未重跑完整训练。

## 运行

```powershell
git clone https://github.com/chenzheng311/scDPCL.git
cd scDPCL

# 激活已安装 requirements.txt 所列依赖的 Python 3.11 环境
# 本机复现实验使用：conda activate sedrenv

# 检查全部输入并打印命令，不启动训练
python run.py --dataset all --dry-run

# 按当前参数训练单个数据集
python run.py --dataset PBMC-10k
python run.py --dataset PBMC-3k
python run.py --dataset BMNC

# 依次训练三个数据集
python run.py --dataset all
```

从任意工作目录使用 `python <本包绝对路径>/run.py ...` 也可以运行。使用 `src/main_dpcl.py` 的完整命令时，请先切换到本包根目录。

## 完整参数

- [FULL_PARAMETERS.md](FULL_PARAMETERS.md)：三个数据集全部 88 项参数对照表，以及主要固定设置。
- [config/ALL_PARAMETERS.csv](config/ALL_PARAMETERS.csv)：当前三个数据集的完整参数，共 264 行，可用 Excel 打开。
- [PBMC-10k 参数](config/parameters/two_group/PBMC-10k.json)、[PBMC-3k 参数](config/parameters/two_group/PBMC-3k.json)、[BMNC 参数](config/parameters/two_group/BMNC.json)：所有显式值、默认值、参数来源、派生设置及完整 argv。
- [完整训练命令](config/parameters/two_group/commands.txt)：默认值也已展开。
- [当前参数协议](config/TWO_GROUP_PROTOCOL.md)：两组配置的说明。

`config/reproduce.py` 是当前运行配置来源，`src/main_dpcl.py` 是模型参数默认值来源。修改后可执行 `python -m config.export_parameters` 更新参数表。正式运行会在日志目录额外保存该次完整参数 JSON。

| 当前配置 | PBMC-10k | PBMC-3k | BMNC |
|---|---:|---:|---:|
| 细胞数 / 类别数 | 9631 / 19 | 3762 / 16 | 9202 / 27 |
| RNA / ATAC 维度 | 100 / 100 | 100 / 49 | 100 / 25 |
| 学习率 | 0.0008 | 0.0008 | 0.001 |
| shared / private | 20 / 0 | 16 / 4 | 16 / 4 |
| 实际图 k | 15,20,25 等权融合 | 10 | 10 |
| 初始化 | 已有 scMDCL 权重 | 重新 scDPCL 预训练 | 重新 scDPCL 预训练 |

PBMC-10k 同时设置 `k=20` 与 `multi_k=15,20,25`，当前共享图模式实际采用后者。三数据集的 beta、gamma 均为 0，相关损失权重关闭。

## 文件结构

```text
scDPCL/
  run.py                   统一入口，使用当前 two_group 参数
  create_tutorials.py       重新生成三个 notebook
  src/                     实际实验使用的完整 scDPCL 源码
  baseline/                scMDCL 初始化/对照实验相关完整源码
  config/                  当前参数、数据元信息、复现及参数导出脚本
  input/                   PBMC-10k、PBMC-3k、BMNC 的特征、标签和图
  model_pretrained/        五个已有预训练文件及说明
  reference_results/       当前配置对应的三个数据集日志、指标、标签和嵌入
  outputs/                 新运行的日志与结果
  tutorial/                三个数据集的 notebook
  environment/             sedrenv 实际环境记录
  provenance/              原文件来源及 SHA-256
  verification/            检查说明；运行验证后生成报告
  verify_release.py        数据、参数、权重、前向/反向和入口检查
  FILES_SHA256.json        交付文件校验清单
  FULL_PARAMETERS.md       中文完整参数表
  requirements.txt         核心依赖的实际版本
```

`src/main_dpcl.py`、`src/encoder_dpcl.py` 使用当前 `model/` 中实际运行的版本。网络、损失、训练逻辑原样复制；发布入口仅调整路径、保留当前配置并增加参数记录。模型源码与本次本地完整包保持一致。

## 实际环境

已核实的本机环境：Python 3.11.3、PyTorch 2.1.0（CUDA 12.1）、NumPy 1.26.4、SciPy 1.14.1、scikit-learn 1.7.0，显卡 NVIDIA GeForce RTX 4070 Ti SUPER。

新环境使用 Python 3.11，并安装随包依赖：

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python verify_release.py
```

依赖文件固定 Python 包版本；CUDA 构建信息单独记录于 `environment/runtime.json`，不保证不同平台自动安装相同 CUDA 构建。完整已安装包列表见 `environment/installed_packages.json`。Notebook 另需安装 Jupyter，CLI 不需要。

## 当前配置对应结果

| 数据集 | best ARI | 恢复后 ARI | 报告结果有效簇 |
|---|---:|---:|---:|
| PBMC-10k | 0.914055 | 0.920613 | 18/19 |
| PBMC-3k | 0.625453 | 0.677099 | 16/16 |
| BMNC | 0.742072 | — | 27/27 |

记录在 `reference_results/<dataset>/`，每个数据集包含 `training.log`、`summary.json` 和现有 `seed0_*.npy`。

PBMC-10k 来自今天的实际运行；PBMC-3k/BMNC 来自与当前参数完全相同的既有共享配置验证。来源见 `reference_results/index.json`。原始日志和 summary 保留当时的路径及运行名称，它们是记录而非执行依赖。

best epoch 依据真实标签 ARI 选择，不能描述为完全无监督的模型选择。PBMC-10k 原始 best 标签为 16/19 簇，恢复后为 18/19。

## 输出与核验

正式运行输出到 `outputs/reproduction_selected_<时间>/`，含命令、完整参数 JSON、日志、summary 以及 `outputs/<dataset>/<dataset>/seed0_*.npy`。为兼容原模型，数据集目录嵌套两层。

PBMC-3k/BMNC 的当前配置会重新预训练并覆盖本包 scDPCL 缓存。预训练缓存可能曾被其他实验覆盖，因此不能仅凭文件名把它归属于某次历史结果；本包按源文件保存哈希并检查权重兼容性。

`python verify_release.py` 检查源码语法、当前三组参数、数据形状、权重兼容性、8 个模拟细胞的前向/反向计算、从外部工作目录启动，以及归档标签对应的 ARI。完整训练未重跑。`--check-hashes` 额外核对交付时文件哈希；修改源码或重训覆盖缓存后，相关哈希会变化。校验清单排除运行输出、字节码缓存和可重生成的 verification 报告。

附带的 scMDCL 基线可用 `python run.py --dataset all --mode original --reuse-original-pretrain` 运行。其输出在本包 `output/<dataset>/`，随机种子行为沿用原代码；重做基线预训练会覆盖本包初始化权重，需保留时请先备份本包。

## 版本记录

本次更新将运行入口改为仓库内独立路径，补齐三个数据集、五个预训练文件、三个 notebook 和全部 264 条参数记录。当前只提供 `two_group` 方案。源文件对应关系见 `provenance/source_files.json`，交付核验见 [DELIVERY_CHECKS.md](DELIVERY_CHECKS.md)。
