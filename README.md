# scDPCL 整理版模型

该目录将当前优化模型、复现实验配置和三个数据集的 notebook 集中到一个独立发布包中。

## 目录

```text
scDPCL_release/
  src/          当前模型源码快照
  config/       参数、数据集信息与复现脚本
  tutorial/     BMNC、PBMC-3k、PBMC-10k 运行 notebook
  outputs/      notebook 实验输出
  run.py        统一命令行入口
```

`src/` 是创建本发布包时的模型快照。`run.py` 使用 `config/reproduce.py` 调用项目根目录下的正式模型入口，数据和预训练权重仍从项目原有的 `input/` 与 `model_pretrained/` 读取，避免复制大文件。

## 环境

```powershell
conda activate sedrenv
cd E:\code\scMDCL-main
```

建议先检查命令和输入文件：

```powershell
python scDPCL_release\run.py --dataset PBMC-10k --dry-run
python scDPCL_release\run.py --dataset PBMC-3k --dry-run
python scDPCL_release\run.py --dataset BMNC --dry-run
```

正式运行：

```powershell
python scDPCL_release\run.py --dataset PBMC-10k
python scDPCL_release\run.py --dataset PBMC-3k
python scDPCL_release\run.py --dataset BMNC
```

默认使用 `two_group` 协议：PBMC-10k 使用一套参数，PBMC-3k 与 BMNC 共用另一套参数。具体配置见 `config/TWO_GROUP_PROTOCOL.md` 和 `config/datasets.json`。

## Notebook

在 `sedrenv` 环境中安装并启动 Jupyter：

```powershell
python -m jupyter lab
```

打开 `scDPCL_release/tutorial/` 下对应 notebook。每个 notebook 包含路径与环境检查、数据统计、dry-run、正式训练开关和结果读取。默认 `RUN_TRAINING = False`，确认配置后改为 `True` 再运行训练单元。
