# 验证报告

在仓库根目录运行 `python verify_release.py`，检查数据、当前参数、权重加载、模拟细胞前向/反向计算、独立入口及归档聚类指标。结果写入本目录的 `report.json` 和 `dry_run_*.txt`，这些机器相关的生成文件不提交到 Git。

`python verify_release.py --check-hashes` 同时验证随版本交付的 `FILES_SHA256.json`。该检查不启动完整数据训练。
