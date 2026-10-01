# 已有预训练权重

当前参数包包含五个预训练文件：

- PBMC-10k、PBMC-3k、BMNC 各一份 `<dataset>_pretrain.pkl`：原 scMDCL 初始化权重。当前 PBMC-10k 直接加载它，另外两份用于附带的基线入口。
- PBMC-3k、BMNC 各一份 `*_dpcl_v8_hard_seed0_d0p4_k10_s16_p4_sepcenters_legacy.pkl`：现存 scDPCL 预训练缓存；当前配置会按完整预训练日程重新生成。

原程序没有保存正式训练后的 best/final 模型权重或优化器状态。缓存可能被其他实验覆盖，不能仅凭文件名确认某次日志的权重。来源哈希见 `provenance/source_files.json`，兼容性检查见 `verification/report.json`。
