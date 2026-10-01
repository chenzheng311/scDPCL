# scDPCL 完整参数

默认方案：`two_group`，与 2026-10-01 的 PBMC-10k 实际运行命令一致。

本表从随包 `src/main_dpcl.py` 的参数解析函数与 `config/reproduce.py` 自动生成，包含所有显式值及隐式默认值。路径以本文件夹为运行根目录；表中的输出路径用于示例，正式运行采用时间戳目录。

`config/parameters/<profile>/<dataset>.json` 包含逐项来源、原始默认值、完整 argv、派生参数与实现固定设置；`config/ALL_PARAMETERS.csv` 仅汇总当前方案的三个数据集配置。

| 参数 | PBMC-10k | PBMC-3k | BMNC |
|---|---|---|---|
| `name` | `"PBMC-10k"` | `"PBMC-3k"` | `"BMNC"` |
| `seed` | `0` | `0` | `0` |
| `no_seed` | `false` | `false` | `false` |
| `rec_epoch` | `50` | `50` | `50` |
| `fus_epoch` | `100` | `100` | `100` |
| `epoch` | `500` | `500` | `500` |
| `pretrain` | `false` | `true` | `true` |
| `train_after_pretrain` | `false` | `true` | `true` |
| `k` | `20` | `10` | `10` |
| `multi_k` | `"15,20,25"` | `""` | `""` |
| `multi_k_weights` | `""` | `""` | `""` |
| `rna_k_weights` | `""` | `""` | `""` |
| `second_k_weights` | `""` | `""` | `""` |
| `graph_fusion_mode` | `"shared"` | `"shared"` | `"shared"` |
| `alpha1` | `0.1` | `0.1` | `0.1` |
| `alpha2` | `7.5` | `10.0` | `10.0` |
| `target_power` | `2.0` | `2.0` | `2.0` |
| `beta` | `0.0` | `0.0` | `0.0` |
| `gamma` | `0.0` | `0.0` | `0.0` |
| `pretrain_gamma` | `0.0` | `0.0` | `0.0` |
| `lambda_cell` | `1.0` | `1.0` | `1.0` |
| `lambda_util` | `0.0` | `0.0` | `0.0` |
| `util_min_mass` | `0.005` | `0.005` | `0.005` |
| `util_start_epoch` | `200` | `200` | `200` |
| `lambda_balance` | `0.0` | `0.0` | `0.0` |
| `balance_start_epoch` | `0` | `0` | `0` |
| `balance_source` | `"all"` | `"all"` | `"all"` |
| `lambda_capacity` | `0.0` | `0.0` | `0.0` |
| `capacity_min_mass` | `0.0` | `0.0` | `0.0` |
| `capacity_max_mass` | `0.0` | `0.0` | `0.0` |
| `capacity_start_epoch` | `0` | `0` | `0` |
| `capacity_source` | `"eval"` | `"eval"` | `"eval"` |
| `lambda_confidence` | `0.0` | `0.0` | `0.0` |
| `confidence_start_epoch` | `0` | `0` | `0` |
| `lambda_teacher` | `0.0` | `0.0` | `0.0` |
| `teacher_start_epoch` | `50` | `50` | `50` |
| `teacher_ema_decay` | `0.99` | `0.99` | `0.99` |
| `lambda_center_sep` | `0.0` | `0.0` | `0.0` |
| `center_sep_margin` | `0.5` | `0.5` | `0.5` |
| `center_sep_start_epoch` | `200` | `200` | `200` |
| `proto_temp` | `0.1` | `0.1` | `0.1` |
| `cell_temp` | `0.01` | `0.01` | `0.01` |
| `cell_loss` | `"legacy"` | `"legacy"` | `"legacy"` |
| `warmup_epochs` | `100` | `100` | `100` |
| `beta_ramp_epochs` | `100` | `100` | `100` |
| `gamma_start_epochs` | `200` | `200` | `200` |
| `gamma_ramp_epochs` | `100` | `100` | `100` |
| `method` | `"euc"` | `"euc"` | `"euc"` |
| `second_view` | `"ATAC"` | `"ATAC"` | `"ATAC"` |
| `lr` | `0.0008` | `0.0008` | `0.001` |
| `lr_decay_epoch` | `0` | `0` | `0` |
| `lr_decay_gamma` | `0.5` | `0.5` | `0.5` |
| `n_d1` | `100` | `100` | `100` |
| `n_d2` | `100` | `49` | `25` |
| `n_z` | `20` | `20` | `20` |
| `n_shared` | `20` | `16` | `16` |
| `n_private` | `0` | `4` | `4` |
| `backbone_type` | `"gae"` | `"gae"` | `"gae"` |
| `disentangle_mode` | `"hard"` | `"hard"` | `"hard"` |
| `gae_n_enc_1` | `256` | `256` | `256` |
| `gae_n_enc_2` | `128` | `128` | `128` |
| `gae_n_dec_1` | `128` | `128` | `128` |
| `gae_n_dec_2` | `256` | `256` | `256` |
| `dropout` | `0.0` | `0.4` | `0.4` |
| `raw_decode` | `false` | `false` | `false` |
| `share_cluster_centers` | `false` | `false` | `false` |
| `center_mode` | `"eval"` | `"train"` | `"train"` |
| `reset_train_seed` | `false` | `false` | `false` |
| `eval_q_mode` | `"adaptive"` | `"adaptive"` | `"adaptive"` |
| `eval_view_weight` | `0.35` | `0.65` | `0.65` |
| `eval_q_power` | `1.0` | `1.0` | `1.0` |
| `eval_firnd_weight` | `0.4` | `0.6` | `0.6` |
| `eval_ema_decay` | `0.975` | `0.95` | `0.95` |
| `eval_confidence` | `"entropy"` | `"entropy"` | `"entropy"` |
| `eval_graph_smooth` | `0.0` | `0.0` | `0.0` |
| `eval_graph_steps` | `1` | `1` | `1` |
| `eval_search_grid` | `false` | `false` | `false` |
| `cluster_recovery_splits` | `2` | `2` | `2` |
| `cluster_recovery_pca` | `20` | `5` | `5` |
| `cluster_recovery_view1_weight` | `1.25` | `3.0` | `3.0` |
| `rare_cluster_recovery` | `false` | `false` | `false` |
| `rare_recovery_min_size` | `2` | `2` | `2` |
| `rare_recovery_max_size` | `20` | `20` | `20` |
| `rare_recovery_max_fraction` | `0.05` | `0.05` | `0.05` |
| `output_dir` | `"outputs/two_group/PBMC-10k"` | `"outputs/two_group/PBMC-3k"` | `"outputs/two_group/BMNC"` |
| `pretrain_dir` | `"model_pretrained"` | `"model_pretrained"` | `"model_pretrained"` |
| `scmdcl_init` | `true` | `false` | `false` |
| `scmdcl_pretrain_path` | `null` | `null` | `null` |

## 实现中固定或数据派生的设置

- 类别数按标签的最大值减最小值加 1 得到：PBMC-10k=19、PBMC-3k=16、BMNC=27。
- 全图训练，float32；编码维度为 输入→256→128→20，解码维度为 20→128→256→输入。
- Adam：betas=(0.9, 0.999)，eps=1e-8，weight_decay=0，amsgrad=False。默认不启用学习率调度。
- KMeans：n_init=10，random_state=0；其余使用归档环境 scikit-learn 1.7.0 的默认值（k-means++、max_iter=300、tol=1e-4、lloyd）。
- Student-t 分配 alpha=1；FIRND 传播为一次邻接矩阵乘法；多图未指定权重时等权平均。
- 预训练日程：每个视图各 50 轮重构，100 轮融合；正式训练 500 轮（日志 epoch 从 0 计数）。PBMC-10k 使用已有 scMDCL 权重，不重新预训练。
- PBMC-10k 默认 n_private=0；全部默认 two_group 配置 beta=gamma=0，相关损失虽然保留在模型中，但权重为零。
- 最佳 epoch 依据真实标签 ARI 选择；恢复步骤受空缺簇数限制，不能把 best/恢复结果称为无标签模型选择。
- 全部模型数值常量及损失实现以随包源码为准；本表覆盖全部 CLI 参数，并补充主要固定设置。

重新导出：`python -m config.export_parameters`。
