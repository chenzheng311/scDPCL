# Two-Group scDPCL Protocol

This protocol uses two fixed parameter groups:

1. PBMC-10k uses the large-PBMC configuration.
2. PBMC-3k and BMNC share one configuration; only learning rate and input
   dimensions differ.

## Group A: PBMC-10k

```text
k / multi_k                   = 20 / 15,20,25
disentangle_mode              = hard
n_shared / n_private          = 20 / 0
alpha2                        = 7.5
beta / gamma                  = 0 / 0
lambda_cell                   = 1
dropout                       = 0
center_mode                   = eval
eval_q_mode                   = adaptive
eval_view_weight              = 0.35
eval_firnd_weight             = 0.4
eval_ema_decay                = 0.975
cluster_recovery_splits       = 2
cluster_recovery_pca          = 20
cluster_recovery_view1_weight = 1.25
learning_rate                 = 0.0008
initialization                = scMDCL pretrained weights
```

Verified result: raw best ARI 0.914055; recovered ARI 0.920613 with 18/19
active clusters.

## Group B: PBMC-3k and BMNC

```text
k                            = 10
disentangle_mode             = hard
n_shared / n_private         = 16 / 4
alpha1 / alpha2              = 0.1 / 10
target_power                  = 2
beta / gamma                 = 0 / 0
lambda_cell                   = 1
cell_loss                     = legacy
dropout                       = 0.4
center_mode                   = train
eval_q_mode                   = adaptive
eval_view_weight              = 0.65
eval_firnd_weight             = 0.6
eval_ema_decay                = 0.95
cluster_recovery_splits       = 2
cluster_recovery_pca          = 5
cluster_recovery_view1_weight = 3
```

Allowed differences:

| Dataset | RNA dim | ATAC dim | Learning rate |
|---|---:|---:|---:|
| PBMC-3k | 100 | 49 | 0.0008 |
| BMNC | 100 | 25 | 0.001 |

Verified PBMC-3k result: raw ARI 0.625453; recovered ARI 0.677099 with
16/16 active clusters.

## Reproduction

```powershell
conda activate sedrenv
cd E:\code\scMDCL-main

python tutorials\scdpcl_reproduction\run_all.py --profile two_group --dry_run
python tutorials\scdpcl_reproduction\run_all.py --profile two_group
```

Individual datasets:

```powershell
python tutorials\scdpcl_reproduction\tutorial_pbmc10k.py --profile two_group
python tutorials\scdpcl_reproduction\tutorial_pbmc3k.py --profile two_group
python tutorials\scdpcl_reproduction\tutorial_bmnc.py --profile two_group
```
