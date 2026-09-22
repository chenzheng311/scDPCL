# Unified scDPCL Protocol

BMNC, PBMC-3k, and PBMC-10k use one model/training configuration. Only the
input dimensions and learning rate differ.

## Shared parameters

```text
seed                         = 0
k                            = 10
n_z                          = 20
disentangle_mode             = hard
n_shared                     = 16
n_private                    = 4
alpha1                       = 0.1
alpha2                       = 10
target_power                  = 2
beta                         = 0
gamma                        = 0
pretrain_gamma                = 0
lambda_cell                   = 1
cell_loss                     = legacy
warmup_epochs                 = 100
beta_ramp_epochs              = 100
gamma_start_epochs            = 200
gamma_ramp_epochs             = 100
dropout                       = 0.4
center_mode                   = train
eval_q_mode                   = adaptive
eval_view_weight              = 0.65
eval_firnd_weight             = 0.6
eval_q_power                  = 1
eval_ema_decay                = 0.95
cluster_recovery_splits       = 2
cluster_recovery_pca          = 5
cluster_recovery_view1_weight = 3
initialization                = unified scDPCL pretraining
```

Cluster recovery is data-adaptive: it only splits enough clusters to fill
unused cluster slots, up to the shared limit of two. Therefore BMNC performs
zero recovery steps, PBMC-3k performs one, and PBMC-10k performs two.

## Allowed dataset differences

| Dataset | RNA dim | ATAC dim | Learning rate |
|---|---:|---:|---:|
| BMNC | 100 | 25 | 0.001 |
| PBMC-3k | 100 | 49 | 0.0008 |
| PBMC-10k | 100 | 100 | 0.0005 |

Input dimensions describe the data and are not tuned model hyperparameters.

## Final validation

Seed 0 results from the joint validation run:

| Dataset | Raw best ARI | Reported ARI | Active clusters |
|---|---:|---:|---:|
| BMNC | 0.742072 | 0.742072 | 27/27 |
| PBMC-3k | 0.625453 | 0.677099 | 16/16 |
| PBMC-10k | 0.854517 | 0.861893 | 18/19 |

PBMC-3k passes the required ARI threshold of 0.65.

Authoritative logs:

```text
model/experiment_logs/unified_final_20260615/
```

Historical `legacy_tuned` results use different parameters for each dataset and
must not be mixed with this unified protocol.

## Commands

```powershell
conda activate sedrenv
cd E:\code\scMDCL-main

python tutorials\scdpcl_reproduction\run_all.py --profile unified --dry_run
python tutorials\scdpcl_reproduction\run_all.py --profile unified
```

Individual datasets:

```powershell
python tutorials\scdpcl_reproduction\tutorial_bmnc.py --profile unified
python tutorials\scdpcl_reproduction\tutorial_pbmc3k.py --profile unified
python tutorials\scdpcl_reproduction\tutorial_pbmc10k.py --profile unified
```
