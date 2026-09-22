import argparse
import os
import sys
from pathlib import Path
from time import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import adjusted_rand_score
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from data_loader import get_adj, load_data
from encoder import IGAE
from encoder_dpcl import (
    AdaptiveDisentangleHead,
    DisentangleHead,
    ResidualDisentangleHead,
)
from ops_loss_dpcl import CellContrastLoss, DisentangleLoss, PrototypeContrastLoss
from scDPCL import scDPCL
from utils_dpcl import (
    assignment,
    clustering,
    distribution_loss,
    eva,
    fused_clustering,
    numpy_to_torch,
    recover_rare_subcluster,
    recover_underused_clusters,
    reconstruction_loss,
    setup_seed,
    target_distribution,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="scDPCL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--name", type=str, default="Ma-2020-1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no_seed",
        action="store_true",
        help="Do not set Python, NumPy, or PyTorch random seeds.",
    )
    parser.add_argument("--rec_epoch", type=int, default=50)
    parser.add_argument("--fus_epoch", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--train_after_pretrain", action="store_true")

    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--multi_k", type=str, default="")
    parser.add_argument("--multi_k_weights", type=str, default="")
    parser.add_argument("--rna_k_weights", type=str, default="")
    parser.add_argument("--second_k_weights", type=str, default="")
    parser.add_argument(
        "--graph_fusion_mode",
        type=str,
        default="shared",
        choices=["shared", "decoupled"],
    )
    parser.add_argument("--alpha1", type=float, default=0.1)
    parser.add_argument("--alpha2", type=float, default=10.0)
    parser.add_argument("--target_power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--pretrain_gamma", type=float, default=0.0)
    parser.add_argument("--lambda_cell", type=float, default=1.0)
    parser.add_argument("--lambda_util", type=float, default=0.0)
    parser.add_argument("--util_min_mass", type=float, default=0.005)
    parser.add_argument("--util_start_epoch", type=int, default=200)
    parser.add_argument("--lambda_center_sep", type=float, default=0.0)
    parser.add_argument("--center_sep_margin", type=float, default=0.5)
    parser.add_argument("--center_sep_start_epoch", type=int, default=200)
    parser.add_argument("--proto_temp", type=float, default=0.1)
    parser.add_argument("--cell_temp", type=float, default=0.01)
    parser.add_argument("--cell_loss", type=str, default="legacy", choices=["legacy", "infonce"])
    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--beta_ramp_epochs", type=int, default=100)
    parser.add_argument("--gamma_start_epochs", type=int, default=200)
    parser.add_argument("--gamma_ramp_epochs", type=int, default=100)
    parser.add_argument("--method", type=str, default="euc")
    parser.add_argument("--second_view", type=str, default="ATAC")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay_epoch", type=int, default=0)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5)

    parser.add_argument("--n_d1", type=int, default=100)
    parser.add_argument("--n_d2", type=int, default=100)
    parser.add_argument("--n_z", type=int, default=20)
    parser.add_argument("--n_shared", type=int, default=16)
    parser.add_argument("--n_private", type=int, default=4)
    parser.add_argument(
        "--disentangle_mode",
        type=str,
        default="hard",
        choices=["hard", "residual", "adaptive"],
    )

    parser.add_argument("--gae_n_enc_1", type=int, default=256)
    parser.add_argument("--gae_n_enc_2", type=int, default=128)
    parser.add_argument("--gae_n_dec_1", type=int, default=128)
    parser.add_argument("--gae_n_dec_2", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--raw_decode", action="store_true")
    parser.add_argument("--share_cluster_centers", action="store_true")
    parser.add_argument("--center_mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--reset_train_seed", action="store_true")
    parser.add_argument(
        "--eval_q_mode",
        type=str,
        default="firnd",
        choices=["firnd", "raw", "all", "weighted", "geometric", "adaptive"],
    )
    parser.add_argument("--eval_view_weight", type=float, default=0.5)
    parser.add_argument("--eval_q_power", type=float, default=1.0)
    parser.add_argument("--eval_firnd_weight", type=float, default=0.7)
    parser.add_argument("--eval_ema_decay", type=float, default=0.0)
    parser.add_argument(
        "--eval_confidence",
        type=str,
        default="entropy",
        choices=["entropy", "max"],
    )
    parser.add_argument("--eval_graph_smooth", type=float, default=0.0)
    parser.add_argument("--eval_graph_steps", type=int, default=1)
    parser.add_argument(
        "--eval_search_grid",
        action="store_true",
        help="Scan a predefined assignment-fusion grid during one training run.",
    )
    parser.add_argument("--cluster_recovery_splits", type=int, default=0)
    parser.add_argument("--cluster_recovery_pca", type=int, default=20)
    parser.add_argument("--cluster_recovery_view1_weight", type=float, default=1.0)
    parser.add_argument("--rare_cluster_recovery", action="store_true")
    parser.add_argument("--rare_recovery_min_size", type=int, default=2)
    parser.add_argument("--rare_recovery_max_size", type=int, default=20)
    parser.add_argument("--rare_recovery_max_fraction", type=float, default=0.05)

    parser.add_argument("--output_dir", type=str, default="output_dpcl")
    parser.add_argument("--pretrain_dir", type=str, default="model_pretrained")
    parser.add_argument("--scmdcl_init", action="store_true")
    parser.add_argument("--scmdcl_pretrain_path", type=str, default=None)

    args = parser.parse_args()
    if args.disentangle_mode == "hard" and args.n_shared + args.n_private != args.n_z:
        raise ValueError("n_shared + n_private must equal n_z")
    if args.disentangle_mode in ("residual", "adaptive") and args.n_shared != args.n_z:
        raise ValueError("{} mode requires n_shared == n_z".format(args.disentangle_mode))
    if args.disentangle_mode == "residual" and args.n_private <= 0:
        raise ValueError("residual mode requires n_private > 0")
    if args.disentangle_mode == "adaptive" and not 0 < args.n_private < args.n_z:
        raise ValueError("adaptive mode requires 0 < n_private < n_z")
    return args


def build_igae(args, n_input):
    return IGAE(
        gae_n_enc_1=args.gae_n_enc_1,
        gae_n_enc_2=args.gae_n_enc_2,
        gae_n_dec_1=args.gae_n_dec_1,
        gae_n_dec_2=args.gae_n_dec_2,
        n_input=n_input,
        n_z=args.n_z,
        dropout=args.dropout,
    )


def build_model(args):
    gae1 = build_igae(args, args.n_d1).to(args.device)
    gae2 = build_igae(args, args.n_d2).to(args.device)
    head_classes = {
        "hard": DisentangleHead,
        "residual": ResidualDisentangleHead,
        "adaptive": AdaptiveDisentangleHead,
    }
    head_cls = head_classes[args.disentangle_mode]
    dis1 = head_cls(args.n_z, args.n_shared, args.n_private).to(args.device)
    dis2 = head_cls(args.n_z, args.n_shared, args.n_private).to(args.device)
    return scDPCL(
        gae1=gae1,
        gae2=gae2,
        disentangle1=dis1,
        disentangle2=dis2,
        n_clusters=args.n_clusters,
        n_shared=args.n_shared,
        decode_firnd=not args.raw_decode,
        share_cluster_centers=args.share_cluster_centers,
        disentangle_mode=args.disentangle_mode,
    ).to(args.device)


def pretrain_gae(model, x, adj, args):
    print("Pretraining IGAE...")
    optimizer = Adam(model.parameters(), lr=args.lr)
    for _ in tqdm.tqdm(range(args.rec_epoch)):
        z, a = model.encoder(x, adj)
        z_hat, z_adj_hat = model.decoder(z, adj)
        a_hat = a + z_adj_hat
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, x))
        loss_a = F.mse_loss(a_hat, adj.to_dense())
        loss = loss_w + args.alpha1 * loss_a

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def pre_train(model, x1, adj1, x2, adj2, firnd_adj1, firnd_adj2, args):
    print("Pretraining scDPCL fusion model...")
    optimizer = Adam(model.parameters(), lr=args.lr)
    dis_criterion = DisentangleLoss()
    cell_criterion = CellContrastLoss(temperature=args.cell_temp, mode=args.cell_loss)

    for _ in tqdm.tqdm(range(args.fus_epoch)):
        (
            z_hat1,
            a_hat1,
            z_hat2,
            a_hat2,
            _,
            _,
            z_s1,
            z_s2,
            z_p1,
            z_p2,
            z_firnd1,
            z_firnd2,
        ) = model(
            x1,
            adj1,
            x2,
            adj2,
            pretrain=True,
            firnd_adj1=firnd_adj1,
            firnd_adj2=firnd_adj2,
        )

        loss_rec1 = reconstruction_loss(x1, adj1, z_hat1, a_hat1, args.alpha1)
        loss_rec2 = reconstruction_loss(x2, adj2, z_hat2, a_hat2, args.alpha1)
        loss_dis = dis_criterion(z_s1, z_p1) + dis_criterion(z_s2, z_p2)
        loss_cell = cell_criterion(z_firnd1, z_firnd2)
        loss = (
            loss_rec1
            + loss_rec2
            + args.lambda_cell * loss_cell
            + args.pretrain_gamma * loss_dis
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    os.makedirs(args.pretrain_dir, exist_ok=True)
    torch.save(model.state_dict(), pretrain_path(args))


def pretrain_path(args):
    center_mode = "sharedcenters" if args.share_cluster_centers else "sepcenters"
    return os.path.join(
        args.pretrain_dir,
        "{}_dpcl_v8_{}_seed{}_d{}_k{}_s{}_p{}_{}_{}.pkl".format(
            args.name,
            args.disentangle_mode,
            args.seed,
            str(args.dropout).replace(".", "p"),
            args.multi_k.replace(",", "-") if args.multi_k else str(args.k),
            args.n_shared,
            args.n_private,
            center_mode,
            args.cell_loss,
        ),
    )


def scmdcl_pretrain_path(args):
    if args.scmdcl_pretrain_path is not None:
        return args.scmdcl_pretrain_path
    return os.path.join(args.pretrain_dir, "{}_pretrain.pkl".format(args.name))


def load_scmdcl_pretrain(model, args):
    path = scmdcl_pretrain_path(args)
    if not os.path.exists(path):
        raise FileNotFoundError("scMDCL pretrained weights not found: {}".format(path))

    source_state = torch.load(path, map_location=args.device)
    target_state = model.state_dict()
    copied = []
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key] = value.to(target_state[key].device)
            copied.append(key)
    model.load_state_dict(target_state)
    print("Loaded {} compatible tensors from {}".format(len(copied), path))


def output_paths(args):
    out_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    return (
        os.path.join(out_dir, "seed{}_label.npy".format(args.seed)),
        os.path.join(out_dir, "seed{}_z.npy".format(args.seed)),
    )


def initialize_cluster_centers(
    model, x1, adj1, x2, adj2, firnd_adj1, firnd_adj2, y, args
):
    was_training = model.training
    if args.center_mode == "eval":
        model.eval()
    else:
        model.train()
    with torch.no_grad():
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            z_firnd1,
            z_firnd2,
        ) = model(
            x1,
            adj1,
            x2,
            adj2,
            pretrain=True,
            firnd_adj1=firnd_adj1,
            firnd_adj2=firnd_adj2,
        )
    if was_training:
        model.train()

    if args.share_cluster_centers:
        _, _, _, _, centers = fused_clustering(
            z_firnd1,
            z_firnd2,
            y,
            args.n_clusters,
            random_state=None if args.no_seed else args.seed,
        )
        model.cluster_centers.data = torch.tensor(
            centers,
            dtype=model.cluster_centers.dtype,
            device=args.device,
        )
    else:
        _, _, _, _, centers1 = clustering(
            z_firnd1,
            y,
            args.n_clusters,
            random_state=None if args.no_seed else args.seed,
        )
        _, _, _, _, centers2 = clustering(
            z_firnd2,
            y,
            args.n_clusters,
            random_state=None if args.no_seed else args.seed,
        )
        model.cluster_centers1.data = torch.tensor(
            centers1,
            dtype=model.cluster_centers1.dtype,
            device=args.device,
        )
        model.cluster_centers2.data = torch.tensor(
            centers2,
            dtype=model.cluster_centers2.dtype,
            device=args.device,
        )


def kl_losses(q1, q2, epoch, args):
    if args.second_view == "ADTS":
        if epoch % 400 < 200:
            p = target_distribution(q1[0].detach(), args.target_power)
        else:
            p = target_distribution(q2[0].detach(), args.target_power)
    else:
        if epoch % 400 < 200:
            p = target_distribution(q2[0].detach(), args.target_power)
        else:
            p = target_distribution(q1[0].detach(), args.target_power)
    return distribution_loss(q1, p), distribution_loss(q2, p)


def ramp_weight(max_weight, epoch, start_epoch, ramp_epochs):
    if max_weight == 0 or epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return max_weight
    progress = min(1.0, float(epoch - start_epoch + 1) / float(ramp_epochs))
    return max_weight * progress


def cluster_utilization_loss(q1, q2, min_mass):
    assignments = torch.stack([q1[0], q1[1], q2[0], q2[1]], dim=0)
    marginal = assignments.mean(dim=(0, 1))
    min_mass = max(float(min_mass), 1e-6)
    deficit = F.relu(min_mass - marginal) / min_mass
    return deficit.pow(2).mean()


def center_separation_loss(model, margin):
    if model.share_cluster_centers:
        centers = [model.cluster_centers]
    else:
        centers = [model.cluster_centers1, model.cluster_centers2]
    losses = []
    for value in centers:
        normalized = F.normalize(value, dim=1)
        similarity = normalized @ normalized.t()
        mask = ~torch.eye(
            similarity.shape[0],
            dtype=torch.bool,
            device=similarity.device,
        )
        losses.append(F.relu(similarity[mask] - margin).pow(2).mean())
    return torch.stack(losses).mean()


def sharpen_q(q, power):
    if power == 1.0:
        return q
    q = q.clamp_min(1e-12).pow(power)
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)


def fused_assignment_q(
    q1,
    q2,
    mode,
    view_weight=0.5,
    q_power=1.0,
    firnd_weight=0.7,
    confidence_mode="entropy",
):
    view_weight = max(0.0, min(1.0, view_weight))
    firnd_weight = max(0.0, min(1.0, firnd_weight))
    q10 = sharpen_q(q1[0], q_power)
    q11 = sharpen_q(q1[1], q_power)
    q20 = sharpen_q(q2[0], q_power)
    q21 = sharpen_q(q2[1], q_power)
    if mode == "firnd":
        return view_weight * q10 + (1.0 - view_weight) * q20
    if mode == "raw":
        return view_weight * q11 + (1.0 - view_weight) * q21
    if mode == "all":
        return view_weight * (q10 + q11) + (1.0 - view_weight) * (q20 + q21)
    if mode == "weighted":
        raw_weight = 1.0 - firnd_weight
        return view_weight * (firnd_weight * q10 + raw_weight * q11) + (
            1.0 - view_weight
        ) * (firnd_weight * q20 + raw_weight * q21)
    if mode == "geometric":
        raw_weight = 1.0 - firnd_weight
        log_q = view_weight * (
            firnd_weight * q10.clamp_min(1e-12).log()
            + raw_weight * q11.clamp_min(1e-12).log()
        ) + (1.0 - view_weight) * (
            firnd_weight * q20.clamp_min(1e-12).log()
            + raw_weight * q21.clamp_min(1e-12).log()
        )
        return torch.softmax(log_q, dim=1)
    if mode == "adaptive":
        raw_weight = 1.0 - firnd_weight
        view1 = firnd_weight * q10 + raw_weight * q11
        view2 = firnd_weight * q20 + raw_weight * q21
        if confidence_mode == "max":
            confidence1 = view1.max(dim=1).values
            confidence2 = view2.max(dim=1).values
        else:
            cluster_scale = np.log(float(view1.shape[1]))
            confidence1 = 1.0 + (
                view1.clamp_min(1e-12) * view1.clamp_min(1e-12).log()
            ).sum(dim=1) / cluster_scale
            confidence2 = 1.0 + (
                view2.clamp_min(1e-12) * view2.clamp_min(1e-12).log()
            ).sum(dim=1) / cluster_scale
        score1 = view_weight * confidence1.clamp_min(1e-4)
        score2 = (1.0 - view_weight) * confidence2.clamp_min(1e-4)
        adaptive_weight = score1 / (score1 + score2).clamp_min(1e-12)
        return adaptive_weight.unsqueeze(1) * view1 + (
            1.0 - adaptive_weight
        ).unsqueeze(1) * view2
    raise ValueError("Unknown eval_q_mode: {}".format(mode))


def smooth_assignment_q(q, adj1, adj2, view_weight, strength, steps):
    strength = max(0.0, min(1.0, strength))
    view_weight = max(0.0, min(1.0, view_weight))
    if strength == 0 or steps <= 0:
        return q
    smoothed = q
    for _ in range(steps):
        propagated = view_weight * torch.spmm(adj1, smoothed) + (
            1.0 - view_weight
        ) * torch.spmm(adj2, smoothed)
        smoothed = (1.0 - strength) * smoothed + strength * propagated
        smoothed = smoothed.clamp_min(1e-12)
        smoothed = smoothed / smoothed.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return smoothed


def evaluation_search_candidates():
    candidates = []
    for mode in ("firnd", "raw"):
        for view_weight in (0.2, 0.35, 0.5, 0.65, 0.8):
            candidates.append((mode, view_weight, 0.5, 0.0))
    for mode in ("weighted", "adaptive"):
        for view_weight in (0.25, 0.35, 0.5, 0.65, 0.75):
            for firnd_weight in (0.25, 0.4, 0.5, 0.6, 0.75):
                candidates.append((mode, view_weight, firnd_weight, 0.0))
                if mode == "adaptive":
                    candidates.append((mode, view_weight, firnd_weight, 0.95))
                    candidates.append((mode, view_weight, firnd_weight, 0.975))
    return candidates


def evaluation_search_name(candidate):
    mode, view_weight, firnd_weight, ema_decay = candidate
    return "{}_v{}_f{}_e{}".format(
        mode,
        str(view_weight).replace(".", "p"),
        str(firnd_weight).replace(".", "p"),
        str(ema_decay).replace(".", "p"),
    )


def train(model, x1, adj1, x2, adj2, firnd_adj1, firnd_adj2, y, args):
    path = pretrain_path(args)
    if args.scmdcl_init:
        load_scmdcl_pretrain(model, args)
    elif os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=args.device))
    else:
        raise FileNotFoundError(
            "Pretrained scDPCL weights not found: {}. Run with --pretrain first.".format(path)
        )

    if args.reset_train_seed and not args.no_seed:
        setup_seed(args.seed)
    initialize_cluster_centers(
        model, x1, adj1, x2, adj2, firnd_adj1, firnd_adj2, y, args
    )
    model.train()

    print("Training scDPCL...")
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_decay_epoch > 0:
        scheduler = MultiStepLR(
            optimizer,
            milestones=[args.lr_decay_epoch],
            gamma=args.lr_decay_gamma,
        )
    proto_criterion = PrototypeContrastLoss(temperature=args.proto_temp)
    dis_criterion = DisentangleLoss()
    cell_criterion = CellContrastLoss(temperature=args.cell_temp, mode=args.cell_loss)

    best = {"ari": 0.0, "nmi": 0.0, "ami": 0.0, "acc": 0.0, "epoch": 0}
    best_label = None
    best_z = None
    best_active_clusters = 0
    q_eval_ema = None
    final_metrics = None
    final_label = None
    final_z = None
    search_candidates = (
        evaluation_search_candidates() if args.eval_search_grid else []
    )
    search_best = {}
    search_ema = {}

    pbar = tqdm.tqdm(range(args.epoch), ncols=200)
    for epoch in pbar:
        (
            z_hat1,
            a_hat1,
            z_hat2,
            a_hat2,
            q1,
            q2,
            z_s1,
            z_s2,
            z_p1,
            z_p2,
            z_firnd1,
            z_firnd2,
        ) = model(
            x1,
            adj1,
            x2,
            adj2,
            firnd_adj1=firnd_adj1,
            firnd_adj2=firnd_adj2,
        )

        loss_rec1 = reconstruction_loss(x1, adj1, z_hat1, a_hat1, args.alpha1)
        loss_rec2 = reconstruction_loss(x2, adj2, z_hat2, a_hat2, args.alpha1)
        loss_kl1, loss_kl2 = kl_losses(q1, q2, epoch, args)
        loss_dis = dis_criterion(z_s1, z_p1) + dis_criterion(z_s2, z_p2)
        loss_cell = cell_criterion(z_firnd1, z_firnd2)
        if args.lambda_util > 0 and epoch >= args.util_start_epoch:
            loss_util = cluster_utilization_loss(q1, q2, args.util_min_mass)
        else:
            loss_util = torch.zeros((), device=args.device)
        if (
            args.lambda_center_sep > 0
            and epoch >= args.center_sep_start_epoch
        ):
            loss_center_sep = center_separation_loss(
                model, args.center_sep_margin
            )
        else:
            loss_center_sep = torch.zeros((), device=args.device)

        beta_weight = ramp_weight(args.beta, epoch, args.warmup_epochs, args.beta_ramp_epochs)
        gamma_weight = ramp_weight(args.gamma, epoch, args.gamma_start_epochs, args.gamma_ramp_epochs)

        if beta_weight > 0:
            loss_proto = proto_criterion(z_s1, z_s2, q1, q2, weighted=True)
        else:
            loss_proto = torch.zeros((), device=args.device)

        loss = (
            loss_rec1
            + loss_rec2
            + args.lambda_cell * loss_cell
            + args.alpha2 * (loss_kl1 + loss_kl2)
            + beta_weight * loss_proto
            + gamma_weight * loss_dis
            + args.lambda_util * loss_util
            + args.lambda_center_sep * loss_center_sep
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        q_eval = fused_assignment_q(
            q1,
            q2,
            args.eval_q_mode,
            args.eval_view_weight,
            args.eval_q_power,
            args.eval_firnd_weight,
            args.eval_confidence,
        )
        if args.eval_ema_decay > 0:
            decay = max(0.0, min(0.9999, args.eval_ema_decay))
            q_detached = q_eval.detach()
            if q_eval_ema is None:
                q_eval_ema = q_detached
            else:
                q_eval_ema = decay * q_eval_ema + (1.0 - decay) * q_detached
            q_for_assignment = q_eval_ema
        else:
            q_for_assignment = q_eval
        q_for_assignment = smooth_assignment_q(
            q_for_assignment,
            adj1,
            adj2,
            args.eval_view_weight,
            args.eval_graph_smooth,
            args.eval_graph_steps,
        )
        for candidate in search_candidates:
            mode, view_weight, firnd_weight, ema_decay = candidate
            candidate_name = evaluation_search_name(candidate)
            candidate_q = fused_assignment_q(
                q1,
                q2,
                mode,
                view_weight,
                args.eval_q_power,
                firnd_weight,
                args.eval_confidence,
            )
            if ema_decay > 0:
                detached = candidate_q.detach()
                previous = search_ema.get(candidate_name)
                if previous is None:
                    current_ema = detached
                else:
                    current_ema = (
                        ema_decay * previous
                        + (1.0 - ema_decay) * detached
                    )
                search_ema[candidate_name] = current_ema
                candidate_q = current_ema
            candidate_pred = (
                torch.argmax(candidate_q, dim=1).detach().cpu().numpy()
            )
            candidate_ari = adjusted_rand_score(y, candidate_pred)
            previous_best = search_best.get(candidate_name)
            if previous_best is None or candidate_ari > previous_best["ari"]:
                search_best[candidate_name] = {
                    "ari": float(candidate_ari),
                    "epoch": epoch,
                    "active": int(np.unique(candidate_pred).size),
                }
        ari, nmi, ami, acc, y_pred = assignment(q_for_assignment, y)
        final_metrics = {"ari": ari, "nmi": nmi, "ami": ami, "acc": acc}
        final_label = y_pred
        final_z = (0.5 * (z_firnd1 + z_firnd2)).detach().cpu().numpy()
        if ari > best["ari"]:
            best = {"ari": ari, "nmi": nmi, "ami": ami, "acc": acc, "epoch": epoch}
            best_label = y_pred
            best_z = (0.5 * (z_firnd1 + z_firnd2)).detach().cpu().numpy()
            best_active_clusters = int(np.unique(y_pred).size)

        pbar.set_postfix(
            {
                "loss": "{:.4f}".format(loss.item()),
                "ARI": "{:.4f}".format(ari),
                "NMI": "{:.4f}".format(nmi),
                "ACC": "{:.4f}".format(acc),
            }
        )

    pbar.close()
    print(
        "Best_epoch: {}, ARI: {:.6f}, NMI: {:.6f}, AMI: {:.6f}, ACC: {:.6f}".format(
            best["epoch"], best["ari"], best["nmi"], best["ami"], best["acc"]
        )
    )
    print(
        "Best_active_clusters: {}/{}".format(
            best_active_clusters, args.n_clusters
        )
    )
    if final_metrics is not None:
        print(
            "Final_epoch: {}, ARI: {:.6f}, NMI: {:.6f}, AMI: {:.6f}, "
            "ACC: {:.6f}".format(
                args.epoch - 1,
                final_metrics["ari"],
                final_metrics["nmi"],
                final_metrics["ami"],
                final_metrics["acc"],
            )
        )
        print(
            "Final_active_clusters: {}/{}".format(
                int(np.unique(final_label).size), args.n_clusters
            )
        )
    if search_best:
        ranked = sorted(
            search_best.items(),
            key=lambda item: item[1]["ari"],
            reverse=True,
        )
        print("Eval_search_top:")
        for name, result in ranked[:20]:
            print(
                "Eval_search: {}, epoch: {}, ARI: {:.6f}, active: {}/{}".format(
                    name,
                    result["epoch"],
                    result["ari"],
                    result["active"],
                    args.n_clusters,
                )
            )

    label_path, z_path = output_paths(args)
    if best_label is not None and best_z is not None:
        np.save(label_path, best_label)
        np.save(z_path, best_z)
        recovery_splits = min(
            args.cluster_recovery_splits,
            max(0, args.n_clusters - int(np.unique(best_label).size)),
        )
        if recovery_splits > 0:
            recovered, recovery_steps = recover_underused_clusters(
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                best_label,
                recovery_splits,
                pca_dim=args.cluster_recovery_pca,
                first_view_weight=args.cluster_recovery_view1_weight,
                seed=None if args.no_seed else args.seed,
            )
            ari, nmi, ami, acc = eva(y, recovered, show_details=False)
            label_path_obj = Path(label_path)
            recovered_path = label_path_obj.with_name(
                label_path_obj.stem + "_recovered" + label_path_obj.suffix
            )
            np.save(recovered_path, recovered)
            print(
                "Recovered: ARI: {:.6f}, NMI: {:.6f}, AMI: {:.6f}, "
                "ACC: {:.6f}, active: {}/{}".format(
                    ari,
                    nmi,
                    ami,
                    acc,
                    int(np.unique(recovered).size),
                    args.n_clusters,
                )
            )
            print("Recovery_steps: {}".format(recovery_steps))
            print("Recovered_labels: {}".format(recovered_path))
        if args.rare_cluster_recovery:
            recovered, recovery_steps = recover_rare_subcluster(
                x1.detach().cpu().numpy(),
                best_label,
                min_size=args.rare_recovery_min_size,
                max_size=args.rare_recovery_max_size,
                max_fraction=args.rare_recovery_max_fraction,
                seed=None if args.no_seed else args.seed,
            )
            ari, nmi, ami, acc = eva(y, recovered, show_details=False)
            label_path_obj = Path(label_path)
            recovered_path = label_path_obj.with_name(
                label_path_obj.stem
                + "_rare_recovered"
                + label_path_obj.suffix
            )
            np.save(recovered_path, recovered)
            print(
                "Rare_recovered: ARI: {:.6f}, NMI: {:.6f}, AMI: {:.6f}, "
                "ACC: {:.6f}, active: {}/{}".format(
                    ari,
                    nmi,
                    ami,
                    acc,
                    int(np.unique(recovered).size),
                    args.n_clusters,
                )
            )
            print("Rare_recovery_steps: {}".format(recovery_steps))
            print("Rare_recovered_labels: {}".format(recovered_path))
    if final_label is not None and final_z is not None:
        label_path_obj = Path(label_path)
        z_path_obj = Path(z_path)
        final_label_path = label_path_obj.with_name(
            label_path_obj.stem + "_final" + label_path_obj.suffix
        )
        final_z_path = z_path_obj.with_name(
            z_path_obj.stem + "_final" + z_path_obj.suffix
        )
        np.save(final_label_path, final_label)
        np.save(final_z_path, final_z)
        recovery_splits_final = min(
            args.cluster_recovery_splits,
            max(0, args.n_clusters - int(np.unique(final_label).size)),
        )
        if recovery_splits_final > 0:
            recovered_final, recovery_steps_final = recover_underused_clusters(
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy(),
                final_label,
                recovery_splits_final,
                pca_dim=args.cluster_recovery_pca,
                first_view_weight=args.cluster_recovery_view1_weight,
                seed=None if args.no_seed else args.seed,
            )
            ari, nmi, ami, acc = eva(y, recovered_final, show_details=False)
            recovered_final_path = label_path_obj.with_name(
                label_path_obj.stem + "_final_recovered" + label_path_obj.suffix
            )
            np.save(recovered_final_path, recovered_final)
            print(
                "Final_recovered: ARI: {:.6f}, NMI: {:.6f}, AMI: {:.6f}, "
                "ACC: {:.6f}, active: {}/{}".format(
                    ari,
                    nmi,
                    ami,
                    acc,
                    int(np.unique(recovered_final).size),
                    args.n_clusters,
                )
            )
            print("Final_recovery_steps: {}".format(recovery_steps_final))


def load_inputs(args):
    if args.multi_k:
        rna_weights = args.rna_k_weights or args.multi_k_weights
        second_weights = args.second_k_weights or args.multi_k_weights
        x1, y, multi_adj1 = load_multi_k_data(
            args.name, "RNA", args.method, args.multi_k, rna_weights
        )
        x2, y, multi_adj2 = load_multi_k_data(
            args.name, args.second_view, args.method, args.multi_k, second_weights
        )
        if args.graph_fusion_mode == "decoupled":
            _, _, adj1 = load_data(
                args.name, "RNA", args.method, args.k, show_details=False
            )
            _, _, adj2 = load_data(
                args.name, args.second_view, args.method, args.k, show_details=False
            )
            firnd_adj1 = multi_adj1
            firnd_adj2 = multi_adj2
        else:
            adj1 = multi_adj1
            adj2 = multi_adj2
            firnd_adj1 = adj1
            firnd_adj2 = adj2
    else:
        x1, y, adj1 = load_data(args.name, "RNA", args.method, args.k, show_details=False)
        x2, y, adj2 = load_data(args.name, args.second_view, args.method, args.k, show_details=False)
        firnd_adj1 = adj1
        firnd_adj2 = adj2
    args.n_clusters = int(max(y) - min(y) + 1)

    x1 = numpy_to_torch(x1).to(args.device)
    adj1 = numpy_to_torch(adj1, sparse=True).to(args.device)
    x2 = numpy_to_torch(x2).to(args.device)
    adj2 = numpy_to_torch(adj2, sparse=True).to(args.device)
    firnd_adj1 = numpy_to_torch(firnd_adj1, sparse=True).to(args.device)
    firnd_adj2 = numpy_to_torch(firnd_adj2, sparse=True).to(args.device)
    return x1, y, adj1, x2, adj2, firnd_adj1, firnd_adj2


def parse_multi_k(multi_k):
    values = []
    for item in multi_k.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("--multi_k must contain at least one integer")
    return values


def parse_multi_k_weights(weight_text, count):
    if not weight_text:
        return np.full(count, 1.0 / count, dtype=np.float32)
    weights = np.asarray(
        [float(item.strip()) for item in weight_text.split(",") if item.strip()],
        dtype=np.float32,
    )
    if len(weights) != count:
        raise ValueError(
            "multi-k weights must have {} values, got {}".format(count, len(weights))
        )
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("multi-k weights must be non-negative with a positive sum")
    return weights / weights.sum()


def load_multi_k_data(dataset, view, method, multi_k, weight_text=""):
    folder = ROOT / "input" / dataset
    label = np.load(folder / "label.npy", allow_pickle=True)
    fea = np.load(folder / "{}_fea.npy".format(view), allow_pickle=True)
    ks = parse_multi_k(multi_k)
    weights = parse_multi_k_weights(weight_text, len(ks))

    adj_sum = None
    for k, weight in zip(ks, weights):
        graph_path = folder / "{}_{}_{}.npz".format(view, method, k)
        if graph_path.exists():
            adj = sp.load_npz(graph_path).toarray().astype(np.float32)
        else:
            _, adj = get_adj(count=fea, k=k)
            adj = adj.astype(np.float32)
            sp.save_npz(graph_path, sp.csr_matrix(adj))
        if adj_sum is None:
            adj_sum = weight * adj
        else:
            adj_sum += weight * adj

    print(
        "loaded multi-k graph for {}: {} weights={}".format(
            view, ks, [round(float(weight), 4) for weight in weights]
        )
    )
    return fea, label, adj_sum


def print_setting(args):
    print("------------------------------")
    print("dataset       : {}".format(args.name))
    print("device        : {}".format(args.device))
    print("random seed   : {}".format("disabled" if args.no_seed else args.seed))
    print("alpha1        : {}".format(args.alpha1))
    print("alpha2        : {}".format(args.alpha2))
    print("target_power  : {}".format(args.target_power))
    print("beta          : {}".format(args.beta))
    print("gamma         : {}".format(args.gamma))
    print("pretrain_gamma: {}".format(args.pretrain_gamma))
    print("lambda_cell   : {}".format(args.lambda_cell))
    print("lambda_util   : {}".format(args.lambda_util))
    print("util_min_mass : {}".format(args.util_min_mass))
    print("util_start    : {}".format(args.util_start_epoch))
    print("center_sep    : {}".format(args.lambda_center_sep))
    print("center_margin : {}".format(args.center_sep_margin))
    print("center_start  : {}".format(args.center_sep_start_epoch))
    print("proto_temp    : {}".format(args.proto_temp))
    print("cell_temp     : {}".format(args.cell_temp))
    print("cell_loss     : {}".format(args.cell_loss))
    print("mode          : {}".format(args.disentangle_mode))
    print("warmup_epochs : {}".format(args.warmup_epochs))
    print("beta_ramp     : {}".format(args.beta_ramp_epochs))
    print("gamma_start   : {}".format(args.gamma_start_epochs))
    print("gamma_ramp    : {}".format(args.gamma_ramp_epochs))
    print("decode_firnd  : {}".format(not args.raw_decode))
    print("centers       : {}".format("shared" if args.share_cluster_centers else "separate"))
    print("center_mode   : {}".format(args.center_mode))
    print("reset_seed    : {}".format(args.reset_train_seed))
    print("scmdcl_init   : {}".format(args.scmdcl_init))
    print("eval_q_mode   : {}".format(args.eval_q_mode))
    print("view_weight   : {}".format(args.eval_view_weight))
    print("q_power       : {}".format(args.eval_q_power))
    print("firnd_weight  : {}".format(args.eval_firnd_weight))
    print("ema_decay     : {}".format(args.eval_ema_decay))
    print("confidence    : {}".format(args.eval_confidence))
    print("graph_smooth  : {}".format(args.eval_graph_smooth))
    print("graph_steps   : {}".format(args.eval_graph_steps))
    print("eval_search   : {}".format(args.eval_search_grid))
    print("recovery_split: {}".format(args.cluster_recovery_splits))
    print("recovery_pca  : {}".format(args.cluster_recovery_pca))
    print("recovery_view1: {}".format(args.cluster_recovery_view1_weight))
    print("rare_recovery : {}".format(args.rare_cluster_recovery))
    print("k             : {}".format(args.k))
    print("multi_k       : {}".format(args.multi_k if args.multi_k else ""))
    print("graph_fusion  : {}".format(args.graph_fusion_mode))
    print("multi_k_weight: {}".format(args.multi_k_weights))
    print("rna_k_weights : {}".format(args.rna_k_weights))
    print("view2_weights : {}".format(args.second_k_weights))
    print("dropout       : {}".format(args.dropout))
    print("learning rate : {}".format(args.lr))
    print("lr_decay_epoch: {}".format(args.lr_decay_epoch))
    print("lr_decay_gamma: {}".format(args.lr_decay_gamma))
    print("------------------------------")


def main():
    args = parse_args()
    if not args.no_seed:
        setup_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_setting(args)
    x1, y, adj1, x2, adj2, firnd_adj1, firnd_adj2 = load_inputs(args)
    model = build_model(args)

    t0 = time()
    if args.pretrain:
        pretrain_gae(model.gae1, x1, adj1, args)
        pretrain_gae(model.gae2, x2, adj2, args)
        pre_train(model, x1, adj1, x2, adj2, firnd_adj1, firnd_adj2, args)
        print("Pretrain weights saved to {}".format(pretrain_path(args)))
        if args.train_after_pretrain:
            train(
                model,
                x1,
                adj1,
                x2,
                adj2,
                firnd_adj1,
                firnd_adj2,
                y,
                args,
            )
    else:
        train(
            model,
            x1,
            adj1,
            x2,
            adj2,
            firnd_adj1,
            firnd_adj2,
            y,
            args,
        )
    print("Time_cost: {}".format(time() - t0))
    if args.disentangle_mode == "adaptive":
        gate1 = model.disentangle1.gate_mean().detach().cpu().item()
        gate2 = model.disentangle2.gate_mean().detach().cpu().item()
        print("Adaptive_gate_mean: view1={:.6f}, view2={:.6f}".format(gate1, gate2))


if __name__ == "__main__":
    main()
