import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.preprocessing import StandardScaler


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def numpy_to_torch(a, sparse=False):
    if not sparse:
        return torch.FloatTensor(a)

    rows, cols = np.nonzero(a)
    values = a[rows, cols].astype(np.float32)
    indices = torch.LongTensor(np.vstack((rows, cols)))
    values = torch.FloatTensor(values)
    return torch.sparse_coo_tensor(indices, values, size=a.shape).coalesce()


def reconstruction_loss(x, adj_norm, z_hat, adj_hat, alpha1):
    loss_w = F.mse_loss(z_hat, torch.spmm(adj_norm, x))
    loss_a = F.mse_loss(adj_hat, adj_norm.to_dense())
    return loss_w + alpha1 * loss_a


def target_distribution(q, power=2.0):
    q = q.clamp_min(1e-12)
    weight = q.pow(power) / q.sum(dim=0).clamp_min(1e-12)
    return (weight.t() / weight.sum(dim=1).clamp_min(1e-12)).t()


def recover_underused_clusters(
    x1,
    x2,
    labels,
    split_count,
    pca_dim=20,
    first_view_weight=1.0,
    seed=0,
):
    """Split internally heterogeneous clusters without using ground-truth labels."""
    if split_count <= 0:
        return labels.copy(), []

    first = StandardScaler().fit_transform(np.asarray(x1))
    second = StandardScaler().fit_transform(np.asarray(x2))
    combined = np.concatenate([first * first_view_weight, second], axis=1)
    n_components = min(int(pca_dim), combined.shape[0], combined.shape[1])
    features = PCA(n_components=n_components, random_state=seed).fit_transform(combined)

    recovered = np.asarray(labels).copy()
    steps = []
    for _ in range(split_count):
        candidates = []
        for cluster_id in np.unique(recovered):
            indices = np.flatnonzero(recovered == cluster_id)
            if len(indices) < 20 or len(indices) > 1000:
                continue
            local = KMeans(
                n_clusters=2,
                n_init=20,
                random_state=seed,
            ).fit_predict(features[indices])
            sizes = np.bincount(local)
            if len(sizes) < 2 or sizes.min() < 5:
                continue

            cluster_features = features[indices]
            center = cluster_features.mean(axis=0, keepdims=True)
            sse_before = float(((cluster_features - center) ** 2).sum())
            sse_after = 0.0
            for value in (0, 1):
                part = cluster_features[local == value]
                part_center = part.mean(axis=0, keepdims=True)
                sse_after += float(((part - part_center) ** 2).sum())
            gain = max(0.0, (sse_before - sse_after) / max(sse_before, 1e-12))
            score = gain * len(indices) ** 0.4
            candidates.append((score, int(cluster_id), indices, local, sizes))

        if not candidates:
            break
        score, source, indices, local, sizes = max(
            candidates, key=lambda item: item[0]
        )
        new_id = int(recovered.max()) + 1
        recovered[indices[local == 1]] = new_id
        steps.append(
            {
                "source": source,
                "new": new_id,
                "size0": int(sizes[0]),
                "size1": int(sizes[1]),
                "score": float(score),
            }
        )
    return recovered, steps


def recover_rare_subcluster(
    x,
    labels,
    min_size=2,
    max_size=20,
    max_fraction=0.05,
    seed=0,
):
    """Split the strongest small, non-singleton subcluster without labels."""
    features = StandardScaler().fit_transform(np.asarray(x))
    recovered = np.asarray(labels).copy()
    candidates = []
    for cluster_id in np.unique(recovered):
        indices = np.flatnonzero(recovered == cluster_id)
        if len(indices) < 2 * int(min_size):
            continue
        local = KMeans(
            n_clusters=2,
            n_init=30,
            random_state=seed,
        ).fit_predict(features[indices])
        sizes = np.bincount(local, minlength=2)
        small_label = int(np.argmin(sizes))
        small_size = int(sizes[small_label])
        fraction = small_size / float(len(indices))
        if (
            small_size < int(min_size)
            or small_size > int(max_size)
            or fraction > float(max_fraction)
        ):
            continue
        score = float(silhouette_score(features[indices], local))
        candidates.append(
            (score, int(cluster_id), indices, local, small_label, sizes)
        )

    if not candidates:
        return recovered, []
    score, source, indices, local, small_label, sizes = max(
        candidates, key=lambda item: item[0]
    )
    new_id = int(recovered.max()) + 1
    recovered[indices[local == small_label]] = new_id
    return recovered, [
        {
            "source": source,
            "new": new_id,
            "size0": int(sizes[0]),
            "size1": int(sizes[1]),
            "small_size": int(sizes[small_label]),
            "small_fraction": float(sizes[small_label] / sizes.sum()),
            "silhouette": score,
        }
    ]


def distribution_loss(q_pair, p):
    q0 = q_pair[0].clamp_min(1e-12)
    q1 = q_pair[1].clamp_min(1e-12)
    return F.kl_div((q0.log() + q1.log()) / 2.0, p, reduction="batchmean")


def clustering(z, y, n_clusters, show_details=True, random_state=0):
    model = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_state,
    )
    cluster_id = model.fit_predict(z.detach().cpu().numpy())
    ari, nmi, ami, acc = eva(y, cluster_id, show_details=show_details)
    return ari, nmi, ami, acc, model.cluster_centers_


def fused_clustering(
    z1,
    z2,
    y,
    n_clusters,
    show_details=True,
    random_state=0,
):
    fused = 0.5 * (z1 + z2)
    return clustering(
        fused,
        y,
        n_clusters,
        show_details=show_details,
        random_state=random_state,
    )


def assignment(q, y):
    y_pred = torch.argmax(q, dim=1).detach().cpu().numpy()
    ari, nmi, ami, acc = eva(y, y_pred, show_details=False)
    return ari, nmi, ami, acc, y_pred


def cluster_acc(y_true, y_pred):
    true_values, true_inverse = np.unique(y_true, return_inverse=True)
    pred_values, pred_inverse = np.unique(y_pred, return_inverse=True)
    contingency = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    np.add.at(contingency, (true_inverse, pred_inverse), 1)
    row_indices, col_indices = linear_sum_assignment(-contingency)
    return float(contingency[row_indices, col_indices].sum()) / len(y_true)


def eva(y_true, y_pred, show_details=True):
    acc = cluster_acc(y_true, y_pred.copy())
    nmi = nmi_score(y_true, y_pred, average_method="arithmetic")
    ari = ari_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)

    if show_details:
        print(
            "\n",
            "ARI: {:.4f},".format(ari),
            "NMI: {:.4f},".format(nmi),
            "AMI: {:.4f}".format(ami),
            "ACC: {:.4f},".format(acc),
        )
    return ari, nmi, ami, acc
