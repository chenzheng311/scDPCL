import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PrototypeContrastLoss(nn.Module):
    """Weighted cross-modal prototype contrastive loss."""

    def __init__(self, temperature=0.1, eps=1e-8):
        super(PrototypeContrastLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    @staticmethod
    def _primary_q(q):
        return q[0] if isinstance(q, (list, tuple)) else q

    def _weighted_prototypes(self, z_shared, q, weights):
        weighted_q = q * weights.unsqueeze(1)
        denom = weighted_q.sum(dim=0).unsqueeze(1).clamp_min(self.eps)
        prototypes = torch.mm(weighted_q.t(), z_shared) / denom
        return F.normalize(prototypes, dim=1)

    def forward(self, z_s1, z_s2, q1, q2, weighted=True):
        q1 = self._primary_q(q1).detach()
        q2 = self._primary_q(q2).detach()
        q_fused = 0.5 * (q1 + q2)

        if weighted:
            consistency = F.cosine_similarity(q1, q2, dim=1).clamp_min(0.0)
            confidence = q_fused.max(dim=1).values
            weights = consistency * confidence
        else:
            weights = torch.ones(z_s1.shape[0], device=z_s1.device, dtype=z_s1.dtype)

        c1 = self._weighted_prototypes(z_s1, q_fused, weights)
        c2 = self._weighted_prototypes(z_s2, q_fused, weights)

        sim = torch.mm(c1, c2.t()) / self.temperature
        labels = torch.arange(sim.shape[0], device=sim.device)
        loss_12 = F.cross_entropy(sim, labels)
        loss_21 = F.cross_entropy(sim.t(), labels)
        return 0.5 * (loss_12 + loss_21)


class LegacySupConLoss(nn.Module):
    """SupCon implementation used by the original scMDCL code."""

    def __init__(self, temperature=0.01, base_temperature=0.7):
        super(LegacySupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def _chunked_identity_loss(self, anchor_feature, contrast_feature, chunk_size=1024):
        batch_size = anchor_feature.shape[0]
        total = anchor_feature.new_zeros(())

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            global_indices = torch.arange(
                start,
                end,
                device=anchor_feature.device,
            )

            def chunk_loss(
                anchor_chunk,
                full_contrast,
                selected_indices=global_indices,
            ):
                raw = torch.matmul(anchor_chunk, full_contrast.t())
                raw = raw / self.temperature
                logits_min = raw.min(dim=1, keepdim=True).values
                logits_max = raw.max(dim=1, keepdim=True).values
                logits = raw / (
                    logits_max.detach() - logits_min
                ).clamp_min(1e-12)
                row_indices = torch.arange(
                    logits.shape[0],
                    device=logits.device,
                )
                positive_logits = logits[row_indices, selected_indices]
                exp_logits = torch.exp(logits)
                negative_sum = (
                    exp_logits.sum(dim=1)
                    - exp_logits[row_indices, selected_indices]
                ).clamp_min(1e-12)
                return (
                    positive_logits - torch.log(negative_sum)
                ).sum()

            total = total + checkpoint(
                chunk_loss,
                anchor_feature[start:end],
                contrast_feature,
                use_reentrant=False,
            )

        mean_log_prob_pos = total / batch_size
        return -(self.temperature / self.base_temperature) * mean_log_prob_pos

    def forward(self, feature1, feature2=None, labels=None, mask=None):
        device = feature1.device
        batch_size = feature1.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both labels and mask")
        identity_pairs = labels is None and mask is None
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.t()).float().to(device)
        elif mask is not None:
            mask = mask.float().to(device) + torch.eye(
                batch_size, dtype=torch.float32, device=device
            )

        anchor_feature = feature1
        contrast_feature = feature2 if feature2 is not None else feature1
        anchor_count = 1
        if identity_pairs and batch_size > 4096:
            return self._chunked_identity_loss(
                anchor_feature,
                contrast_feature,
            )

        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.t())
        anchor_dot_contrast = anchor_dot_contrast / self.temperature

        logits_min, _ = torch.min(anchor_dot_contrast, dim=1, keepdim=True)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast / (logits_max.detach() - logits_min).clamp_min(1e-12)

        if identity_pairs:
            exp_logits = torch.exp(logits)
            positive_logits = torch.diagonal(logits)
            negative_sum = (
                exp_logits.sum(dim=1) - torch.diagonal(exp_logits)
            ).clamp_min(1e-12)
            mean_log_prob_pos = positive_logits - torch.log(negative_sum)
            loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
            return loss.view(anchor_count, batch_size).mean()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-12)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


class CellContrastLoss(nn.Module):
    """Instance-level cross-modal contrastive loss for paired cells."""

    def __init__(self, temperature=0.01, mode="legacy"):
        super(CellContrastLoss, self).__init__()
        self.temperature = temperature
        self.mode = mode
        self.legacy_loss = LegacySupConLoss(temperature=temperature)

    def forward(self, z_s1, z_s2):
        if self.mode == "legacy":
            return self.legacy_loss(z_s1, z_s2)

        z_s1 = F.normalize(z_s1, dim=1)
        z_s2 = F.normalize(z_s2, dim=1)
        logits = torch.mm(z_s1, z_s2.t()) / self.temperature
        labels = torch.arange(z_s1.shape[0], device=z_s1.device)
        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_12 + loss_21)


class DisentangleLoss(nn.Module):
    """Centered covariance penalty between shared and private dimensions."""

    def __init__(self, eps=1e-8):
        super(DisentangleLoss, self).__init__()
        self.eps = eps

    def forward(self, z_shared, z_private):
        if z_shared.shape[1] == 0 or z_private.shape[1] == 0:
            return torch.zeros((), dtype=z_shared.dtype, device=z_shared.device)

        z_shared = z_shared - z_shared.mean(dim=0, keepdim=True)
        z_private = z_private - z_private.mean(dim=0, keepdim=True)
        z_shared = z_shared / z_shared.std(dim=0, keepdim=True).clamp_min(self.eps)
        z_private = z_private / z_private.std(dim=0, keepdim=True).clamp_min(self.eps)
        cross_cov = torch.mm(z_shared.t(), z_private) / max(z_shared.shape[0] - 1, 1)
        return torch.mean(cross_cov.pow(2))


def compute_disentangle_loss(z_s1, z_p1, z_s2, z_p2):
    criterion = DisentangleLoss()
    return criterion(z_s1, z_p1) + criterion(z_s2, z_p2)
