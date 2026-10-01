import torch
from torch import nn


class MLPEncoder(nn.Module):
    """Simple feature only encoder used for strong ablation baselines."""

    def __init__(self, n_input, n_hidden1, n_hidden2, n_z, dropout):
        super(MLPEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden1),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden1, n_hidden2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden2, n_z),
        )
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z = self.net(x)
        z_adj = self.s(torch.mm(z, z.t()))
        return z, z_adj


class MLPDecoder(nn.Module):
    """Simple feature only decoder with the same interface as IGAE."""

    def __init__(self, n_input, n_hidden1, n_hidden2, n_z):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_z, n_hidden1),
            nn.Tanh(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.Tanh(),
            nn.Linear(n_hidden2, n_input),
            nn.Tanh(),
        )
        self.s = nn.Sigmoid()

    def forward(self, z, adj):
        z_hat = self.net(z)
        z_adj_hat = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_adj_hat


class MLPAE(nn.Module):
    """MLP autoencoder that ignores graph propagation.

    It keeps the encoder and decoder call signatures compatible with IGAE.
    """

    def __init__(
        self,
        n_input,
        n_z,
        enc_hidden1=256,
        enc_hidden2=128,
        dec_hidden1=128,
        dec_hidden2=256,
        dropout=0.0,
    ):
        super(MLPAE, self).__init__()
        self.encoder = MLPEncoder(n_input, enc_hidden1, enc_hidden2, n_z, dropout)
        self.decoder = MLPDecoder(n_input, dec_hidden1, dec_hidden2, n_z)


class DisentangleHead(nn.Module):
    """Split an IGAE embedding into shared and private representations."""

    def __init__(self, n_z, n_shared, n_private):
        super(DisentangleHead, self).__init__()
        if n_shared + n_private != n_z:
            raise ValueError(
                "n_shared ({}) + n_private ({}) must equal n_z ({})".format(
                    n_shared, n_private, n_z
                )
            )
        self.n_shared = n_shared
        self.n_private = n_private
        self.shared_head = nn.Linear(n_z, n_shared, bias=False)
        self.private_head = nn.Linear(n_z, n_private, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize so cat(shared, private) is initially equal to z_igae."""
        with torch.no_grad():
            self.shared_head.weight.zero_()
            self.private_head.weight.zero_()
            self.shared_head.weight[:, : self.n_shared] = torch.eye(self.n_shared)
            for i in range(self.n_private):
                self.private_head.weight[i, self.n_shared + i] = 1.0

    def forward(self, z_igae):
        z_shared = self.shared_head(z_igae)
        z_private = self.private_head(z_igae)
        return z_shared, z_private


class ResidualDisentangleHead(nn.Module):
    """Full shared representation plus a private residual branch."""

    def __init__(self, n_z, n_shared, n_private):
        super(ResidualDisentangleHead, self).__init__()
        if n_shared != n_z:
            raise ValueError("residual mode requires n_shared ({}) == n_z ({})".format(n_shared, n_z))
        if n_private <= 0:
            raise ValueError("residual mode requires n_private > 0")

        self.n_shared = n_shared
        self.n_private = n_private
        self.shared_head = nn.Linear(n_z, n_shared, bias=False)
        self.private_head = nn.Linear(n_z, n_private, bias=False)
        self.private_adapter = nn.Linear(n_private, n_shared, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.shared_head.weight.zero_()
            self.shared_head.weight[:, :] = torch.eye(self.n_shared)
            nn.init.xavier_uniform_(self.private_head.weight)
            self.private_adapter.weight.zero_()

    def forward(self, z_igae):
        z_shared = self.shared_head(z_igae)
        z_private = self.private_head(z_igae)
        return z_shared, z_private

    def private_residual(self, z_private):
        return self.private_adapter(z_private)


class AdaptiveDisentangleHead(nn.Module):
    """Learn a per-dimension mixture of full and bottleneck shared features."""

    def __init__(self, n_z, n_shared, n_private):
        super(AdaptiveDisentangleHead, self).__init__()
        if n_shared != n_z:
            raise ValueError(
                "adaptive mode requires n_shared ({}) == n_z ({})".format(
                    n_shared, n_z
                )
            )
        if n_private <= 0 or n_private >= n_z:
            raise ValueError("adaptive mode requires 0 < n_private < n_z")

        self.n_shared = n_shared
        self.n_private = n_private
        self.n_bottleneck = n_z - n_private
        self.full_shared_head = nn.Linear(n_z, n_shared, bias=False)
        self.bottleneck_head = nn.Linear(n_z, self.n_bottleneck, bias=False)
        self.shared_adapter = nn.Linear(self.n_bottleneck, n_shared, bias=False)
        self.private_head = nn.Linear(n_z, n_private, bias=False)
        self.private_adapter = nn.Linear(n_private, n_shared, bias=False)
        self.gate_logits = nn.Parameter(torch.zeros(n_shared))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.full_shared_head.weight.copy_(torch.eye(self.n_shared))
            self.bottleneck_head.weight.zero_()
            self.bottleneck_head.weight[:, : self.n_bottleneck] = torch.eye(
                self.n_bottleneck
            )
            self.shared_adapter.weight.zero_()
            self.shared_adapter.weight[: self.n_bottleneck, :] = torch.eye(
                self.n_bottleneck
            )
            self.private_head.weight.zero_()
            for index in range(self.n_private):
                self.private_head.weight[
                    index, self.n_bottleneck + index
                ] = 1.0
            self.private_adapter.weight.zero_()
            self.gate_logits.zero_()

    def forward(self, z_igae):
        z_full = self.full_shared_head(z_igae)
        z_bottleneck = self.shared_adapter(self.bottleneck_head(z_igae))
        gate = torch.sigmoid(self.gate_logits).unsqueeze(0)
        z_shared = gate * z_full + (1.0 - gate) * z_bottleneck
        z_private = self.private_head(z_igae)
        return z_shared, z_private

    def private_residual(self, z_private):
        return self.private_adapter(z_private)

    def gate_mean(self):
        return torch.sigmoid(self.gate_logits).mean()


class SharedQDistribution(nn.Module):
    """Student-t cluster assignment in the shared latent space."""

    def __init__(self, centers):
        super(SharedQDistribution, self).__init__()
        self.cluster_centers = centers

    def _student_t(self, z):
        q = 1.0 / (
            1.0
            + torch.sum(
                torch.pow(z.unsqueeze(1) - self.cluster_centers, 2),
                dim=2,
            )
        )
        q = (q.t() / torch.clamp(q.sum(dim=1), min=1e-12)).t()
        return q

    def forward(self, z_shared_firnd, z_shared_raw):
        return [self._student_t(z_shared_firnd), self._student_t(z_shared_raw)]
