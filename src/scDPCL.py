import torch
from torch import nn
from torch.nn import Parameter

from encoder_dpcl import SharedQDistribution


class scDPCL(nn.Module):
    """Disentangled Prototype Contrastive Learning model."""

    def __init__(
        self,
        gae1,
        gae2,
        disentangle1,
        disentangle2,
        n_clusters,
        n_shared,
        decode_firnd=True,
        share_cluster_centers=False,
        disentangle_mode="hard",
    ):
        super(scDPCL, self).__init__()
        self.gae1 = gae1
        self.gae2 = gae2
        self.disentangle1 = disentangle1
        self.disentangle2 = disentangle2
        self.decode_firnd = decode_firnd
        self.share_cluster_centers = share_cluster_centers
        self.disentangle_mode = disentangle_mode

        if share_cluster_centers:
            self.cluster_centers = Parameter(torch.Tensor(n_clusters, n_shared))
            torch.nn.init.xavier_normal_(self.cluster_centers.data)
            self.q_distribution1 = SharedQDistribution(self.cluster_centers)
            self.q_distribution2 = SharedQDistribution(self.cluster_centers)
        else:
            self.cluster_centers1 = Parameter(torch.Tensor(n_clusters, n_shared))
            self.cluster_centers2 = Parameter(torch.Tensor(n_clusters, n_shared))
            torch.nn.init.xavier_normal_(self.cluster_centers1.data)
            torch.nn.init.xavier_normal_(self.cluster_centers2.data)
            self.q_distribution1 = SharedQDistribution(self.cluster_centers1)
            self.q_distribution2 = SharedQDistribution(self.cluster_centers2)

    @staticmethod
    def FIRND(adj, z_shared):
        return torch.spmm(adj, z_shared)

    def forward(
        self,
        x1,
        adj1,
        x2,
        adj2,
        pretrain=False,
        firnd_adj1=None,
        firnd_adj2=None,
    ):
        z_igae1, a_igae1 = self.gae1.encoder(x1, adj1)
        z_igae2, a_igae2 = self.gae2.encoder(x2, adj2)

        z_s1, z_p1 = self.disentangle1(z_igae1)
        z_s2, z_p2 = self.disentangle2(z_igae2)

        if firnd_adj1 is None:
            firnd_adj1 = adj1
        if firnd_adj2 is None:
            firnd_adj2 = adj2
        z_firnd1 = self.FIRND(firnd_adj1, z_s1)
        z_firnd2 = self.FIRND(firnd_adj2, z_s2)

        z_shared_decode1 = z_firnd1 if self.decode_firnd else z_s1
        z_shared_decode2 = z_firnd2 if self.decode_firnd else z_s2
        if self.disentangle_mode in ("residual", "adaptive"):
            z_decode1 = z_shared_decode1 + self.disentangle1.private_residual(z_p1)
            z_decode2 = z_shared_decode2 + self.disentangle2.private_residual(z_p2)
        else:
            z_decode1 = torch.cat([z_shared_decode1, z_p1], dim=1)
            z_decode2 = torch.cat([z_shared_decode2, z_p2], dim=1)

        z_hat1, z_adj_hat1 = self.gae1.decoder(z_decode1, adj1)
        a_hat1 = a_igae1 + z_adj_hat1

        z_hat2, z_adj_hat2 = self.gae2.decoder(z_decode2, adj2)
        a_hat2 = a_igae2 + z_adj_hat2

        if pretrain:
            q1, q2 = None, None
        else:
            q1 = self.q_distribution1(z_firnd1, z_s1)
            q2 = self.q_distribution2(z_firnd2, z_s2)

        return (
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
        )
