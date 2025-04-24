import torch
import torch.nn as nn
import math


class FlashAttentionV2(nn.Module):
    def __init__(self):
        super(FlashAttentionV2, self).__init__()

    def forward(self, Q, K, V):
        n, d = Q.shape
        M = 128  # 假设SRAM的size为16
        Bc, Br = math.ceil(M//(4*d)), min(d, math.ceil(M//(4*d)))
        O = torch.zeros((n, d))

        for q_start in range(0, n, Br):
            Qi = Q[q_start:q_start+Br, :]  # Br x d

            Oi = torch.zeros((Br, d))  # Br x d
            li = torch.zeros((Br, 1))  # Br x 1
            mi = torch.full((Br, 1), -torch.inf)  # Br x 1

            for kv_start in range(0, n, Bc):
                Ki = K[kv_start:kv_start+Bc, :]  # Bc x d
                Vi = V[kv_start:kv_start+Bc, :]  # Bc x d

                Sij = Qi @ Ki.T  # Br x Bc
                Sij_max = torch.max(Sij, dim=1).values[:, None]  # Br x 1
                mi_new = torch.max(torch.cat([mi, Sij_max], dim=1), dim=1).values[:, None]  # Br x 1
                Pij = torch.exp(Sij - mi_new)  # Br x Bc
                li = torch.exp(mi - mi_new) * li + torch.sum(Pij, dim=1)[:, None]

                Oi = Pij @ Vi + torch.exp(mi - mi_new) * Oi

                mi = mi_new
            Oi = Oi / li
            O[q_start:q_start+Br, :] = Oi
        return O


torch.manual_seed(0)

N, d = 8, 4

Q_mat = torch.rand((N, d))
K_mat = torch.rand((N, d))
V_mat = torch.rand((N, d))
print(Q_mat)
print(K_mat)
print(V_mat)
model = FlashAttentionV2()
print(model(Q_mat, K_mat, V_mat))

standard_softmax = torch.softmax(Q_mat @ K_mat.T, dim=1)
standard_attention = standard_softmax @ V_mat
print(standard_attention)


