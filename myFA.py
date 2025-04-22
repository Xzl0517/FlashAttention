import torch
import torch.nn as nn
import math


# Q b x
class FlashAttention(nn.Module):
    def __init__(self):
        super(FlashAttention, self).__init__()

    def forward(self, Q, K, V):
        n, d = Q.shape
        M = 128  # 假设SRAM的size为16
        Bc, Br = math.ceil(M // (4 * d)), min(math.ceil(M // (4 * d)), d)
        print(Bc)
        print(Br)
        O = torch.zeros((n, d))
        l = torch.zeros((n, 1))  # 每块分母和
        m = torch.full((n, 1), -torch.inf)  # 每块最大值

        for kv_start in range(0, n, Bc):
            Ki = K[kv_start:kv_start + Bc, :]  # Bc x d
            Vi = V[kv_start:kv_start + Bc, :]  # Bc x d

            for q_start in range(0, n, Br):
                Oi = O[q_start:q_start + Br:, :]  # Br x d
                li = l[q_start:q_start + Br:, :]  # Br x 1
                mi = m[q_start:q_start + Br:, :]  # Br x 1
                Qi = Q[q_start:q_start + Br:, :]  # Br x d

                Sij = Qi @ Ki.T  # Br x Bc

                _mij = torch.max(Sij, dim=1).values[:, None]  # Br x 1
                _pij = torch.exp(Sij - _mij)  # Br x Bc
                _lij = torch.sum(_pij, dim=1)[:, None]  # Br x 1
                mi_new = torch.max(torch.cat([_mij, mi], dim=1), dim=1).values[:, None]  # Br x 1
                li_new = torch.exp(mi - mi_new) * li + torch.exp(_mij - mi_new) * _lij
                print(li_new)
                A = (torch.exp(mi - mi_new) * Oi) * li
                B = torch.exp(_mij - mi_new) * _pij @ Vi
                print(A)
                print(B)
                Oi = (A + B) / li_new

                O[q_start:q_start + Br:, :] = Oi
                l[q_start:q_start + Br:, :] = li_new
                m[q_start:q_start + Br:, :] = mi_new

        return O


torch.manual_seed(0)

N, d = 8, 4

Q_mat = torch.rand((N, d))
K_mat = torch.rand((N, d))
V_mat = torch.rand((N, d))
print(Q_mat)
print(K_mat)
print(V_mat)
model = FlashAttention()
print(model(Q_mat, K_mat, V_mat))

standard_softmax = torch.softmax(Q_mat @ K_mat.T, dim=1)
standard_attention = standard_softmax @ V_mat
print(standard_attention)
