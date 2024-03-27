import numpy as np
import torch


def RBF_kernel(X, Y, h_min=1e-3):
    N, Kx, Ky, D = X.shape[0], X.shape[1], Y.shape[1], X.shape[2]
    diff = X.unsqueeze(-2) - Y.unsqueeze(-3) # shape = (N, Kx, Ky, D)
    dist_sq = diff.pow(2).sum(-1, keepdim=True) # shape = (N, Kx, Ky, 1)

    # Apply the median heuristic (PyTorch does not give true median)
    median_sq = torch.quantile(dist_sq.reshape(N, Kx*Ky), q=0.5, dim=1)
    sigma_sq = median_sq / (2 * np.log(N + 1))
    sigma_sq = sigma_sq.detach().reshape(N, 1, 1, 1) # shape = (N, 1, 1, 1)  

    # calculate kappa and kappa_grad for sql
    gamma = 1.0 / (1e-8 + 2 * sigma_sq)
    kappa = (-gamma * dist_sq).exp() # shape = (N, Kx, Ky, 1)
    kappa_grad = -2. * (gamma* diff) * kappa # shape = (N, Kx, Ky, D)

    return kappa, kappa_grad


if __name__ == '__main__':
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 3, 4)
    kappa, kappa_grad = RBF_kernel(a, b)
    print(kappa.shape)
    print(kappa_grad.shape)