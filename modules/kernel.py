import numpy as np
import torch


def RBF_kernel(X, Y, h_min=1e-3):
    N, Kx, Ky, D = X.shape[0], X.shape[1], Y.shape[1], X.shape[2]
    diff = X.unsqueeze(-2) - Y.unsqueeze(-3) # shape = (N, Kx, Ky, D)
    dist_sq = diff.pow(2).sum(-1, keepdim=True) # shape = (N, Kx, Ky, 1)

    # Apply the median heuristic (PyTorch does not give true median)
    median_sq = torch.quantile(dist_sq.reshape(N, -1), q=0.5, dim=1)
    h = median_sq / (2*np.log(N) + 1)
    h = torch.max(h, torch.tensor(h_min)) 
    h = h.detach().reshape(N, 1, 1, 1) # shape = (N, 1, 1, 1)

    # calculate kappa and kappa_grad for sql
    kappa = (-dist_sq / h).exp() # shape = (N, Kx, Ky, 1)
    kappa_grad = -2. * (diff / h) * kappa # shape = (N, Kx, Ky, D)

    return kappa, kappa_grad


if __name__ == '__main__':
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 3, 4)
    kappa, kappa_grad = RBF_kernel(a, b)
    print(kappa.shape)
    print(kappa_grad.shape)