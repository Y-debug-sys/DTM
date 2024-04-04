import torch
import numpy as np


def NMAE(reals, fakes, masks=None):
    if masks is None:
        masks = np.ones_like(reals).astype(np.bool_)
    nmae = np.abs(fakes[masks] - reals[masks]).sum() / np.abs(reals[masks]).sum()
    return nmae


def NRMSE(reals, fakes, masks=None):
    if masks is None:
        masks = np.ones_like(reals).astype(np.bool_)
    nrmse = np.sqrt(np.square(fakes[masks] - reals[masks]).sum()) / np.sqrt(np.square(reals[masks]).sum())
    return nrmse


def MMD(x, y, kernel="rbf"):
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    # mmd_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]

    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = torch.clamp(rx.t() + rx - 2.*xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2.*yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

    XX, YY, XY = (torch.zeros(xx.shape),
                  torch.zeros(xx.shape),
                  torch.zeros(xx.shape))

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return (XX + YY - 2.*XY).mean().item()