import torch
import numpy as np


def expectation_maximization(
    flows_loads, 
    links_loads, 
    rm, 
    num_epoch: int,
    flows_masks=None,
    link_masks=None,
    is_sequence=True
):
    b, device = flows_loads.shape[0], flows_loads.device
    m = torch.zeros_like(flows_loads) if flows_masks is None else flows_masks
    m = (m>0).reshape(*m.shape)

    if not is_sequence:
        flows_loads, links_loads = flows_loads.unsqueeze(1), links_loads.unsqueeze(1)
        m = m.unsqueeze(1)

    flows_loads_final = torch.zeros_like(flows_loads, device=device)

    for i in range(b):
        flows_loads_i = em_iteration(flows_loads[i], links_loads[i], rm, num_epoch, m[i])
        flows_loads_final[i] = flows_loads_i

    return flows_loads_final if is_sequence else flows_loads_final.squeeze(1)


def em_iteration(x, y, rm, num_epoch, mask=None):
    device, length, feature = x.device, x.shape[0], x.shape[1]
    idxes = torch.arange(0, length).to(device)
    rm = rm.to(device)
    rm, x_final = rm.to(device), x.clone()
    x_known = x.clone()
    loss_min = torch.empty(length,).to(device)
    loss_min[:] = np.Inf

    div = rm.sum(dim=1)
    zero_mask = (div!=0)
    x = x[:, zero_mask]
    rm = rm[zero_mask, :]
    zero_mask = zero_mask.unsqueeze(0).repeat(length, 1)

    for _ in range(num_epoch):
        a = x / rm.sum(dim=1)
        b = rm / (x @ rm).clamp_min_(1e-6).unsqueeze(1)

        c = y.unsqueeze(1) @ b.transpose(1, 2)
        x = torch.mul(a, c.squeeze(1))

        loss = torch.abs(x @ rm - y).sum(dim=1)
        select = (loss < loss_min).reshape(loss.shape)
        idx = idxes[select]

        if len(idx) != 0:
            loss_min[idx] = loss[idx]
            m = zero_mask.clone()
            m[idxes[~select], :] = False
            x_final[m] = x[idx, :].reshape(-1,)

    x_final[mask] = x_known[mask]
    return x_final
