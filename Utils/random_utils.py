import torch
import random
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def random_mask(observed_values, missing_ratio=0.1, seed=1984, exclude_features=None, exclude_zeros=True):
    observed_masks = ~np.isnan(observed_values)
    if exclude_zeros:
        observed_masks[observed_values==0.] = False
    if exclude_features is not None:
        observed_masks[:, exclude_features] = False

    # randomly set some percentage as ground-truth
    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()

    # Store the state of the RNG to restore later.
    st0 = np.random.get_state()
    np.random.seed(seed)

    miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
    )

    # Restore RNG.
    np.random.set_state(st0)
    
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    return torch.from_numpy(observed_values).float(), torch.from_numpy(observed_masks).float(),\
           torch.from_numpy(gt_masks).float()


def select(num, ratio=0.5, seed=1984):
    unknown_num = int(np.ceil(num * ratio))

    # Store the state of the RNG to restore later.
    st0 = np.random.get_state()
    np.random.seed(seed)

    id_rdm = torch.randperm(num).cpu().numpy()
    ex_fs = id_rdm[:unknown_num]

    # Restore RNG.
    np.random.set_state(st0)
    return ex_fs