import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .image_utils import read_image_any, to_01_3chw_224
from .topo_maps import norm_path


_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMNET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class AlfredoFDDataset(Dataset):
    def __init__(self, idxs, fd_paths, fd_y, topo_store, device, return_path: bool = False):
        self.idxs = np.asarray(idxs, dtype=np.int64)
        self.fd_paths = list(fd_paths)
        self.fd_y = np.asarray(fd_y, dtype=np.int64)
        self.topo_store = topo_store
        self.device = device
        self.return_path = bool(return_path)

    def __len__(self):
        return int(len(self.idxs))

    def __getitem__(self, i):
        idx = int(self.idxs[i])
        path = str(self.fd_paths[idx])
        npath = norm_path(path)

        img = read_image_any(path)
        x3 = torch.from_numpy(to_01_3chw_224(img)).float()
        x3 = (x3 - _IMNET_MEAN) / _IMNET_STD

        if str(self.device).startswith("cuda"):
            x3 = x3.half()

        topo_np = self.topo_store.get_all_levels(npath)
        topo_levels = {
            "L0": torch.from_numpy(topo_np["L0"]).float(),
            "L1": torch.from_numpy(topo_np["L1"]).float(),
            "L2": torch.from_numpy(topo_np["L2"]).float(),
            "L3": torch.from_numpy(topo_np["L3"]).float(),
        }

        for k in list(topo_levels.keys()):
            topo_levels[k] = torch.nan_to_num(topo_levels[k], nan=0.0, posinf=0.0, neginf=0.0)
            if str(self.device).startswith("cuda"):
                topo_levels[k] = topo_levels[k].half()

        y = int(self.fd_y[idx])

        if self.return_path:
            return x3, y, topo_levels, path
        return x3, y, topo_levels


def make_loaders(
    fd_paths,
    fd_y,
    fd_train_idx,
    fd_test_idx,
    topo_store,
    device,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    num_workers = int(globals().get("NUM_WORKERS", max(2, (os.cpu_count() or 4) // 2)))
    pin = str(device).startswith("cuda")

    kw = dict(num_workers=num_workers, pin_memory=pin)
    if num_workers > 0:
        kw.update(dict(persistent_workers=True, prefetch_factor=2))

    train_loader = DataLoader(
        AlfredoFDDataset(fd_train_idx, fd_paths, fd_y, topo_store, device, return_path=False),
        batch_size=batch_size,
        shuffle=True,
        **kw,
    )

    test_loader = DataLoader(
        AlfredoFDDataset(fd_test_idx, fd_paths, fd_y, topo_store, device, return_path=False),
        batch_size=batch_size,
        shuffle=False,
        **kw,
    )

    return train_loader, test_loader


def make_cache_loaders(
    fd_paths,
    fd_y,
    fd_train_idx,
    fd_test_idx,
    topo_store,
    device,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    num_workers = int(globals().get("NUM_WORKERS", max(2, (os.cpu_count() or 4) // 2)))
    pin = str(device).startswith("cuda")

    kw = dict(num_workers=num_workers, pin_memory=pin)
    if num_workers > 0:
        kw.update(dict(persistent_workers=True, prefetch_factor=2))

    train_loader_p = DataLoader(
        AlfredoFDDataset(fd_train_idx, fd_paths, fd_y, topo_store, device, return_path=True),
        batch_size=batch_size,
        shuffle=False,
        **kw,
    )

    test_loader_p = DataLoader(
        AlfredoFDDataset(fd_test_idx, fd_paths, fd_y, topo_store, device, return_path=True),
        batch_size=batch_size,
        shuffle=False,
        **kw,
    )

    return train_loader_p, test_loader_p