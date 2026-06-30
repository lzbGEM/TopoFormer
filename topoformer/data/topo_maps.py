import os
from typing import Dict

import numpy as np


def norm_path(p: str) -> str:
    return os.path.realpath(str(p))


class TopoMapStore:
    def __init__(self, topo_npz_base: str):
        self.topo_npz_base = str(topo_npz_base).strip()
        self.levels = {f"L{i}": self._load_level_npz(i) for i in [0, 1, 2, 3]}
        self.topo_c = int(self.levels["L0"]["C"])
        self.topo_hw = tuple(self.levels["L0"]["HW"])

        print(
            "[topo] loaded maps:",
            {k: v["path"] for k, v in self.levels.items()},
            "| C=",
            self.topo_c,
            "HW=",
            self.topo_hw,
        )

    def _load_level_npz(self, level: int) -> Dict:
        path = f"{self.topo_npz_base}_L{int(level)}.npz"
        assert os.path.exists(path), f"Topo npz not found: {path}"

        npz = np.load(path, allow_pickle=False)

        train_files = np.asarray(npz["train_files"], dtype=str)
        test_files = np.asarray(npz["test_files"], dtype=str)
        train_maps = np.asarray(npz["train_phc_map_level"], dtype=np.float32)
        test_maps = np.asarray(npz["test_phc_map_level"], dtype=np.float32)

        assert train_maps.shape[0] == train_files.shape[0]
        assert test_maps.shape[0] == test_files.shape[0]

        c = int(train_maps.shape[1])
        h = int(train_maps.shape[2])
        w = int(train_maps.shape[3])
        n_train = int(train_files.shape[0])

        path2idx = {}
        for i, p in enumerate(train_files):
            path2idx[norm_path(p)] = int(i)
        for i, p in enumerate(test_files):
            path2idx[norm_path(p)] = int(n_train + i)

        return {
            "path": path,
            "C": c,
            "HW": (h, w),
            "n_train": n_train,
            "train": train_maps,
            "test": test_maps,
            "path2idx": path2idx,
        }

    def get_topo_map(self, level_key: str, npath: str) -> np.ndarray:
        info = self.levels[str(level_key)]
        gi = int(info["path2idx"][npath])
        if gi < int(info["n_train"]):
            return info["train"][gi]
        return info["test"][gi - int(info["n_train"])]

    def get_all_levels(self, path: str) -> Dict[str, np.ndarray]:
        npath = norm_path(path)
        return {
            "L0": self.get_topo_map("L0", npath),
            "L1": self.get_topo_map("L1", npath),
            "L2": self.get_topo_map("L2", npath),
            "L3": self.get_topo_map("L3", npath),
        }

    def check_alignment(self, image_paths) -> None:
        missing = []
        for p in image_paths:
            pp = norm_path(p)
            ok = all(pp in self.levels[f"L{i}"]["path2idx"] for i in [0, 1, 2, 3])
            if not ok:
                missing.append(pp)

        if len(missing) > 0:
            print("[topo][WARN] missing topo for paths (show up to 10):")
            for x in missing[:10]:
                print("  ", x)
            raise AssertionError(f"Topo features missing for {len(missing)}/{len(image_paths)} images")

        print("[topo] path alignment OK")