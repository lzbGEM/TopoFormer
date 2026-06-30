import torch
import torch.nn.functional as F


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()

        if not isinstance(base, torch.nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base = base
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.dropout = torch.nn.Dropout(float(dropout)) if float(dropout) > 0 else torch.nn.Identity()

        self.lora_A = torch.nn.Parameter(torch.zeros(self.r, self.in_features, dtype=torch.float32))
        self.lora_B = torch.nn.Parameter(torch.zeros(self.out_features, self.r, dtype=torch.float32))

        torch.nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_B)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        x_d = self.dropout(x).to(self.lora_A.dtype)
        z = F.linear(x_d, self.lora_A)
        z = F.linear(z, self.lora_B)
        return y + (z * self.scaling).to(y.dtype)


def get_vit_blocks(encoder: torch.nn.Module):
    for path in [("blocks",), ("backbone", "blocks"), ("model", "blocks"), ("trunk", "blocks")]:
        m = encoder
        ok = True
        for a in path:
            if not hasattr(m, a):
                ok = False
                break
            m = getattr(m, a)

        if ok and isinstance(m, (torch.nn.ModuleList, list, tuple)) and len(m) > 0:
            return list(m)

    return None


def inject_lora_vit(
    encoder: torch.nn.Module,
    r: int,
    alpha: float,
    dropout: float,
    targets=("qkv", "proj", "fc1", "fc2"),
):
    blocks = get_vit_blocks(encoder)
    if blocks is None:
        raise RuntimeError("Cannot locate ViT blocks on encoder.")

    replaced = 0

    for blk in blocks:
        for name, mod in list(blk.named_modules()):
            if not isinstance(mod, torch.nn.Linear):
                continue
            if isinstance(mod, LoRALinear):
                continue

            short = name.split(".")[-1]
            if short not in targets:
                continue

            parent = blk
            if name:
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                leaf = parts[-1]
            else:
                leaf = short

            setattr(parent, leaf, LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
            replaced += 1

    return replaced