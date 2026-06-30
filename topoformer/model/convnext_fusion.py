import os

import torch

from topoformer.model.disentanglement import grad_reverse


def _select_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "network", "net"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj

    raise TypeError(f"Unexpected checkpoint object type: {type(obj)}")


def _strip_prefix(sd: dict, prefix: str = "module.") -> dict:
    out = {}
    for k, v in sd.items():
        kk = k[len(prefix):] if isinstance(k, str) and k.startswith(prefix) else k
        out[kk] = v
    return out


def build_convnextv2(convnext_ckpt: str, convnext_name: str, device):
    import timm  # type: ignore

    assert os.path.exists(convnext_ckpt), f"ConvNeXtV2 ckpt not found: {convnext_ckpt}"

    names_to_try = []
    if convnext_name:
        names_to_try.append(convnext_name)

    names_to_try.extend([
        "convnextv2_base",
        "convnextv2_base.fcmae",
        "convnextv2_base.fb_in1k",
        "convnextv2_base_in1k",
    ])

    last_err = None
    cnx = None

    for nm in names_to_try:
        try:
            cnx = timm.create_model(nm, pretrained=False, num_classes=0, global_pool="avg")
            print("[cnx] timm model =", nm, "| num_features=", getattr(cnx, "num_features", "?"))
            break
        except Exception as e:
            last_err = e
            cnx = None

    if cnx is None:
        raise RuntimeError(f"Failed to create ConvNeXtV2 via timm. Tried={names_to_try}. Last error={last_err!r}")

    raw = torch.load(convnext_ckpt, map_location="cpu")
    sd = _strip_prefix(_select_state_dict(raw), prefix="module.")
    missing, unexpected = cnx.load_state_dict(sd, strict=False)

    print(f"[cnx] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")

    for p in cnx.parameters():
        p.requires_grad = False

    cnx.eval()

    if str(device).startswith("cuda"):
        cnx.to(dtype=torch.float16)

    return cnx


class GatedFusionFD(torch.nn.Module):
    def __init__(self, vit_model: torch.nn.Module, cnx_model: torch.nn.Module):
        super().__init__()

        self.vit = vit_model
        self.cnx = cnx_model

        self.out_dim = int(getattr(vit_model, "out_dim", 1))
        self.feature_dim = int(getattr(vit_model, "feature_dim", 256))

        cnx_dim = getattr(cnx_model, "num_features", None)
        if cnx_dim is None:
            raise RuntimeError("ConvNeXtV2 model has no num_features; cannot infer feature dim.")

        self.cnx_dim = int(cnx_dim)

        self.proj_vit = torch.nn.Identity()
        self.proj_cnx = torch.nn.Linear(self.cnx_dim, self.feature_dim)

        self.gate_ln = torch.nn.LayerNorm(self.feature_dim + self.cnx_dim)
        self.gate_fc = torch.nn.Linear(self.feature_dim + self.cnx_dim, self.feature_dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.vit, name)

    def _encode_fused(self, x: torch.Tensor, topo_levels=None):
        feat_vit = self.vit._encode(x, topo_levels=topo_levels).float()

        feat_cnx = self.cnx(x)
        if isinstance(feat_cnx, (tuple, list)):
            feat_cnx = feat_cnx[0]
        if feat_cnx.dim() > 2:
            feat_cnx = feat_cnx.mean(dim=tuple(range(2, feat_cnx.dim())))
        feat_cnx = feat_cnx.float()

        cat = torch.cat([feat_vit, feat_cnx], dim=1)
        g = torch.sigmoid(self.gate_fc(self.gate_ln(cat)))
        fused = g * self.proj_vit(feat_vit) + (1.0 - g) * self.proj_cnx(feat_cnx)

        return fused, g

    def forward(self, x: torch.Tensor, grl_lambda: float = 1.0, use_adapter: bool = True, topo_levels=None):
        feats, fusion_gate = self._encode_fused(x, topo_levels=topo_levels)

        if use_adapter:
            f_disc = self.vit.adapter_disc(feats)
            f_indisc = self.vit.adapter_indisc(feats)
        else:
            f_disc = self.vit.proj_disc_simple(feats)
            f_indisc = self.vit.proj_indisc_simple(feats)

        gate = None
        if bool(getattr(self.vit, "use_gating", False)):
            gate = torch.sigmoid(self.vit.gate_layer(feats))
            f_disc = f_disc * gate
            f_indisc = f_indisc * (1.0 - gate)

        logitsA = self.vit.decoderA(f_disc)
        logitsB = self.vit.decoderB(grad_reverse(f_indisc, grl_lambda))

        return {
            "logitsA": logitsA,
            "logitsB": logitsB,
            "f_disc": f_disc,
            "f_indisc": f_indisc,
            "feat": feats,
            "gate": gate,
            "fusion_gate": fusion_gate,
        }