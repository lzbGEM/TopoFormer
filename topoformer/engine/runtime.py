import os
import argparse
import logging
import sys

import numpy as np
import torch


def setup_logger(name: str = "dinofd_nb"):
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger


def build_train_args(
    device,
    has_fp16_trainable: bool,
    fd_epochs: int,
    fd_lr: float,
    fd_finetune_mode: str,
    fd_unfreeze_last_k: int,
    fd_lora_r: int,
    fd_lora_alpha: float,
    fd_lora_dropout: float,
):
    return argparse.Namespace(
        epochs=fd_epochs,
        lr=fd_lr,
        min_lr=1e-6,
        warmup_epochs=2,
        weight_decay=0.05,
        amp=bool(str(device).startswith("cuda")) and (not has_fp16_trainable),
        accum_steps=1,
        clip_grad_norm=1.0,
        finetune_mode=fd_finetune_mode,
        unfreeze_last_k=fd_unfreeze_last_k,
        lora_r=fd_lora_r,
        lora_alpha=fd_lora_alpha,
        lora_dropout=fd_lora_dropout,
        label_smoothing=0.0,
        use_pos_weight=False,
        ema=True,
        ema_decay=0.999,
        decouple_mode="cov",
        normalize_ortho=True,
        ortho_topk_ratio=0.0,
        lambda_ortho=0.05,
        lambda_ortho_final=None,
        ortho_delay_epochs=1,
        ortho_decay_epochs=0,
        mine_lr=1e-4,
        adv_mode="kl_uniform",
        lambda_adv=0.1,
        lambda_adv_final=None,
        grl_lambda=0.0,
        adv_warmup_epochs=1,
        adv_delay_epochs=1,
        adv_decay_epochs=0,
        dynamic_lambda_adv=False,
        dyn_delta=5e-4,
        dyn_patience=2,
        dyn_lambda_min=0.01,
        adv_loss_type="bce",
        adv_focal_alpha=0.25,
        adv_focal_gamma=2.0,
        alpha_entropy=0.0,
        cam_topk=0.3,
        no_adapter=False,
        fusion_concat=False,
        val_quick_samples=512,
        val_full_interval=1,
        early_stop_patience=10**9,
        early_stop_min_delta=0.0,
        smoke_one_batch=False,
        supcon_on_A=True,
        lambda_supcon=0.0,
        supcon_temp=0.07,
        use_gate=False,
        gate_l1=0.0,
        kd_on_A=False,
        kd_weight=0.0,
        kd_teacher_path=None,
        kd_temperature=1.0,
    )


@torch.no_grad()
def collect_split(model, loader, device):
    model.eval()

    feats = []
    disc = []
    indisc = []
    logitsA = []
    logitsB = []
    ys = []
    paths = []

    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            images, targets, topo_levels, pths = batch
        elif isinstance(batch, (tuple, list)) and len(batch) == 3:
            images, targets, topo_levels = batch
            pths = [""] * int(images.shape[0])
        else:
            raise ValueError(f"Unexpected cache batch format: {type(batch)}")

        images = images.to(device, non_blocking=True)
        topo_levels = {k: v.to(device, non_blocking=True) for k, v in topo_levels.items()}

        out = model(images, grl_lambda=0.0, topo_levels=topo_levels)

        feats.append(out["feat"].detach().cpu().numpy())
        disc.append(out["f_disc"].detach().cpu().numpy())
        indisc.append(out["f_indisc"].detach().cpu().numpy())
        logitsA.append(out["logitsA"].detach().cpu().numpy())
        logitsB.append(out["logitsB"].detach().cpu().numpy())
        ys.append(np.asarray(targets, dtype=np.int64))
        paths.extend([str(x) for x in pths])

    return {
        "feat": np.concatenate(feats, axis=0).astype(np.float32),
        "disc": np.concatenate(disc, axis=0).astype(np.float32),
        "indisc": np.concatenate(indisc, axis=0).astype(np.float32),
        "logitsA": np.concatenate(logitsA, axis=0).astype(np.float32),
        "logitsB": np.concatenate(logitsB, axis=0).astype(np.float32),
        "y": np.concatenate(ys, axis=0).astype(np.int64),
        "paths": np.asarray(paths, dtype=object),
    }


def save_npz(
    split_name: str,
    loader,
    model_for_cache,
    device,
    cache_dir: str,
    dataset_tag: str,
    fd_id_to_domain,
    fd_label_names,
    encoder_name: str,
    dino_local_ckpt: str,
    fd_bs: int,
    best_epoch: int,
    topo_npz_base: str,
    topo_c: int,
    topo_hw,
):
    out = collect_split(model_for_cache, loader, device)
    out_path = os.path.join(cache_dir, f"dinov3FD_{dataset_tag}_{split_name}.npz")

    np.savez_compressed(
        out_path,
        feat=out["feat"],
        disc=out["disc"],
        indisc=out["indisc"],
        logitsA=out["logitsA"],
        logitsB=out["logitsB"],
        y=out["y"],
        paths=out["paths"],
        classes=np.asarray([fd_id_to_domain[i] for i in range(len(fd_label_names))], dtype=object),
        encoder=np.asarray([encoder_name], dtype=object),
        ckpt=np.asarray([dino_local_ckpt], dtype=object),
        batch=np.asarray([fd_bs], dtype=np.int64),
        best_epoch=np.asarray([best_epoch], dtype=np.int64),
        refit=np.asarray([False], dtype=np.bool_),
        topo_npz_base=np.asarray([topo_npz_base], dtype=object),
        topo_c=np.asarray([topo_c], dtype=np.int64),
        topo_hw=np.asarray([topo_hw], dtype=object),
        topo_scheme=np.asarray(["mrfp_cti_maps"], dtype=object),
    )

    print("[cache] saved:", out_path)
    return out_path


@torch.no_grad()
def predict_a(model, loader, device):
    model.eval()

    ys = []
    ps = []

    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            images, targets, topo_levels, _pths = batch
        else:
            images, targets, topo_levels = batch

        images = images.to(device, non_blocking=True)
        topo_levels = {k: v.to(device, non_blocking=True) for k, v in topo_levels.items()}

        out = model(images, grl_lambda=0.0, topo_levels=topo_levels)
        logits = out["logitsA"].detach().cpu()
        pred = logits.argmax(dim=1).numpy()

        ys.append(np.asarray(targets, dtype=np.int64))
        ps.append(pred)

    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)

    return y, p


def print_report_and_confusion_matrix(
    model_for_cache,
    test_cache_loader,
    device,
    fd_label_names,
    fd_id_to_domain,
    dataset_tag: str,
    encoder_name: str,
    test_metrics,
):
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    y_true, y_pred = predict_a(model_for_cache, test_cache_loader, device)

    labels = list(range(len(fd_label_names)))
    target_names = [fd_id_to_domain[i] for i in labels]

    print("\nclassification_report (test|A head):")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")

    title_acc = float(test_metrics.get("acc_A", float("nan")))
    title_bacc = float(test_metrics.get("bal_acc_A", float("nan")))

    plt.title(
        f"DINO-FD {dataset_tag} + TopoInject(MRFP+CTI maps) | "
        f"{encoder_name} | test acc={title_acc:.3f} bAcc={title_bacc:.3f}"
    )
    plt.tight_layout()
    plt.show()