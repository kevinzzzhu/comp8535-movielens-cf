"""Training loop shared by the proposed model and the MF/NMF baselines."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5  # early stopping
    device: str = "cpu"
    seed: int = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate(model, loader, user_features, item_features, device: str, use_features: bool):
    """Return RMSE, MAE, rounded-prediction accuracy, and NLL when the model exposes class probs."""
    model.eval()
    preds, targets, nll_sum, nll_n = [], [], 0.0, 0
    with torch.no_grad():
        for u, i, r in loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            if use_features:
                out = model(u, i, user_features, item_features)
                p = out["pred"]
                if "probs" in out:
                    cls = (r.round().long() - 1).clamp(0, 4)
                    log_p = torch.log(out["probs"] + 1e-12)
                    nll_sum += float(-log_p.gather(-1, cls.unsqueeze(-1)).sum())
                    nll_n += r.shape[0]
            else:
                p = model(u, i)
            preds.append(p.cpu())
            targets.append(r.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    rmse = math.sqrt(float(((preds - targets) ** 2).mean()))
    mae = float((preds - targets).abs().mean())
    rounded = preds.round().clamp(1, 5)
    acc = float((rounded == targets).float().mean())
    metrics = {"rmse": rmse, "mae": mae, "acc": acc}
    if nll_n > 0:
        metrics["nll"] = nll_sum / nll_n
    return metrics


def train_model(
    model,
    train_ds,
    test_ds,
    cfg: TrainConfig,
    user_features: torch.Tensor | None = None,
    item_features: torch.Tensor | None = None,
    use_features: bool = True,
    log_gate: bool = False,
):
    device = cfg.device
    model.to(device)
    if user_features is not None:
        user_features = user_features.to(device)
        item_features = item_features.to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
    # Split parameters: ordinal thresholds are few and their scale is calibration-critical,
    # so they should not be shrunk by L2. Everything else gets cfg.weight_decay.
    no_decay_params, decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head.theta1") or name.startswith("head.deltas"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    param_groups = [{"params": decay_params, "weight_decay": cfg.weight_decay}]
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    opt = torch.optim.Adam(param_groups, lr=cfg.lr)

    best_rmse = float("inf")
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        gate_sum = 0.0
        gate_n = 0
        for u, i, r in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            u, i, r = u.to(device), i.to(device), r.to(device)
            if use_features:
                out = model(u, i, user_features, item_features)
                loss = model.loss(out, r)
                if log_gate and out.get("gate_u") is not None:
                    gate_sum += float(out["gate_u"].detach().mean()) + float(out["gate_i"].detach().mean())
                    gate_n += 2
            else:
                p = model(u, i)
                loss = model.loss(p, r)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Projected gradient descent: models expose project_() for constraint enforcement
            # (e.g. NMF's non-negativity on factor matrices).
            if hasattr(model, "project_"):
                model.project_()
            running += float(loss.detach()) * u.shape[0]

        train_loss = running / len(train_ds)
        metrics = evaluate(model, test_loader, user_features, item_features, device, use_features)
        entry = {"epoch": epoch, "train_loss": train_loss, **metrics}
        if log_gate and gate_n > 0:
            entry["mean_gate"] = gate_sum / gate_n
        history.append(entry)

        if metrics["rmse"] < best_rmse - 1e-4:
            best_rmse = metrics["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_rmse": best_rmse, "history": history}
