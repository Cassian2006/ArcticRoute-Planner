from __future__ import annotations

import os
import json
import math
import time
import random
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional, Tuple
from pathlib import Path
import csv

import numpy as np  # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

try:
    from logging_config import get_logger
except Exception:
    import logging
    def get_logger(name: str):
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        return logging.getLogger(name)

from ArcticRoute.core.prior.transformer_model import TrajectoryTransformer, ModelConfig
from ArcticRoute.core.prior.transformer_loss import total_loss, LossConfig, ProjectionHead
from ArcticRoute.core.prior.transformer_ds import TrajPairDataset, PairDatasetConfig, collate_pairs
from ArcticRoute.core.prior.transformer_feats import FeatureConfig, apply_posenc
from ArcticRoute.core.prior.transformer_eval import embed_loader, eval_retrieval
from ArcticRoute.cache.index_util import register_artifact

logger = get_logger(__name__)


@dataclass
class TrainerConfig:
    ym: str = "202412"
    epochs: int = 50
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    bf16: bool = False
    amp: bool = False
    compile: bool = False
    ema_decay: Optional[float] = None
    patience: int = 10
    seed: int = 42
    log_dir: Optional[str] = None
    resume: Optional[str] = None
    lambda_mim: float = 0.5
    temperature: float = 0.1
    # E-T07.5 new fields
    dry_run: bool = False
    seq_len: int = 512
    num_workers: int = 0
    split_json: Optional[str] = None
    segment_index: Optional[str] = None
    tracks: Optional[str] = None


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert n in self.shadow
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_run_dir(cfg: TrainerConfig) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%S")
    run_id = f"prior_{cfg.ym}_{ts}"
    base = Path(cfg.log_dir or (Path.cwd() / "reports" / "phaseE" / "train_logs"))
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_ckpt(path: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any], epoch: int, best_metric: float, projector: Optional[nn.Module] = None, ema: Optional[EMA] = None, cfg: Optional[TrainerConfig] = None, run_id: str = "") -> None:
    obj: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        },
        "run_id": run_id,
    }
    if scheduler is not None:
        try:
            obj["scheduler"] = scheduler.state_dict()
        except Exception:
            pass
    if projector is not None:
        obj["projector"] = projector.state_dict()
    if ema is not None:
        obj["ema"] = {k: v.clone() for k, v in ema.shadow.items()}
    if cfg is not None:
        obj["cfg"] = asdict(cfg)
    torch.save(obj, str(path))


def load_ckpt(path: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any], projector: Optional[nn.Module] = None, ema: Optional[EMA] = None) -> Tuple[int, float]:
    ck = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ck["model"])  # type: ignore
    try:
        optimizer.load_state_dict(ck["optimizer"])  # type: ignore
    except Exception:
        # 当优化器参数组不匹配（例如仅需加载推理权重）时，忽略优化器状态恢复
        pass
    if scheduler is not None and "scheduler" in ck:
        try:
            scheduler.load_state_dict(ck["scheduler"])  # type: ignore
        except Exception:
            pass
    if projector is not None and "projector" in ck:
        projector.load_state_dict(ck["projector"])  # type: ignore
    if ema is not None and "ema" in ck:
        for k, v in ck["ema"].items():
            ema.shadow[k] = v.to(dtype=ema.shadow[k].dtype, device=ema.shadow[k].device) if k in ema.shadow else v
    try:
        torch.set_rng_state(ck["rng"]["torch"])  # type: ignore
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])  # type: ignore
        np.random.set_state(ck["rng"]["numpy"])  # type: ignore
        random.setstate(ck["rng"]["random"])  # type: ignore
    except Exception:
        pass
    return int(ck.get("epoch", 0)), float(ck.get("best_metric", float("-inf")))

def plot_metrics_csv(csv_path: Path, png_path: Path) -> None:
    if plt is None: return
    import csv
    data = {h: [] for h in ("epoch", "loss", "nce", "mim", "recall@1", "recall@5", "recall@10", "mAP", "lr")}
    with open(csv_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            for h in data:
                data[h].append(float(row.get(h, 0.0)))
    epochs = data["epoch"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, data["loss"], label="train_loss")
    ax[0].plot(epochs, data["nce"], label="nce", linestyle=':')
    ax[0].plot(epochs, data["mim"], label="mim", linestyle=':')
    ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(True); ax[0].set_title("Losses")
    ax[1].plot(epochs, data["recall@5"], label="Recall@5")
    ax[1].plot(epochs, data["mAP"], label="mAP", linestyle='--')
    ax1_twin = ax[1].twinx()
    ax1_twin.plot(epochs, data["lr"], label="LR", color='gray', alpha=0.6)
    ax[1].legend(loc='upper left'); ax1_twin.legend(loc='upper right'); ax[1].grid(True); ax[1].set_title("Val Metrics & LR")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path))
    plt.close(fig)

def train_loop(cfg: TrainerConfig, *, model_cfg: Optional[ModelConfig] = None) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    run_dir = build_run_dir(cfg)
    run_id = run_dir.name
    (run_dir / "attrs.json").write_text(json.dumps({"run_id": run_id, "cfg": asdict(cfg)}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Data
    ds_cfg = PairDatasetConfig(ym=cfg.ym, max_len=cfg.seq_len)
    train_ds = TrajPairDataset(Path(cfg.split_json) if cfg.split_json else None, Path(cfg.segment_index), Path(cfg.tracks), cfg=ds_cfg)
    val_ds = TrajPairDataset(Path(cfg.split_json) if cfg.split_json else None, Path(cfg.segment_index), Path(cfg.tracks), cfg=PairDatasetConfig(ym=cfg.ym, split="val", max_len=cfg.seq_len))
    collate_fn = lambda b: collate_pairs(b, cfg.seq_len, feat_cfg=FeatureConfig())
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Model
    m_cfg = model_cfg or ModelConfig()
    model = TrajectoryTransformer(d_in=9, cfg=m_cfg).to(device)
    if cfg.compile and hasattr(torch, "compile"): model = torch.compile(model)
    projector = ProjectionHead(d_in=m_cfg.d_model).to(device)

    # Optimizer, Scheduler, EMA
    optim = torch.optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # 注意：学习率调度以优化器步数为单位（考虑梯度累积）
    opt_steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, cfg.grad_accum)))
    sched = None  # 将在处理 resume 之后按剩余轮数重建 OneCycleLR
    ema = EMA(model, cfg.ema_decay) if (cfg.ema_decay and cfg.ema_decay > 0) else None

    start_epoch, best_metric = 0, float("-inf")
    if cfg.resume and Path(cfg.resume).exists():
        # 恢复模型/优化器与随机种子，不恢复旧 scheduler，避免总步数不匹配
        start_epoch, best_metric = load_ckpt(Path(cfg.resume), model=model, optimizer=optim, scheduler=None, projector=projector, ema=ema)
    # 按剩余轮数重建新的 OneCycleLR（考虑梯度累积）
    epochs_remaining = max(0, int(cfg.epochs - start_epoch))
    if epochs_remaining == 0:
        logger.info(f"Nothing to train: start_epoch={start_epoch}, target epochs={cfg.epochs}")
        return {"run_dir": str(run_dir), "best_recall@5": float(best_metric)}
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=cfg.lr, steps_per_epoch=opt_steps_per_epoch, epochs=epochs_remaining)

    scaler = torch.amp.GradScaler(device=device, enabled=(cfg.amp and not cfg.bf16))
    csv_path = run_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["epoch", "loss", "nce", "mim", "recall@1", "recall@5", "recall@10", "mAP", "lr"])

    no_improve, loss_cfg = 0, LossConfig(temperature=cfg.temperature, lambda_mim=cfg.lambda_mim)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, (cfg.epochs - start_epoch) * steps_per_epoch)

    for epoch in range(start_epoch, cfg.epochs):
        model.train(); projector.train()
        running_loss, running_nce, running_mim, steps = 0.0, 0.0, 0.0, 0

        for it, batch in enumerate(train_loader):
            for k in ("x_i", "mask_i", "x_j", "mask_j"): batch[k] = batch[k].to(device)
            posenc = apply_posenc(cfg.seq_len, method="rope")

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if cfg.bf16 else torch.float16, enabled=cfg.amp or cfg.bf16):
                z_i = model(batch["x_i"], mask=batch["mask_i"], posenc=posenc).mean(dim=1)
                z_j = model(batch["x_j"], mask=batch["mask_j"], posenc=posenc).mean(dim=1)
                loss, logs = total_loss(z_i, z_j, projector=projector, cfg=loss_cfg)

            (scaler.scale(loss) if scaler.is_enabled() else loss).backward()

            if (it + 1) % cfg.grad_accum == 0:
                if scaler.is_enabled(): scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                if ema: ema.update(model)

            running_loss += logs.get("loss.total", 0.0); running_nce += logs.get("nce.loss", 0.0); running_mim += logs.get("mim.loss", 0.0); steps += 1

            # 进度显示：按总体步数估算百分比
            done_steps = (epoch - start_epoch) * steps_per_epoch + (it + 1)
            pct = 100.0 * float(done_steps) / float(total_steps)
            # 每个 epoch 打印 20 次进度或每 50 步打印一次（取较大间隔以降低刷屏）
            interval = max(1, max(steps_per_epoch // 20, 50))
            if (it + 1) % interval == 0 or (it + 1) == steps_per_epoch:
                print(f"[PROG] {done_steps}/{total_steps} ({pct:.1f}%) epoch {epoch+1}/{cfg.epochs} iter {it+1}/{steps_per_epoch}")

            if cfg.dry_run and it >= 10: break

        train_loss, train_nce, train_mim = running_loss/steps, running_nce/steps, running_mim/steps

        # Eval
        model.eval(); projector.eval()
        if ema: ema.apply_to(model)
        E, ids = embed_loader(model, val_loader, device)
        metrics = eval_retrieval(E, ids, k_list=[1, 5, 10])
        val_recall5 = metrics.get("recall@5", 0.0)
        if ema: load_ckpt(run_dir / "last.ckpt", model=model, optimizer=optim, scheduler=sched, projector=projector, ema=None) # restore non-ema

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wr = csv.writer(f); wr.writerow([epoch, f"{train_loss:.6f}", f"{train_nce:.6f}", f"{train_mim:.6f}", f"{metrics.get('recall@1',0):.6f}", f"{val_recall5:.6f}", f"{metrics.get('recall@10',0):.6f}", f"{metrics.get('mAP',0):.6f}", f"{sched.get_last_lr()[0]:.6e}"])
        plot_metrics_csv(csv_path, run_dir / "metrics.png")

        save_ckpt(run_dir / "last.ckpt", model=model, optimizer=optim, scheduler=sched, epoch=epoch + 1, best_metric=best_metric, projector=projector, ema=ema, cfg=cfg, run_id=run_id)
        improved = val_recall5 > best_metric
        if improved:
            best_metric = val_recall5; no_improve = 0
            save_ckpt(run_dir / "best.ckpt", model=model, optimizer=optim, scheduler=sched, epoch=epoch + 1, best_metric=best_metric, projector=projector, ema=ema, cfg=cfg, run_id=run_id)
            register_artifact(run_id=run_id, kind="prior_ckpt", path=str(run_dir / "best.ckpt"), attrs={"ym": cfg.ym, "recall@5": best_metric, "epoch": epoch})
        else: no_improve += 1

        logger.info(f"epoch {epoch}: loss={train_loss:.4f} nce={train_nce:.4f} mim={train_mim:.4f} R@5={val_recall5:.4f} mAP={metrics.get('mAP',0):.4f} lr={sched.get_last_lr()[0]:.2e} {'*BEST' if improved else ''}")
        if no_improve >= cfg.patience or (cfg.dry_run and epoch >= 1): break

    return {"run_dir": str(run_dir), "best_recall@5": float(best_metric)}


__all__ = ["TrainerConfig", "train_loop"]
