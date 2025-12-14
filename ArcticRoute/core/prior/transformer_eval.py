from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np  # type: ignore
import torch
from torch.utils.data import DataLoader


def _embed_batch(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    model.eval()
    with torch.no_grad():
        x_i = batch["x_i"]  # [B,T,C]
        m_i = batch["mask_i"]  # [B,T]
        x_j = batch.get("x_j")
        m_j = batch.get("mask_j")
        posenc = batch.get("posenc", None)
        z_i = model(x_i, mask=m_i, posenc=posenc).mean(dim=1)  # [B,D]
        if x_j is not None:
            z_j = model(x_j, mask=m_j, posenc=posenc).mean(dim=1)
            z = torch.cat([z_i, z_j], dim=0)
            ids = batch.get("meta", {}).get("seg_ids", [""] * x_i.size(0))
            ids = [str(s) for s in ids]
            ids = ids + ids
            return z, torch.tensor(list(range(z.size(0))), device=z.device), ids
        ids = batch.get("meta", {}).get("seg_ids", [""] * x_i.size(0))
        ids = [str(s) for s in ids]
        return z_i, torch.tensor(list(range(z_i.size(0))), device=z_i.device), ids


def embed_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, List[str]]:
    all_vecs: List[torch.Tensor] = []
    all_ids: List[str] = []
    for batch in loader:
        # 将 batch 搬到 device
        for k in ("x_i", "mask_i", "x_j", "mask_j"):
            if k in batch and batch[k] is not None:
                batch[k] = batch[k].to(device)
        z, _, ids = _embed_batch(model, batch)
        all_vecs.append(z.detach().to("cpu"))
        all_ids.extend(ids)
    E = torch.cat(all_vecs, dim=0) if all_vecs else torch.zeros(0, model.cfg.d_model)
    return E, all_ids


def eval_retrieval(E: torch.Tensor, ids: Sequence[str], k_list: Sequence[int] = (1, 5, 10), batch_size: int = 2048) -> Dict[str, float]:
    """评估检索指标（Recall@k, mAP@k），支持分块，避免构造 N×N 大矩阵导致 OOM。

    - E: [N,D] tensor（可在 CPU/GPU，但会分批搬运到 CPU 排序）
    - ids: 长度为 N 的 segment_id 列表（或任意可区分同类的键）
    - k_list: 评估的 k 值列表
    - batch_size: 分块大小（根据内存调整）
    """
    N = E.size(0)
    if N == 0:
        return {f"recall@{k}": 0.0 for k in k_list} | {"mAP": 0.0}

    V = torch.nn.functional.normalize(E, dim=-1)

    # 预构建同 id 的索引集合（这里按完整 id，相同 id 视作正样本；如需按 mmsi，请改为 split('_')[0]）
    from collections import defaultdict
    id_to_indices: Dict[str, List[int]] = defaultdict(list)
    str_ids = [str(s) for s in ids]
    for i, sid in enumerate(str_ids):
        id_to_indices[sid].append(i)

    recalls: Dict[int, float] = {k: 0.0 for k in k_list}
    aps: List[float] = []

    # 分块计算相似度并排序（逐行 topK 排序）
    with torch.no_grad():
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            # [B,D] x [D,N] -> [B,N]
            sim_chunk = (V[i:j] @ V.t()).to("cpu")
            # 排除自身匹配
            for r in range(i, j):
                sim_chunk[r - i, r] = -1e9
            # 对分块内每一行做降序排序
            order_chunk = torch.argsort(sim_chunk, dim=1, descending=True).numpy()

            for r in range(i, j):
                sid = str_ids[r]
                pos_set = set(id_to_indices.get(sid, [])) - {r}
                if not pos_set:
                    aps.append(0.0)
                    continue
                order = order_chunk[r - i]
                # Recall@k
                for k in k_list:
                    topk = order[:k]
                    hit = any(idx in pos_set for idx in topk)
                    if hit:
                        recalls[k] += 1.0
                # AP
                hits = 0
                precision_sum = 0.0
                retrieved = 0
                for idx in order:
                    retrieved += 1
                    if idx in pos_set:
                        hits += 1
                        precision_sum += hits / retrieved
                        if hits == len(pos_set):
                            break
                ap = precision_sum / max(1, len(pos_set))
                aps.append(ap)

    metrics = {f"recall@{k}": (recalls.get(k, 0.0) / N if N > 0 else 0.0) for k in k_list}
    metrics["mAP"] = float(np.mean(aps) if aps else 0.0)
    return metrics


__all__ = ["embed_loader", "eval_retrieval"]


