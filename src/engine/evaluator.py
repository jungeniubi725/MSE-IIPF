"""推理与指标聚合。

用 mask_t 剔除 padding 时间单元，只在有效 unit 上算指标。
同时收集三模态辅助头预测（Krippendorff α 用）与 κ_t（校准相关性用）。
"""

from __future__ import annotations

import numpy as np
import torch

from ..losses.divergence import conflict_kappa
from ..metrics.agreement import modality_agreement
from ..metrics.classification import classification_metrics, per_class_metrics
from ..metrics.uncertainty import uncertainty_report


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: str = "cuda",
    num_classes: int = 8,
    class_names: list[str] | None = None,
    epoch: int = 0,
    amp: bool = True,
    collect_extra: bool = True,
) -> dict:
    """跑完整 loader，返回指标字典 + 原始预测（供出表用）。"""
    model.eval()
    buf: dict[str, list] = {
        "y": [], "yp": [], "p": [], "u": [], "kappa": [],
        "pv": [], "pa": [], "pl": [], "seq_ids": [], "unit_ids": [], "speakers": [],
        "seq_y": [], "seq_yp": [], "seq_p": [], "seq_u": [],
    }
    loss_sum, n_batch = 0.0, 0

    for batch in loader:
        h = {k: batch[k].to(device, non_blocking=True)
             for k in ("h_v", "h_a", "h_sv", "h_sa", "mask_v", "mask_a",
                       "mask_sv", "mask_sa", "mask_t", "labels", "seq_labels")}
        y = h["labels"]
        mt = h["mask_t"]
        with torch.autocast("cuda", enabled=amp and device.startswith("cuda")):
            out, losses = model(
                h["h_v"], h["h_a"], h["h_sv"], h["h_sa"], y, epoch=epoch,
                mask_v=h["mask_v"], mask_a=h["mask_a"],
                mask_sv=h["mask_sv"], mask_sa=h["mask_sa"], mask_t=mt,
                labels_for_itam=_expand_seq_labels(h["seq_labels"], y.shape[1]),
                seq_labels=h["seq_labels"],
            )
        loss_sum += float(losses["loss"].detach())
        n_batch += 1

        keep = mt.reshape(-1).bool().cpu().numpy()
        buf["y"].append(y.reshape(-1).cpu().numpy()[keep])
        buf["yp"].append(out["y_pred"].reshape(-1).cpu().numpy()[keep])
        K = out["p_hat"].size(-1)
        buf["p"].append(out["p_hat"].reshape(-1, K).float().cpu().numpy()[keep])
        if out.get("u") is not None:
            buf["u"].append(out["u"].reshape(-1).float().cpu().numpy()[keep])
        buf["seq_y"].append(h["seq_labels"].reshape(-1).cpu().numpy())
        buf["seq_yp"].append(out["y_seq_pred"].reshape(-1).cpu().numpy())
        buf["seq_p"].append(out["p_seq"].float().cpu().numpy())
        if out.get("u_seq") is not None:
            buf["seq_u"].append(out["u_seq"].reshape(-1).float().cpu().numpy())
        if collect_extra and out.get("p_v") is not None:
            pv, pa, pl = out["p_v"], out["p_a"], out["p_l"]
            buf["pv"].append(pv.argmax(-1).reshape(-1).cpu().numpy()[keep])
            buf["pa"].append(pa.argmax(-1).reshape(-1).cpu().numpy()[keep])
            buf["pl"].append(pl.argmax(-1).reshape(-1).cpu().numpy()[keep])
            k = conflict_kappa(pv.float(), pa.float(), pl.float(), use_sqrt=True)
            buf["kappa"].append(k.reshape(-1).cpu().numpy()[keep])
        buf["seq_ids"].extend(batch["seq_ids"])
        buf["unit_ids"].extend(batch["unit_ids"])
        buf["speakers"].extend(batch["speakers"])

    cat = lambda k: np.concatenate(buf[k]) if buf[k] else None  # noqa: E731
    y_true, y_pred, p_hat = cat("y"), cat("yp"), cat("p")
    if y_true is None or len(y_true) == 0:
        return {"error": "无有效样本"}

    res: dict = {"loss": loss_sum / max(n_batch, 1), "n_units": int(len(y_true))}
    res.update(classification_metrics(y_true, y_pred, num_classes))
    res["per_class"] = per_class_metrics(y_true, y_pred, num_classes, class_names)

    seq_y, seq_yp, seq_p = cat("seq_y"), cat("seq_yp"), cat("seq_p")
    if seq_y is None or seq_yp is None or seq_p is None:
        raise RuntimeError("模型未返回 sequence 级预测，无法计算 seq_macro_f1")
    res["n_sequences"] = int(len(seq_y))
    res.update(classification_metrics(seq_y, seq_yp, num_classes, prefix="seq_"))
    res["seq_per_class"] = per_class_metrics(
        seq_y, seq_yp, num_classes, class_names
    )

    u = cat("u")
    kappa = cat("kappa")
    res["uncertainty"] = uncertainty_report(y_true, y_pred, p_hat, u, kappa)
    seq_u = cat("seq_u")
    res["seq_uncertainty"] = uncertainty_report(
        seq_y, seq_yp, seq_p, seq_u, None
    )

    if buf["pv"]:
        res["agreement"] = modality_agreement(
            cat("pv"), cat("pa"), cat("pl"), y_pred, num_classes
        )
    res["_raw"] = {"y_true": y_true, "y_pred": y_pred, "p_hat": p_hat,
                   "u": u, "kappa": kappa,
                   "pred_v": cat("pv"), "pred_a": cat("pa"), "pred_l": cat("pl"),
                   "seq_y_true": seq_y, "seq_y_pred": seq_yp,
                   "seq_p_hat": seq_p, "seq_u": seq_u,
                   "unit_ids": np.asarray(buf["unit_ids"], dtype=str),
                   "seq_ids": np.asarray(buf["seq_ids"], dtype=str),
                   "speakers": np.asarray(buf["speakers"], dtype=str)}
    return res


def _expand_seq_labels(seq_labels: torch.Tensor, T: int) -> torch.Tensor:
    """(B,) -> (B*T,)，与 collate 的 B×T 展平对齐，供 TMA 伪负样本抑制用。"""
    return seq_labels.unsqueeze(1).expand(-1, T).reshape(-1)


def format_metrics(res: dict, prefix: str = "") -> str:
    """单行日志摘要。"""
    u = res.get("uncertainty", {})
    parts = [
        f"loss {res.get('loss', float('nan')):.4f}",
        f"acc {res.get('acc', 0):.4f}",
        f"maF1 {res.get('macro_f1', 0):.4f}",
        f"wF1 {res.get('weighted_f1', 0):.4f}",
    ]
    if "seq_macro_f1" in res:
        parts.extend([
            f"seqAcc {res.get('seq_acc', 0):.4f}",
            f"seqMaF1 {res.get('seq_macro_f1', 0):.4f}",
        ])
    if "ece" in u:
        parts.append(f"ECE {u['ece']:.4f}")
    if u.get("auroc_u") is not None and not np.isnan(u.get("auroc_u", np.nan)):
        parts.append(f"AUROC(u) {u['auroc_u']:.4f}")
    ag = res.get("agreement", {})
    if ag.get("alpha_val") is not None and not np.isnan(ag["alpha_val"]):
        parts.append(f"α {ag['alpha_val']:.4f}")
    return f"{prefix}" + "  ".join(parts)
