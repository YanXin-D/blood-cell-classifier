"""推理后处理：候选过滤、metadata 合并。"""

from __future__ import annotations

import pandas as pd


def cell_id_from_name(cell_name: str) -> str:
    return cell_name.replace(".png", "")


def merge_predictions_with_metadata(meta: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    pred["cell_id"] = pred["cell_name"].map(cell_id_from_name)
    return meta.merge(pred, on="cell_id", how="left")


def filter_candidates(
    meta: pd.DataFrame,
    pred: pd.DataFrame | None,
    det: dict,
    *,
    use_model: bool,
) -> pd.DataFrame:
    if pred is not None:
        df = merge_predictions_with_metadata(meta, pred)
        df["probability"] = df["probability"].fillna(0.0)
        df["prediction"] = df["prediction"].fillna(0).astype(int)
    else:
        df = meta.copy()
        df["probability"] = 0.0
        df["prediction"] = 0
        df["cell_name"] = df["cell_id"] + ".png"

    hits = df.copy()
    if use_model:
        min_prob = det.get("min_probability", 0.5)
        hits = hits[hits["probability"] >= min_prob]

    if det.get("morph_filter", False):
        hits = hits[hits["area"] >= det.get("min_area", 350)]

    return hits.sort_values(
        ["probability", "area"], ascending=[False, False]
    ).reset_index(drop=True)


def describe_filters(det: dict, use_model: bool) -> str:
    parts = []
    if use_model:
        parts.append(f"prob>={det.get('min_probability', 0.5)}")
    if det.get("morph_filter", False):
        parts.append(f"area>={det.get('min_area', 350)}")
    return ", ".join(parts) if parts else "无过滤"
