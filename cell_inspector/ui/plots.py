"""Plotly 图像渲染与同步 zoom。"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from ..core.roi import circle_points


def _to_display_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    return image


def build_image_figure(
    image: np.ndarray,
    title: str = "",
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
    circle_radius: float = 20.0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    show_bbox: bool = True,
    x_range: Optional[tuple[float, float]] = None,
    y_range: Optional[tuple[float, float]] = None,
    uirevision: str = "sync",
    height: int = 320,
) -> go.Figure:
    """构建带定位 overlay 的 plotly 图像。"""
    rgb = _to_display_rgb(image)
    h, w = rgb.shape[:2]

    fig = go.Figure()
    fig.add_trace(
        go.Image(z=rgb, hoverinfo="skip")
    )

    shapes = []
    if center_x is not None and center_y is not None:
        shapes.append(
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=center_x - circle_radius,
                y0=center_y - circle_radius,
                x1=center_x + circle_radius,
                y1=center_y + circle_radius,
                line=dict(color="red", width=2),
                fillcolor="rgba(255,0,0,0)",
            )
        )
        cx_pts, cy_pts = circle_points(center_x, center_y, circle_radius)
        fig.add_trace(
            go.Scatter(
                x=cx_pts,
                y=cy_pts,
                mode="lines",
                line=dict(color="red", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if show_bbox and bbox is not None:
        x1, y1, x2, y2 = bbox
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=x1,
                y0=y1,
                x1=x2,
                y1=y2,
                line=dict(color="cyan", width=1, dash="dot"),
                fillcolor="rgba(0,255,255,0)",
            )
        )

    layout = dict(
        title=dict(text=title, font=dict(size=12)),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            range=x_range if x_range else [-0.5, w - 0.5],
            constrain="domain",
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            range=y_range if y_range else [h - 0.5, -0.5],
            constrain="domain",
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        shapes=shapes,
        uirevision=uirevision,
        dragmode="pan",
        height=height,
        newshape=dict(line=dict(color="red")),
    )
    fig.update_layout(**layout)
    return fig


def build_crop_figure(
    crop: np.ndarray,
    title: str = "",
    scale: int = 4,
    height: int = 280,
) -> go.Figure:
    rgb = _to_display_rgb(crop)
    h, w = rgb.shape[:2]
    fig = go.Figure(go.Image(z=rgb))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(visible=False, range=[-0.5, w - 0.5]),
        yaxis=dict(visible=False, range=[h - 0.5, -0.5], scaleanchor="x"),
        height=height,
    )
    return fig


def extract_relayout_ranges(relayout: dict, img_h: int, img_w: int) -> tuple:
    """从 plotly relayout 事件解析 zoom 范围。"""
    if not relayout:
        return None, None
    if relayout.get("xaxis.autorange") or relayout.get("yaxis.autorange"):
        return None, None
    xr = relayout.get("xaxis.range")
    yr = relayout.get("yaxis.range")
    if xr is None and "xaxis.range[0]" in relayout:
        xr = [relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]]
    if yr is None and "yaxis.range[0]" in relayout:
        yr = [relayout["yaxis.range[0]"], relayout["yaxis.range[1]"]]
    if xr and yr:
        return tuple(xr), tuple(yr)
    return None, None


def apply_sync_ranges(
    fig: go.Figure,
    x_range: Optional[tuple],
    y_range: Optional[tuple],
) -> go.Figure:
    if x_range:
        fig.update_xaxes(range=list(x_range))
    if y_range:
        fig.update_yaxes(range=list(y_range))
    return fig
