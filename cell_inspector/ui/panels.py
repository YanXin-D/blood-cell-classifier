"""各视图 panel 渲染。"""

from __future__ import annotations

import streamlit as st

from ..core.fluorescence import adjust_intensity, apply_pseudocolor, compute_fluorescence_stats
from ..core.image_cache import GLOBAL_CACHE
from ..core.roi import extract_roi, roi_local_coords, upscale_crop
from .plots import apply_sync_ranges, build_crop_figure, build_image_figure, extract_relayout_ranges

try:
    from streamlit_plotly_events import plotly_events
except ImportError:
    plotly_events = None


def _load_row_images(row):
    crop = GLOBAL_CACHE.get(row["crop_path"])
    bf = GLOBAL_CACHE.get(row["bf_path"])
    phase = GLOBAL_CACHE.get(row["phase_path"])
    fluor = GLOBAL_CACHE.get(row["fluor_path"])
    return crop, bf, phase, fluor


def _sync_plot(fig, key: str, img_h: int, img_w: int):
    if plotly_events:
        kwargs = {
            "click_event": False,
            "hover_event": False,
            "select_event": False,
            "key": key,
            "override_height": fig.layout.height,
            "override_width": "100%",
        }
        try:
            event = plotly_events(fig, relayout_event=True, **kwargs)
        except TypeError:
            st.plotly_chart(fig, use_container_width=True, key=key)
            return
        if event:
            relayout = event[0] if isinstance(event, list) else event
            if isinstance(relayout, dict) and "xaxis.range[0]" in relayout:
                xr, yr = extract_relayout_ranges(relayout, img_h, img_w)
                if xr and yr:
                    st.session_state.sync_x_range = xr
                    st.session_state.sync_y_range = yr
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def render_source_image_panel(row) -> None:
    """显示 crop 所属全视野原图，并标出细胞位置。"""
    from pathlib import Path

    st.markdown("**所属原图**")
    _, _, phase, _ = _load_row_images(row)
    if phase is None:
        st.warning("原图加载失败")
        return

    crop_path = str(row["crop_path"])
    phase_path = str(row["phase_path"])
    if Path(crop_path).resolve() == Path(phase_path).resolve():
        st.info("该条 metadata 未关联独立原图（crop 与原图路径相同）")
        return

    bbox = (
        float(row["bbox_x1"]),
        float(row["bbox_y1"]),
        float(row["bbox_x2"]),
        float(row["bbox_y2"]),
    )
    fig = build_image_figure(
        phase,
        title=f"{row['image_id']} | {Path(phase_path).name}",
        center_x=float(row["center_x"]),
        center_y=float(row["center_y"]),
        circle_radius=st.session_state.circle_radius,
        bbox=bbox,
        show_bbox=st.session_state.show_bbox,
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("青框 = 分割 bbox，红圈 = 细胞中心")


def render_crop_panel(row) -> None:
    st.markdown("**单细胞 Crop**")
    crop, _, _, _ = _load_row_images(row)
    if crop is None:
        st.error("无法加载 crop")
        return
    scaled = upscale_crop(
        crop,
        scale=st.session_state.crop_scale,
        method=st.session_state.crop_interp,
    )
    title = f"{row['cell_id']} | {row['image_id']}"
    fig = build_crop_figure(scaled, title=title)
    st.plotly_chart(fig, use_container_width=True)


def render_feature_panel(store, row) -> None:
    st.markdown("**形态特征**")
    st.dataframe(store.feature_table(row), hide_index=True, use_container_width=True)
    label_map = {1: "fetal", 0: "blood", -1: "skip"}
    st.caption(f"当前 label: **{label_map.get(int(row['label']), row['label'])}**")


def _prepare_roi_view(full_img, row, title: str, key: str, fluor_adjust: bool = False):
    if full_img is None:
        st.warning(f"{title} 加载失败")
        return
    roi_size = st.session_state.roi_size
    roi, x0, y0 = extract_roi(
        full_img, float(row["center_x"]), float(row["center_y"]), roi_size
    )
    if fluor_adjust:
        roi = adjust_intensity(
            roi,
            gamma=st.session_state.fluor_gamma,
            contrast=st.session_state.fluor_contrast,
            brightness=st.session_state.fluor_brightness,
        )
        roi = apply_pseudocolor(roi, st.session_state.fluor_cmap)

    local = roi_local_coords(
        float(row["center_x"]),
        float(row["center_y"]),
        x0,
        y0,
        float(row["bbox_x1"]),
        float(row["bbox_y1"]),
        float(row["bbox_x2"]),
        float(row["bbox_y2"]),
    )
    bbox = local.get("bbox")
    fig = build_image_figure(
        roi,
        title=title,
        center_x=local["center_x"],
        center_y=local["center_y"],
        circle_radius=st.session_state.circle_radius,
        bbox=bbox,
        show_bbox=st.session_state.show_bbox,
        x_range=st.session_state.sync_x_range,
        y_range=st.session_state.sync_y_range,
    )
    h, w = roi.shape[:2]
    apply_sync_ranges(fig, st.session_state.sync_x_range, st.session_state.sync_y_range)
    _sync_plot(fig, key, h, w)


def render_phase_panel(row) -> None:
    _, _, phase, _ = _load_row_images(row)
    _prepare_roi_view(phase, row, "相位原图 ROI", "phase_sync")


def render_bf_panel(row) -> None:
    _, bf, _, _ = _load_row_images(row)
    _prepare_roi_view(bf, row, "明场原图 ROI", "bf_sync")


def render_fluor_panel(row) -> None:
    _, _, _, fluor = _load_row_images(row)
    _prepare_roi_view(fluor, row, "荧光图 ROI", "fluor_sync", fluor_adjust=True)


def render_fluor_controls() -> None:
    st.markdown("**荧光增强**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.fluor_gamma = st.slider("Gamma", 0.2, 3.0, st.session_state.fluor_gamma, 0.1)
    with c2:
        st.session_state.fluor_contrast = st.slider("Contrast", 0.5, 3.0, st.session_state.fluor_contrast, 0.1)
    with c3:
        st.session_state.fluor_brightness = st.slider("Brightness", -0.5, 0.5, st.session_state.fluor_brightness, 0.05)
    st.session_state.fluor_cmap = st.selectbox("伪彩色", ["green", "red", "hot"], index=["green", "red", "hot"].index(st.session_state.fluor_cmap))


def render_inspector_layout(store, row) -> None:
    col_crop, col_src = st.columns([1, 1.15])
    with col_crop:
        render_crop_panel(row)
        render_feature_panel(store, row)
    with col_src:
        render_source_image_panel(row)

    with st.expander("局部放大 ROI（相位 / 明场 / 荧光）", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            render_phase_panel(row)
        with c2:
            render_bf_panel(row)
        with c3:
            render_fluor_panel(row)
        render_fluor_controls()


def render_compare_mode(row) -> None:
    st.markdown("**Compare Mode** — crop / phase / brightfield / fluorescence")
    zoom = st.session_state.compare_zoom
    crop, bf, phase, fluor = _load_row_images(row)
    cols = st.columns(4)
    panels = [
        ("Crop", crop, False),
        ("Phase ROI", phase, False),
        ("BF ROI", bf, False),
        ("Fluor ROI", fluor, True),
    ]
    roi_size = st.session_state.roi_size
    for col, (title, img, fluor_adj) in zip(cols, panels):
        with col:
            if img is None:
                st.warning(title)
                continue
            if title == "Crop":
                disp = upscale_crop(img, scale=zoom, method=st.session_state.crop_interp)
            else:
                disp, _, _ = extract_roi(
                    img, float(row["center_x"]), float(row["center_y"]), roi_size
                )
                if fluor_adj:
                    disp = adjust_intensity(
                        disp,
                        gamma=st.session_state.fluor_gamma,
                        contrast=st.session_state.fluor_contrast,
                        brightness=st.session_state.fluor_brightness,
                    )
                    disp = apply_pseudocolor(disp, st.session_state.fluor_cmap)
            fig = build_image_figure(disp, title=f"{title} ({zoom}×)")
            st.plotly_chart(fig, use_container_width=True)


def render_gallery_mode(store, row, history) -> None:
    st.markdown("**Gallery Mode** — 当前细胞 vs 最近 fetal / blood 样本")
    grid = st.session_state.gallery_grid
    st.caption(f"网格 {grid}×{grid}")

    def _thumb(idx):
        r = store.get_row(idx)
        c = GLOBAL_CACHE.get(r["crop_path"])
        if c is None:
            return None
        return upscale_crop(c, scale=2, method="nearest")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Candidate**")
        t = _thumb(st.session_state.current_idx)
        if t is not None:
            st.plotly_chart(build_crop_figure(t, title=str(row["cell_id"])), use_container_width=True)
    with c2:
        st.markdown("**Recent Fetal**")
        cols = st.columns(grid)
        for i, idx in enumerate(history.recent_fetal[: grid * grid]):
            with cols[i % grid]:
                t = _thumb(idx)
                if t is not None:
                    st.image(t, use_container_width=True)
    with c3:
        st.markdown("**Recent Blood**")
        cols = st.columns(grid)
        for i, idx in enumerate(history.recent_blood[: grid * grid]):
            with cols[i % grid]:
                t = _thumb(idx)
                if t is not None:
                    st.image(t, use_container_width=True)


def render_fluo_verify_mode(row) -> None:
    st.markdown("**荧光验证模式** — 128×128 ROI 统计")
    _, _, _, fluor = _load_row_images(row)
    if fluor is None:
        st.error("荧光图加载失败")
        return
    stats = compute_fluorescence_stats(
        fluor, float(row["center_x"]), float(row["center_y"]), signal_size=128
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Fluo Mean", f"{stats['fluo_mean']:.2f}")
    m2.metric("Fluo Max", f"{stats['fluo_max']:.2f}")
    m3.metric("SBR", f"{stats['sbr']:.2f}")

    roi = stats["signal_roi"]
    enhanced = apply_pseudocolor(
        adjust_intensity(
            roi,
            gamma=st.session_state.fluor_gamma,
            contrast=st.session_state.fluor_contrast,
            brightness=st.session_state.fluor_brightness,
        ),
        st.session_state.fluor_cmap,
    )
    local = {"center_x": 64.0, "center_y": 64.0}
    fig = build_image_figure(
        enhanced,
        title="128×128 荧光 ROI",
        center_x=local["center_x"],
        center_y=local["center_y"],
        circle_radius=15,
        show_bbox=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        render_phase_panel(row)
    with c2:
        render_bf_panel(row)
