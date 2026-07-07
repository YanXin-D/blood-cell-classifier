"""Cell Inspector — 单细胞多视图标注工作站。"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from cell_inspector.core.atlas import AtlasManager
from cell_inspector.core.image_cache import GLOBAL_CACHE
from cell_inspector.core.metadata import MetadataStore, resolve_metadata_path
from cell_inspector.ui.navigation import (
    apply_label,
    ensure_view_indices,
    go_next,
    go_prev,
    go_random,
    handle_hotkey,
    jump_to_cell_id,
    rebuild_view_indices,
    render_progress_bar,
    undo_label,
)
from cell_inspector.ui.panels import (
    render_compare_mode,
    render_fluo_verify_mode,
    render_gallery_mode,
    render_inspector_layout,
)
from cell_inspector.ui.shortcuts import render_hotkey_listener
from cell_inspector.ui.state import init_session_state

st.set_page_config(
    page_title="Cell Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_METADATA = ROOT / "sample_data" / "metadata.csv"
FETAL_METADATA = (
    ROOT.parent / "baseline_v0" / "data" / "crops" / "fetal_culture" / "metadata.csv"
)
BLOOD_METADATA = (
    ROOT.parent / "baseline_v0" / "data" / "crops" / "blood_cell" / "metadata.csv"
)


def default_metadata_path() -> Path:
    if BLOOD_METADATA.is_file():
        return BLOOD_METADATA
    if FETAL_METADATA.is_file():
        return FETAL_METADATA
    return DEFAULT_METADATA


@st.cache_resource
def load_store(metadata_path: str) -> MetadataStore:
    return MetadataStore(Path(metadata_path))


def prefetch_neighbors(store: MetadataStore, indices: list[int], pos: int) -> None:
    """预取 current / prev / next 的全部视图图像。"""
    if not indices:
        return
    paths = []
    for offset in (-1, 0, 1):
        p = pos + offset
        if 0 <= p < len(indices):
            row = store.get_row(indices[p])
            paths.extend(
                [
                    row["crop_path"],
                    row["bf_path"],
                    row["phase_path"],
                    row["fluor_path"],
                ]
            )
    GLOBAL_CACHE.prefetch(paths)


def render_sidebar() -> tuple[str, AtlasManager]:
    st.sidebar.title("Cell Inspector")
    st.sidebar.caption("单细胞多视图标注工作站")

    default_meta = st.session_state.get("_meta_path", str(default_metadata_path()))
    meta_path = st.sidebar.text_input(
        "metadata.csv 路径（可填目录，自动找 metadata.csv）",
        value=default_meta,
    )
    if resolve_metadata_path(meta_path) != resolve_metadata_path(
        st.session_state.get("_meta_path", "")
    ):
        st.session_state._meta_path = meta_path
        load_store.clear()

    atlas_root = st.sidebar.text_input("Atlas 目录", value=str(ROOT / "atlas"))
    st.session_state.atlas_enabled = st.sidebar.checkbox(
        "启用 Atlas 自动归档", value=st.session_state.atlas_enabled
    )
    atlas = AtlasManager(Path(atlas_root), enabled=st.session_state.atlas_enabled)

    st.sidebar.markdown("---")
    st.session_state.mode = st.sidebar.selectbox(
        "视图模式",
        ["Inspector", "Compare", "Gallery", "Fluo Verify"],
        index=["Inspector", "Compare", "Gallery", "Fluo Verify"].index(st.session_state.mode),
    )

    st.sidebar.markdown("**ROI 设置**")
    st.session_state.roi_size = st.sidebar.selectbox(
        "ROI 尺寸", [256, 512, 1024], index=[256, 512, 1024].index(st.session_state.roi_size)
    )
    st.session_state.show_bbox = st.sidebar.checkbox("显示 bbox", st.session_state.show_bbox)
    if st.sidebar.button("重置 zoom"):
        st.session_state.sync_x_range = None
        st.session_state.sync_y_range = None

    st.sidebar.markdown("**Crop 设置**")
    st.session_state.crop_scale = st.sidebar.selectbox("放大倍数", [2, 4, 8], index=1)
    st.session_state.crop_interp = st.sidebar.selectbox("插值", ["nearest", "bicubic"])

    if st.session_state.mode == "Compare":
        st.session_state.compare_zoom = st.sidebar.selectbox("Compare 放大", [2, 4, 8], index=0)
    if st.session_state.mode == "Gallery":
        st.session_state.gallery_grid = st.sidebar.selectbox("Gallery 网格", [3, 5], index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
**快捷键**

| 键 | 动作 |
|---|---|
| F | 胎儿细胞 |
| B | 血细胞 |
| S | 跳过 |
| U | 撤销 |
| ← → | 切换 |
| R | 随机 |
"""
    )
    return meta_path, atlas


def render_toolbar(store: MetadataStore, atlas: AtlasManager, indices: list[int]) -> None:
    history = st.session_state.label_history
    c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1, 1.5, 1.5, 1])

    with c1:
        if st.button("◀ 上一张", use_container_width=True):
            go_prev(indices)
            st.rerun()
    with c2:
        if st.button("下一张 ▶", use_container_width=True):
            go_next(indices)
            st.rerun()
    with c3:
        if st.button("随机 R", use_container_width=True):
            go_random(indices)
            st.rerun()
    with c4:
        if st.button("随机未标注", use_container_width=True):
            st.session_state.unlabeled_only = True
            indices = rebuild_view_indices(store)
            st.session_state.view_indices = indices
            if indices:
                go_random(indices)
            st.rerun()

    with c5:
        jump_id = st.text_input("跳转 cell_id", key="jump_cell_id", label_visibility="collapsed", placeholder="cell_id")
        if st.button("跳转", key="jump_btn"):
            if jump_id and jump_to_cell_id(store, indices, jump_id.strip()):
                st.rerun()

    with c6:
        st.session_state.unlabeled_only = st.checkbox("仅未标注 (label=-1)", st.session_state.unlabeled_only)
        st.session_state.filter_image_id = st.text_input(
            "搜索 image_id",
            value=st.session_state.filter_image_id,
            placeholder="image_id",
        )

    with c7:
        if st.button("F 胎儿", type="primary", use_container_width=True):
            apply_label(store, atlas, history, 1)
            st.rerun()
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("B 血", use_container_width=True):
                apply_label(store, atlas, history, 0)
                st.rerun()
        with bc2:
            if st.button("S 跳过", use_container_width=True):
                apply_label(store, atlas, history, -1)
                st.rerun()


def main() -> None:
    init_session_state()

    meta_path, atlas = render_sidebar()
    resolved_meta = resolve_metadata_path(meta_path)
    st.session_state._meta_path = str(resolved_meta)

    if not resolved_meta.is_file():
        st.error(f"未找到 metadata: {resolved_meta}")
        st.info(
            "示例: F:\\method_2\\baseline_v0\\data\\crops\\fetal_culture\\metadata.csv\n"
            "也可只填目录: ...\\fetal_culture"
        )
        st.stop()

    store = load_store(str(resolved_meta))

    st.session_state.view_indices = rebuild_view_indices(store)
    indices = ensure_view_indices(store)

    render_progress_bar(store, indices)
    render_toolbar(store, atlas, indices)

    hotkey = render_hotkey_listener()
    if hotkey:
        last = st.session_state.get("_last_hotkey_val", "")
        cur = st.session_state.get("_hotkey_capture", "")
        if cur != last:
            st.session_state._last_hotkey_val = cur
            handle_hotkey(hotkey, store, atlas, st.session_state.label_history, indices)
            st.rerun()

    if not indices:
        st.stop()

    pos = indices.index(st.session_state.current_idx) if st.session_state.current_idx in indices else 0
    prefetch_neighbors(store, indices, pos)
    row = store.get_row(st.session_state.current_idx)

    mode = st.session_state.mode
    if mode == "Inspector":
        render_inspector_layout(store, row)
    elif mode == "Compare":
        render_compare_mode(row)
    elif mode == "Gallery":
        render_gallery_mode(store, row, st.session_state.label_history)
    elif mode == "Fluo Verify":
        render_fluo_verify_mode(row)


if __name__ == "__main__":
    main()
