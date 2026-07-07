"""导航与标注动作。"""

from __future__ import annotations

import random

import streamlit as st

from ..core.atlas import AtlasManager
from ..core.labeling import LabelHistory
from ..core.metadata import LABEL_BLOOD, LABEL_FETAL, LABEL_SKIP, MetadataStore


def rebuild_view_indices(store: MetadataStore) -> list[int]:
    return store.filtered_indices(
        unlabeled_only=st.session_state.unlabeled_only,
        image_id=st.session_state.filter_image_id or None,
    )


def ensure_view_indices(store: MetadataStore) -> list[int]:
    indices = rebuild_view_indices(store)
    st.session_state.view_indices = indices
    if not indices:
        st.session_state.current_idx = 0
        return indices
    pos = st.session_state.get("_pos_in_view", 0)
    pos = min(max(0, pos), len(indices) - 1)
    st.session_state.current_idx = indices[pos]
    st.session_state._pos_in_view = pos
    return indices


def pos_in_view(indices: list[int]) -> int:
    cur = st.session_state.current_idx
    if cur in indices:
        return indices.index(cur)
    return st.session_state.get("_pos_in_view", 0)


def go_to_pos(indices: list[int], pos: int) -> None:
    if not indices:
        return
    pos = max(0, min(pos, len(indices) - 1))
    st.session_state._pos_in_view = pos
    st.session_state.current_idx = indices[pos]
    st.session_state.sync_x_range = None
    st.session_state.sync_y_range = None


def go_next(indices: list[int]) -> None:
    go_to_pos(indices, pos_in_view(indices) + 1)


def go_prev(indices: list[int]) -> None:
    go_to_pos(indices, pos_in_view(indices) - 1)


def go_random(indices: list[int]) -> None:
    if not indices:
        return
    pos = random.randint(0, len(indices) - 1)
    go_to_pos(indices, pos)


def jump_to_cell_id(store: MetadataStore, indices: list[int], cell_id: str) -> bool:
    idx = store.idx_from_cell_id(cell_id)
    if idx is None:
        return False
    if idx in indices:
        go_to_pos(indices, indices.index(idx))
        return True
    st.session_state.current_idx = idx
    st.session_state._pos_in_view = 0
    return True


def apply_label(
    store: MetadataStore,
    atlas: AtlasManager,
    history: LabelHistory,
    label: int,
    advance: bool = True,
) -> None:
    idx = st.session_state.current_idx
    row = store.get_row(idx)
    prev = int(row["label"])
    cell_id = row["cell_id"]
    store.save_label(cell_id, label)
    history.record(cell_id, prev, label, idx)
    if label in (LABEL_FETAL, LABEL_BLOOD):
        atlas.archive(row["crop_path"], cell_id, label)
    elif prev in (LABEL_FETAL, LABEL_BLOOD):
        atlas.remove(cell_id)

    if advance:
        indices = st.session_state.view_indices or rebuild_view_indices(store)
        pos = pos_in_view(indices)
        if st.session_state.unlabeled_only:
            indices = rebuild_view_indices(store)
            st.session_state.view_indices = indices
            if indices:
                go_to_pos(indices, min(pos, len(indices) - 1))
        else:
            go_next(indices)


def undo_label(
    store: MetadataStore,
    atlas: AtlasManager,
    history: LabelHistory,
) -> None:
    action = history.pop_undo()
    if action is None:
        return
    store.save_label(action.cell_id, action.prev_label)
    if action.new_label in (LABEL_FETAL, LABEL_BLOOD):
        atlas.remove(action.cell_id)
    if action.prev_label in (LABEL_FETAL, LABEL_BLOOD):
        row = store.get_row(action.idx)
        atlas.archive(row["crop_path"], action.cell_id, action.prev_label)
    indices = st.session_state.view_indices or rebuild_view_indices(store)
    if action.idx in indices:
        go_to_pos(indices, indices.index(action.idx))
    else:
        st.session_state.current_idx = action.idx


def handle_hotkey(
    key: str,
    store: MetadataStore,
    atlas: AtlasManager,
    history: LabelHistory,
    indices: list[int],
) -> None:
    if key == "F":
        apply_label(store, atlas, history, LABEL_FETAL)
    elif key == "B":
        apply_label(store, atlas, history, LABEL_BLOOD)
    elif key == "S":
        apply_label(store, atlas, history, LABEL_SKIP)
    elif key == "U":
        undo_label(store, atlas, history)
    elif key == "LEFT":
        go_prev(indices)
    elif key == "RIGHT":
        go_next(indices)
    elif key == "R":
        go_random(indices)


def render_progress_bar(store: MetadataStore, indices: list[int]) -> None:
    total = len(indices) if indices else store.n_cells
    if not indices:
        st.warning("当前筛选条件下无细胞")
        return
    pos = pos_in_view(indices)
    labeled = store.n_labeled
    pct = labeled / store.n_cells if store.n_cells else 0
    st.progress(pct, text=f"已标注 {labeled}/{store.n_cells} | 当前 {pos + 1}/{total}")
