"""Streamlit session state 初始化。"""

from __future__ import annotations

import streamlit as st

from ..core.labeling import LabelHistory


def init_session_state() -> None:
    defaults = {
        "initialized": True,
        "current_idx": 0,
        "view_indices": None,
        "roi_size": 256,
        "show_bbox": True,
        "circle_radius": 20.0,
        "crop_scale": 4,
        "crop_interp": "nearest",
        "compare_zoom": 2,
        "gallery_grid": 3,
        "mode": "Inspector",
        "unlabeled_only": False,
        "filter_image_id": "",
        "fluor_gamma": 1.0,
        "fluor_contrast": 1.0,
        "fluor_brightness": 0.0,
        "fluor_cmap": "green",
        "sync_x_range": None,
        "sync_y_range": None,
        "last_hotkey": None,
        "label_history": LabelHistory(),
        "atlas_enabled": True,
        "fluo_verify_active": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
