"""项目级常量：类别映射、文件扩展名、数据源路径。"""

from __future__ import annotations

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})

# blood=0, fetal=1
CLASS_MAP = {"blood": 0, "fetal": 1}
LABEL_TO_DIR = {0: "blood", 1: "fetal"}

# 标注 crop 数据源（用于 rebuild_train_val / export_val_predictions）
TRAINING_SOURCES = {
    "fetal_culture": {
        "metadata": "data/crops/fetal_culture/metadata.csv",
        "labels": "data/crops/fetal_culture/labels.csv",
        "target_class": 1,
    },
    "blood_cell": {
        "metadata": "data/crops/blood_cell/metadata.csv",
        "labels": "data/crops/blood_cell/labels.csv",
        "target_class": 0,
    },
}

# Cellpose 分割批次默认路径
SEGMENT_BATCH_PATHS = {
    "fetal_culture": {
        "input_dir": "data/raw/fetal_culture",
        "crops_dir": "data/crops/fetal_culture",
        "qc_dir": "outputs/segmentation_qc/fetal_culture",
    },
    "blood_cell": {
        "input_dir": "data/raw/blood_cell",
        "crops_dir": "data/crops/blood_cell",
        "qc_dir": "outputs/segmentation_qc/blood_cell",
    },
}

HARD_NEGATIVES_METADATA = "data/all_cells_large/metadata.csv"
