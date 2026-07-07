"""Cellpose 模型加载（支持 cyto2/cyto3 与 Cellpose 4 cpsam）。"""

from __future__ import annotations

CYTO_MODELS = {"cyto", "cyto2", "cyto3", "nuclei"}


def create_segmentation_model(gpu: bool, pretrained_model: str = "cyto3"):
    from cellpose import models

    name = pretrained_model or "cyto3"

    if hasattr(models, "Cellpose"):
        if name in ("cpsam", "default"):
            name = "cyto3"
        if name not in CYTO_MODELS:
            raise ValueError(f"不支持的 cyto 模型: {name}，可选: {sorted(CYTO_MODELS)}")
        return _CytoModel(models.Cellpose(gpu=gpu, model_type=name), name)

    if name in CYTO_MODELS:
        raise RuntimeError(
            f"当前为 Cellpose 4.x，不支持 {name}。"
            "请安装: pip install 'cellpose>=3.1.1,<4'"
        )
    return _CpsamModel(models.CellposeModel(gpu=gpu, pretrained_model="cpsam"))


def segment_image(model, img, seg_params: dict):
    return model.eval(img, seg_params)


class _CytoModel:
    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self.backend = "cellpose3"

    def eval(self, img, seg_params: dict):
        masks, _, _, _ = self.model.eval(
            img,
            diameter=seg_params["diameter"],
            channels=[0, 0],
            flow_threshold=seg_params["flow_threshold"],
            cellprob_threshold=seg_params["cellprob_threshold"],
            min_size=seg_params["min_size"],
        )
        return masks


class _CpsamModel:
    def __init__(self, model):
        self.model = model
        self.name = "cpsam"
        self.backend = "cellpose4"

    def eval(self, img, seg_params: dict):
        masks, _, _ = self.model.eval(
            img,
            diameter=seg_params["diameter"],
            channel_axis=None,
            min_size=seg_params["min_size"],
            cellprob_threshold=seg_params["cellprob_threshold"],
            flow_threshold=seg_params["flow_threshold"],
        )
        return masks
