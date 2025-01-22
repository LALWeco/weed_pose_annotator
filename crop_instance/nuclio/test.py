from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from model_handler import add_maskformer2_config, add_deeplab_config
from detectron2.engine.defaults import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from PIL import Image
import numpy as np
import torch

cfg = get_cfg()
add_maskformer2_config(cfg)
add_deeplab_config(cfg)
cfg.merge_from_file("/opt/nuclio/mask2former_phenobench.yaml")
cfg.DATASETS.TRAIN = ('phenobench_train',)
cfg.DATASETS.TEST = ('phenobench_val',)
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.WEIGHTS = "/opt/nuclio/model_final_synthetic.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
model = build_model(cfg)
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)
model.eval()
aug = T.ResizeShortestEdge(
    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
)
input_format = cfg.INPUT.FORMAT
assert input_format in ["RGB", "BGR"], input_format
image = Image.open("/opt/nuclio/sample.png")
height, width = image.size
image = np.array(image)
image = aug.get_transform(image).apply_image(image)
image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
image.to(cfg.MODEL.DEVICE)
inputs = {"image": image, "height": height, "width": width}
predictions = model([inputs])[0]
print(predictions)