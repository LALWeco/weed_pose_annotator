# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
import io
import yaml
import numpy as np
from PIL import Image
from ultralytics import YOLO
import debugpy
try:
    debugpy.listen(5678)
except RuntimeError as e:
    # If port is already in use, we can either skip debugging or try another port
    print(f"Debugger initialization failed: {e}")
    pass  # Continue execution without debugger

iou_threshold = 0.2
conf_threshold = 0.7
# "labels": {0: 'soil', 1: 'crop', 2: 'weed'}
def init_context(context):
    context.logger.info("Init detector...")

    pose_checkpoint = "yolov8-pose-1024-cropweed.onnx"
    
    inferencer = YOLO(pose_checkpoint, task='pose')

    context.logger.info("Init labels...")
    with open("/opt/nuclio/function.yaml", "rb") as function_file:
        functionconfig = yaml.safe_load(function_file)
        labels_spec = functionconfig["metadata"]["annotations"]["spec"]
        labels = json.loads(labels_spec)

    context.user_data.labels = labels
    context.user_data.inferencer = inferencer
    context.logger.info("Function initialized")

def handler(context, event):
    context.logger.info("Run YOLOv8pose Weeds model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf).convert("RGB")

    results = []
    pred_instances = context.user_data.inferencer(np.array(image)[...,::-1], conf=conf_threshold, imgsz=1024, iou=iou_threshold)[0]

    for instance_idx in range(pred_instances.boxes.shape[0]):
        box, cls, conf = pred_instances.boxes.xyxy[instance_idx, :], pred_instances.boxes.cls[instance_idx].item(), pred_instances.boxes.conf[instance_idx].item()
        keypoints, keypoint_scores = pred_instances.keypoints.xy[instance_idx, 0, :], pred_instances.keypoints.conf[instance_idx].item()
        label = context.user_data.labels[int(cls)-1] # The class indices are 1-indexed since no background. 
        elements = []
        for element in label["sublabels"]:
            if element["name"] == "center":
                points = [float(keypoints[0]),float(keypoints[1])]
            elif element["name"] == "top_left":
                points = [float(box[0].item()),float(box[1].item())]
            elif element["name"] == "top_right":
                points = [float(box[2].item()),float(box[1].item())]
            elif element["name"] == "bottom_left":
                points = [float(box[0].item()),float(box[3].item())]
            elif element["name"] == "bottom_right":
                points = [float(box[2].item()),float(box[3].item())]
            elements.append({
                "label": element["name"],
                "type": "points",
                "outside": 0 if conf_threshold < keypoint_scores else 1,
                "points": points,
                "confidence": str(keypoint_scores),
            })
        skeleton = {
            "confidence": str(conf),
            "label": label["name"],
            "type": "skeleton",
            "elements": elements
        }

        if not all([element['outside'] for element in skeleton["elements"]]):
            results.append(skeleton)

    return context.Response(body=json.dumps(results), headers={}, content_type="application/json", status_code=200)
