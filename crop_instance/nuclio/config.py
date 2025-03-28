# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN

# ... (keep existing header comments) ...

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2

# ---------------------------------------------------------------------------- #
# Model config
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.LOAD_PROPOSALS = False
_C.MODEL.MASK_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

# Path (a file path, or URL like detectron2://.., https://..) to a checkpoint file
_C.MODEL.WEIGHTS = "/home/niqbal/git/aa_transformers/mask2former_forked/weights/model_final.pth"

# Values to be used for image normalization (BGR order)
_C.MODEL.PIXEL_MEAN = [124.578, 97.773, 65.357]
_C.MODEL.PIXEL_STD = [58.341, 48.268, 37.847]

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
_C.MODEL.BACKBONE.FREEZE_AT = 0

# ---------------------------------------------------------------------------- #
# ResNet config
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()
_C.MODEL.RESNETS.DEPTH = 50
_C.MODEL.RESNETS.STEM_TYPE = "basic"  # not used
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
_C.MODEL.RESNETS.STRIDE_IN_1X1 = False
_C.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.RESNETS.NORM = "BN"
_C.MODEL.RESNETS.RES5_MULTI_GRID = [1, 1, 1]  # not used

# ... (keep existing sections for FPN, RPN, ROI HEADS, etc.) ...

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "ADAMW"
_C.SOLVER.BACKBONE_MULTIPLIER = 0.1
_C.SOLVER.MAX_ITER = 10000
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.05
_C.SOLVER.WARMUP_FACTOR = 1.0
_C.SOLVER.WARMUP_ITERS = 0
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.CHECKPOINT_PERIOD = 1000
_C.SOLVER.IMS_PER_BATCH = 2
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupPolyLR"

_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": True})
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.01
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

_C.SOLVER.AMP = CN({"ENABLED": True})

# ---------------------------------------------------------------------------- #
# Input config
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.MIN_SIZE_TRAIN = [int(x * 0.1 * 1024) for x in range(5, 21)]
_C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
_C.INPUT.MIN_SIZE_TEST = 1024
_C.INPUT.MAX_SIZE_TRAIN = 1024
_C.INPUT.MAX_SIZE_TEST = 1024
_C.INPUT.FORMAT = "RGB"
_C.INPUT.DATASET_MAPPER_NAME = "pheno_mask_former_panoptic"

_C.INPUT.CROP = CN({"ENABLED": False})
_C.INPUT.CROP.TYPE = "absolute"
_C.INPUT.CROP.SIZE = (1024, 1024)
_C.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0

_C.INPUT.COLOR_AUG_SSD = True
_C.INPUT.SIZE_DIVISIBILITY = -1

# ---------------------------------------------------------------------------- #
# Dataset config
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ("phenobench_train",)
_C.DATASETS.TEST = ("phenobench_val",)

# ---------------------------------------------------------------------------- #
# DataLoader
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EVAL_PERIOD = 50

_C.TEST.AUG = CN({"ENABLED": False})
_C.TEST.AUG.MIN_SIZES = [512, 768, 1024, 1280, 1536, 1792]
_C.TEST.AUG.MAX_SIZE = 4096
_C.TEST.AUG.FLIP = True

# ---------------------------------------------------------------------------- #
# Output config
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "/mnt/e/trainers/mask2former_panoptic_semantic_phenobench_baseline"

# ... (keep rest of the existing configurations) ...

def base_config():
    return _C.clone()