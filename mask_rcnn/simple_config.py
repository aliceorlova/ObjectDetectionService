import mrcnn.config

class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    NUM_CLASSES = 81
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
