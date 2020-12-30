import mrcnn.model
import os
from Mask_RCNN.simple_config import SimpleConfig
import cv2
import mrcnn.visualize
import imutils
from mrcnn import visualize


CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

config = SimpleConfig()
config.display()
model = mrcnn.model.MaskRCNN(mode="inference",
                             config=SimpleConfig(),
                             model_dir=os.getcwd())
model.load_weights("C:/Users/Alisa/Documents/bachelors/ml/mask_rcnn_coco.h5", by_name=True)
image = cv2.imread("C:/Users/Alisa/Documents/bachelors/ml/img2.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = imutils.resize(image, width=512)
print("[INFO] making predictions with Mask R-CNN...")
result = model.detect([image], verbose=1)
r1 = result[0]
print(r1)

visualize.display_instances(image, r1['rois'], r1['masks'], r1['class_ids'], CLASS_NAMES, r1['scores'])
model.keras_model.summary()



