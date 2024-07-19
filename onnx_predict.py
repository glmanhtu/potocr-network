import argparse

import cv2
import numpy as np
import onnxruntime

parser = argparse.ArgumentParser('Onnx network prediction', add_help=False)
parser.add_argument('--bbox-network', required=True, type=str)
parser.add_argument('--img-path', required=True, type=str)
args = parser.parse_args()

x = cv2.imread(args.img_path)
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
x = np.expand_dims(x, axis=0)
ort_inputs = {
    "images": x,
}
ort_pre_session = onnxruntime.InferenceSession(args.bbox_network)
bboxes = ort_pre_session.run(None, ort_inputs)
for bbox in bboxes[0]:
    x1, y1, x2, y2, conf, cls = bbox
    # Draw the bounding box on the image using OpenCV
    cv2.rectangle(x[0], (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
cv2.imshow('image', x[0])
cv2.waitKey(0)
