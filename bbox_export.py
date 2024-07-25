import argparse
import os.path

import cv2
import torch
from ultralytics import YOLO

from strhub.data.utils import resized_center_pad

parser = argparse.ArgumentParser('Export bbox onnx model script', add_help=False)
parser.add_argument('--pretrained-path', required=True, type=str)
parser.add_argument('--output-path', required=True, type=str)
args = parser.parse_args()


# It is important to pass an input image which has the target object in it
# to ensure that the torchscript is generated correctly for non-max suppression function
x = cv2.imread(os.path.join('assets', 'test_imgs', 'test-18.jpg'))
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
target_shape = (384, 640)
img = resized_center_pad(x, target_shape)

bbox_tensor = torch.from_numpy(img[None, ...]).permute(0, 3, 1, 2) / 255.
# Load a pretrained YOLO model (recommended for training)
model = YOLO(args.pretrained_path)

model.export(format="torchscript", imgsz=target_shape)
