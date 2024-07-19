import argparse
import glob
import os.path

from ultralytics import YOLO


parser = argparse.ArgumentParser('Test bbox script', add_help=False)
parser.add_argument('--pretrained-path', required=True, type=str)
args = parser.parse_args()

# Load a pretrained YOLO model (recommended for training)
model = YOLO(args.pretrained_path)

images = glob.glob(os.path.join('assets', 'test_imgs', '*.jpg'))

for image in images:
    result = model.predict(image, imgsz=(384, 640), rect=True,  iou=0.1, max_det=10, conf=0.7)
    for item in result:
        item.show()
