import argparse

from ultralytics import YOLO


parser = argparse.ArgumentParser('Train bbox script', add_help=False)
parser.add_argument('--config-file', required=True, type=str)
parser.add_argument('--project', required=True, type=str)
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--batch-size', type=float, default=0.8)    # 80% of GPU memory
parser.add_argument('--lr', type=float, default=7e-4)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data=args.config_file, epochs=args.epochs, imgsz=(360, 640), rect=True,
                      project=args.project, device=args.device, optimizer='AdamW', batch=args.batch_size,
                      lr0=args.lr, freeze=0)

# Evaluate the model's performance on the validation set
metrics = model.val(imgz=(360, 640), rect=True, iou=0.1, max_det=10, device=args.device,
                    batch=args.batch_size, save_json=True, conf=0.5)
print(f"Eval mAP@50-95: {metrics.box.map}")
print(f"Eval mAP@50: {metrics.box.map50}")
print(f"Eval mAP@75: {metrics.box.map75}")

model.save("yolov8n_bbox.pt")
