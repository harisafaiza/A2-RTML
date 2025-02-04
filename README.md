YOLOv4 Implementation and Training

Overview

This project implements YOLOv4 using PyTorch for object detection. It includes model inference, dataset preparation, and training on the COCO dataset.

Inference

Mish Activation: Implemented Mish activation for better gradient flow.

def mish(x):
    return x * torch.tanh(F.softplus(x))

MaxPool Support: Added maxpool layers in create_modules() for downsampling.

Route Layers: Ensured correct concatenation of feature maps.

Pretrained Weights: Function implemented to load YOLOv4 weights.

Training

Dataset: COCO dataset loaded using FiftyOne.

dataset = foz.load_zoo_dataset("coco-2017", split="validation")

Training Pipeline: Implements forward propagation, CIoU loss, and weight updates.

def train_model():
    optimizer = optim.Adam(model.parameters(), lr=0.001)

Fixes: Addressed missing images, route layer issues, and feature map mismatches.

Deliverables

Files: mish.py, train.py, darknet_test.py, detect.py, full_train_yolo4.py

Execution: Training started, with further debugging needed for optimization.

Next Steps

Fine-tune training to improve accuracy.

Optimize hyperparameters for better performance.

