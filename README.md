# YOLOv4 Implementation and Training

## Overview
This project implements YOLOv4 using PyTorch for object detection. It includes model inference, dataset preparation, training, and evaluation on the COCO dataset.

## Inference
- **Mish Activation**: Integrated Mish activation function to enhance gradient flow and improve convergence.
  ```python
  def mish(x):
      return x * torch.tanh(F.softplus(x))
  ```
- **MaxPool Support**: Modified `create_modules()` to handle maxpool layers for effective downsampling.
- **Route Layers**: Ensured proper concatenation of feature maps to maintain feature reuse and enhance detection accuracy.
- **Pretrained Weights**: Implemented weight loading function to initialize the model with YOLOv4 pretrained weights for improved performance.

## Training
- **Dataset Preparation**: COCO dataset loaded using FiftyOne and split into training and validation sets.
  ```python
  dataset = foz.load_zoo_dataset("coco-2017", split="train")
  ```
- **Training Pipeline**:
  - Forward propagation and loss computation using CIoU loss for bounding box regression.
  - Gradient updates using Adam optimizer.
  ```python
  def train_model():
      optimizer = optim.Adam(model.parameters(), lr=0.001)
  ```
- **Fixes & Improvements**:
  - Addressed missing images that caused dataset errors.
  - Resolved feature map size mismatches in route layers.
  - Improved parsing of negative indices for route layers to prevent missing connections.

## Evaluation & Results
- Implemented mAP (mean Average Precision) evaluation to measure model performance.
- Detection pipeline tested on sample images to validate object detection accuracy.

## Deliverables
- **Files Included**:
  - `mish.py` (Activation function implementation)
  - `train.py` (Training pipeline)
  - `darknet_test.py` (Model definition and inference support)
  - `detect.py` (Object detection script)
  - `full_train_yolo4.py` (Complete training script with dataset integration)
- **Execution**:
  - Training initiated with batch processing and loss monitoring.
  - Results analyzed, with further debugging and optimizations planned.

## Next Steps
- Fine-tune training hyperparameters to enhance accuracy and reduce overfitting.
- Optimize model performance for real-time inference.
- Implement additional post-processing techniques to refine detections.

