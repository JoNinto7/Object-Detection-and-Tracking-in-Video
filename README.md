# Object Detection and Tracking in Video using Custom CNN and SORT

## Problem Description

This project builds a complete computer vision system that:
- Trains a custom convolutional neural network (CNN) for object detection.
- Detects and tracks multiple objects across frames in video sequences.
- Combines detection with tracking using the SORT (Simple Online and Realtime Tracking) algorithm.

To implement Object Detection and Tracking using a custom CNN and SORT, by using the KITTI dataset as a benchmark.

## Features

- Custom YOLOv3-style object detection model.
- Multi-object tracking using SORT.
- Object class and tracking ID visualization.
- Automatic generation of output videos.

## Data

- KITTI Training Sequences: ~6 GB (RGB images and labels)
- KITTI Test Sequences: ~8.5 GB (RGB images only)

The dataset is organized into train/test folders according to KITTI format.

## Requirements

- torch
- numpy
- pillow
- matplotlib
- opencv-python
- filterpy
- scipy
- tqdm

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Setup Instructions

1. Clone the repository.
2. Install required Python packages.
3. Organize your dataset:
   - Training images: `../data/train/images/`
   - Training labels: `../data/train/labels/`
   - Testing images: `../data/test/images/`
4. Adjust paths inside `detect.py` if necessary.
5. Run the detection script.

## File Structure

```
project/
│
├── checkpoints/               # Saved model weights
├── config/                    # YOLO model .cfg files
├── data/
│   ├── data_tracking_label_2/ # (raw KITTI labels)
│   ├── train/
│   │   ├── images/            # Training images (organized in subfolders)
│   │   └── labels/            # Training labels
│   └── test/images            # Testing sequences (organized in subfolders)
│   └── names.txt              # Class names
│
├── output/                    # Output videos
├── source/                    
│   ├── dataset.py
│   ├── detect.py
│   ├── main.py
│   ├── model.py
│   ├── parse_config.py
│   ├── sort.py
│   ├── test.py
│   ├── train_model.py
│   └── utils.py
│
├── yolo_conv.py              # Conversion utility from KITTI to YOLO
├── requirements.txt
└── README.md
```

## Usage

To run detection and generate video:

```bash
python detect.py
```

The output video will be saved inside the `output/` directory.

## References:

The code has been inspired and adapted from - 

- https://github.com/pytorch/pytorch
- https://github.com/ultralytics/yolov3
- https://github.com/BobLiu20/YOLOv3_PyTorch

## Reference Papers

- YOLOv3: [https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)
- SORT: [https://arxiv.org/abs/1602.00763](https://arxiv.org/abs/1602.00763)
