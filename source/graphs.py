import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import torch
import cv2
from io import BytesIO
from model import Darknet
from utils import non_max_suppression, load_class_names
from sort import Sort

# ===== CONFIGURATION =====
TEST_ROOT = '../data/test/images/'
WEIGHTS_PATH = '../checkpoints/trained_model.pth'
CONFIG_PATH = '../config/yolov3-kitti.cfg'
NAMES_FILE = '../data/names.txt'

# ===== HELPER FUNCTIONS =====
def load_class_names(names_path):
    with open(names_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def select_sequence(test_root):
    sequences = sorted(os.listdir(test_root))
    sequences = [seq for seq in sequences if os.path.isdir(os.path.join(test_root, seq))]

    print("Available sequences:")
    for idx, seq in enumerate(sequences):
        print(f"[{idx}] {seq}")

    choice = input("Enter the sequence name you want to analyze (e.g., 0000): ").strip()
    if choice not in sequences:
        raise ValueError(f"Sequence '{choice}' not found in {test_root}.")
    return choice


def detect_and_collect(sequence, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(CONFIG_PATH, img_size=416).to(device)
    model.load_weights(WEIGHTS_PATH)
    model.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    seq_path = os.path.join(TEST_ROOT, sequence)
    img_files = sorted([f for f in os.listdir(seq_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    tracker = Sort()
    class_counts = defaultdict(int)
    id_per_class = defaultdict(set)

    for img_file in img_files:
        img_path = os.path.join(seq_path, img_file)
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape

        pad1, pad2 = abs(h - w) // 2, abs(h - w) - abs(h - w) // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        padded = np.pad(img, pad, mode='constant', constant_values=127.5) / 255.
        resized = torch.from_numpy(np.transpose(padded, (2, 0, 1))).unsqueeze(0).float()
        resized = torch.nn.functional.interpolate(resized, size=(416, 416), mode='bilinear', align_corners=False)
        input_tensor = resized.type(Tensor)

        with torch.no_grad():
            detections = model(input_tensor)
            detections = non_max_suppression(detections, num_classes=len(classes), conf_thresh=0.8, nms_thresh=0.4)

        detections = detections[0]

        if detections is not None:
            dets = []
            pad_x = max(img.shape[0] - img.shape[1], 0) * (416 / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * (416 / max(img.shape))
            unpad_h = 416 - pad_y
            unpad_w = 416 - pad_x

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                dets.append([x1.item(), y1.item(), (x1 + box_w).item(), (y1 + box_h).item(), conf.item(), int(cls_pred.item())])

            dets = np.array(dets)
            tracked_objects = tracker.update(dets[:, :5])

            for trk, cls_pred in zip(tracked_objects, dets[:, 5]):
                x1, y1, x2, y2, track_id = trk
                class_name = classes[int(cls_pred)]
                class_counts[class_name] += 1
                id_per_class[class_name].add(int(track_id))

    return class_counts, id_per_class


def plot_statistics(class_counts, id_per_class):
    classes = list(class_counts.keys())
    detections = [class_counts[c] for c in classes]
    unique_ids = [len(id_per_class[c]) for c in classes]

    plt.figure(figsize=(10, 5))
    plt.bar(classes, detections, color='skyblue')
    plt.title('Number of Detections per Class')
    plt.xlabel('Class')
    plt.ylabel('Detections')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('detections_per_class.png', dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(classes, unique_ids, color='lightgreen')
    plt.title('Unique Track IDs per Class')
    plt.xlabel('Class')
    plt.ylabel('Unique IDs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('ids_per_class.png', dpi=300)
    plt.show()


# ===== MAIN SCRIPT =====
if __name__ == '__main__':
    classes = load_class_names(NAMES_FILE)
    sequence = select_sequence(TEST_ROOT)
    class_counts, id_per_class = detect_and_collect(sequence, classes)
    plot_statistics(class_counts, id_per_class)

    print("Detection and Tracking Analysis Complete!")