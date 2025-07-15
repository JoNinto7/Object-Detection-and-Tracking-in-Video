from __future__ import division

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import cv2
from io import BytesIO
import time

from model import Darknet
from utils import non_max_suppression, load_class_names
from sort import Sort

def detect(weights_path='../checkpoints/trained_model.pth', config_path='../config/yolov3-kitti.cfg', names_path='../data/names.txt'):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    model = Darknet(config_path, img_size=416).to(device)
    model.load_weights(weights_path)
    model.eval()

    classes = load_class_names(names_path)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    test_root = '../data/test/images/'
    output_root = '../output/'
    os.makedirs(output_root, exist_ok=True)

    sequences = sorted(os.listdir(test_root))
    sequences = [seq for seq in sequences if os.path.isdir(os.path.join(test_root, seq))]

    print("Available sequences:")
    for idx, seq in enumerate(sequences):
        print(f"[{idx}] {seq}")

    choice = input("Enter the sequence name you want to detect (e.g., 0000): ").strip()

    if choice not in sequences:
        print(f"Sequence '{choice}' not found in {test_root}. Exiting.")
        return

    seq = choice
    seq_path = os.path.join(test_root, seq)

    img_files = sorted([f for f in os.listdir(seq_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    total_imgs = len(img_files)
    print(f'Processing sequence {seq} with {total_imgs} images...')

    tracker = Sort()

    first_img = np.array(Image.open(os.path.join(seq_path, img_files[0])))
    height, width, _ = first_img.shape
    video_path = os.path.join(output_root, f'{seq}.mp4')
    out_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

    start_time = time.time()

    for idx, img_file in enumerate(img_files):
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

        fig, ax = plt.subplots(1)
        ax.imshow(img)

        if detections is not None:
            dets = []
            infos = []
            pad_x = max(img.shape[0] - img.shape[1], 0) * (416 / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * (416 / max(img.shape))
            unpad_h = 416 - pad_y
            unpad_w = 416 - pad_x

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                dets.append([x1.item(), y1.item(), (x1 + box_w).item(), (y1 + box_h).item(), conf.item()])
                infos.append((classes[int(cls_pred)], conf.item()))

            dets = np.array(dets)
            tracked_objects = tracker.update(dets)

            for trk, (class_name, conf) in zip(tracked_objects, infos):
                x1, y1, x2, y2, track_id = trk
                bbox = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1), linewidth=1, edgecolor='lime', facecolor='none')
                ax.add_patch(bbox)

                plt.text(
                    x1, y1 - 5,
                    f'{class_name} ID {int(track_id)} ({conf:.2f})',
                    color='black',
                    fontsize=6,
                    bbox=dict(facecolor='lime', alpha=0.5, boxstyle='round,pad=0.2')
                )

        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
        buf.seek(0)
        frame = np.array(Image.open(buf))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (width, height))
        out_video.write(frame)
        plt.close()

        elapsed = time.time() - start_time
        avg_time_per_img = elapsed / (idx + 1)
        eta = avg_time_per_img * (total_imgs - idx - 1)
        print(f'[{idx+1}/{total_imgs}] Processed. ETA: {eta:.1f} sec', end='\r')

    out_video.release()
    print('\nDetection, tracking, and video saving complete.')

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    detect()
