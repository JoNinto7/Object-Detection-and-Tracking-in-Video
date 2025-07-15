import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

if not hasattr(np, 'bool'):
    np.bool = np.bool_

import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage.transform import resize
from imgaug import augmenters as iaa
import random
import glob

class KittiDetectionDataset(Dataset):
    """
    Custom Dataset for KITTI 2D Object Detection in YOLO Format.
    Converts each sequence into images and bounding boxes with YOLO-compatible labels.
    """
    def __init__(self, image_dir, label_dir, image_size=(416, 416), max_objects=50,
                 use_augmentation=True, fraction=1.0, split_ratio=0.8, train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.max_objects = max_objects
        self.fraction = fraction
        self.split_ratio = split_ratio
        self.train = train
        self.use_augmentation = use_augmentation

        self.image_filenames = []
        self.state = {"w": 0, "h": 0, "pad": (0, 0), "padded_w": 0, "padded_h": 0}
        self._load_image_filenames()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_filenames[index])
        image = self._load_image(image_path)
        label = self._load_label(self.image_filenames[index])
        return image_path, image, label

    def _load_image(self, path):
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        np_img = np.asarray(img)
        if self.train and self.use_augmentation:
            np_img = self._apply_augmentation(np_img)
        img_tensor = torch.from_numpy(self._pad_and_resize(np_img)).float()
        return img_tensor

    def _pad_and_resize(self, image):
        h, w, _ = image.shape
        pad_amount = abs(h - w) // 2
        if h < w:
            pad = ((pad_amount, 0), (0, 0), (0, 0))
        else:
            pad = ((0, 0), (pad_amount, 0), (0, 0))

        padded = np.pad(image, pad, mode='constant', constant_values=128) / 255.0
        self.state.update({
            "w": w, "h": h, "pad": pad,
            "padded_h": padded.shape[0], "padded_w": padded.shape[1]
        })
        resized = resize(padded, (*self.image_size, 3), mode='reflect')
        return np.transpose(resized, (2, 0, 1))

    def _load_label(self, image_filename):
        base = image_filename.split(".")[0]
        label_file = os.path.join(self.label_dir, f"{base}.txt")
        labels = np.zeros((self.max_objects, 5))

        if os.path.exists(label_file):
            raw = np.loadtxt(label_file).reshape(-1, 5)
            w, h = self.state["w"], self.state["h"]
            pw, ph = self.state["padded_w"], self.state["padded_h"]
            pad = self.state["pad"]

            x1 = w * (raw[:, 1] - raw[:, 3] / 2)
            y1 = h * (raw[:, 2] - raw[:, 4] / 2)
            x2 = w * (raw[:, 1] + raw[:, 3] / 2)
            y2 = h * (raw[:, 2] + raw[:, 4] / 2)

            x1 += pad[1][0]; y1 += pad[0][0]; x2 += pad[1][0]; y2 += pad[0][0]

            raw[:, 1] = ((x1 + x2) / 2) / pw
            raw[:, 2] = ((y1 + y2) / 2) / ph
            raw[:, 3] *= w / pw
            raw[:, 4] *= h / ph

            labels[:len(raw)] = raw[:self.max_objects]

        return torch.tensor(labels, dtype=torch.float32)

    def _load_image_filenames(self):
        self.image_filenames = []
        for root, _, files in os.walk(self.image_dir):
            for file in sorted(files):
                if file.endswith((".png", ".jpg", ".jpeg")):
                    relative_path = os.path.relpath(os.path.join(root, file), self.image_dir)
                    self.image_filenames.append(relative_path)

        n = int(len(self.image_filenames) * self.fraction)
        self.image_filenames = self.image_filenames[:n]
        
        if self.train:
            split = int(len(self.image_filenames) * self.split_ratio)
            self.image_filenames = self.image_filenames[:split]
        else:
            split = int(len(self.image_filenames) * self.split_ratio)
            self.image_filenames = self.image_filenames[split:]

    def _apply_augmentation(self, image):
        value = max(0, random.randint(-5, 5))
        augmenter = iaa.Sequential([
            iaa.SomeOf((0, 2), [
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.75)),
                iaa.OneOf([
                    iaa.GaussianBlur((0, 2.0)),
                    iaa.AverageBlur(k=(5, 7)),
                    iaa.MedianBlur(k=(3, 11))
                ]),
                iaa.OneOf([
                    iaa.Multiply((0.8, 1.2), per_channel=0.5),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
                ]),
                iaa.OneOf([
                    iaa.Dropout(p=0.05, per_channel=True),
                    iaa.Crop(px=(0, value))
                ])
            ])
        ])
        return augmenter.augment_image(image)


class ImageFolderDataset(Dataset):
    """
    Dataset for inference: loads images from a folder (no labels)
    Used for evaluation or detect.py
    """
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob(f"{folder_path}/*.*"))
        self.img_shape = (img_size, img_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        pad1, pad2 = abs(h - w) // 2, abs(h - w) - abs(h - w) // 2

        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        padded = np.pad(img, pad, mode='constant', constant_values=127.5) / 255.
        resized = resize(padded, (*self.img_shape, 3), mode='reflect')
        tensor = torch.from_numpy(np.transpose(resized, (2, 0, 1))).float()

        return img_path, tensor
