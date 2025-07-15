from __future__ import division

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import csv
import warnings

from torch.utils.data import DataLoader
from model import Darknet
from dataset import KittiDetectionDataset
from train_model import trainmodel

warnings.filterwarnings("ignore")

def main(
    train_path="../data/train/images/",
    val_path="../data/train/images/",
    labels_path="../data/train/labels/",
    weights_path="../checkpoints/",
    output_path="../output",
    config_file="../config/yolov3-kitti.cfg",
    fraction=1,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=2,
    epochs=25,
    freeze=[True, 5]
):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor


    model = Darknet(config_file)
    model.to(device)

    train_set = KittiDetectionDataset(train_path, labels_path, fraction=fraction, train=True)
    val_set = KittiDetectionDataset(val_path, labels_path, fraction=fraction, train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)


    print("Starting training...")

    trainmodel(
        model,
        device,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        weights_path,
        max_epochs=epochs,
        tensor_type=tensor_type,
        update_every=1,
        freeze_backbone=freeze[0],
        freeze_epoch=freeze[1]
    )

    print("Training done.")

if __name__ == "__main__":
    main()
