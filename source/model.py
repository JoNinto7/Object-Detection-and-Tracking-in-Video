from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image

from parse_config import * 
from utils import build_targets
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs):
    """Create PyTorch modules from parsed .cfg definitions"""
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()

    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        mtype = module_def["type"]

        if mtype == "convolutional":
            bn = int(module_def.get("batch_normalize", 0))
            filters = int(module_def["filters"])
            ksize = int(module_def["size"])
            pad = (ksize - 1) // 2 if int(module_def.get("pad", 0)) else 0

            conv = nn.Conv2d(output_filters[-1], filters, ksize, int(module_def["stride"]), pad, bias=not bn)
            modules.add_module(f"conv_{i}", conv)
            if bn:
                modules.add_module(f"bn_{i}", nn.BatchNorm2d(filters))
            if module_def.get("activation") == "leaky":
                modules.add_module(f"leaky_{i}", nn.LeakyReLU(0.1))

        elif mtype == "maxpool":
            k, s = int(module_def["size"]), int(module_def["stride"])
            if k == 2 and s == 1:
                modules.add_module(f"pad_{i}", nn.ZeroPad2d((0, 1, 0, 1)))
            modules.add_module(f"maxpool_{i}", nn.MaxPool2d(k, s, (k - 1) // 2))

        elif mtype == "upsample":
            modules.add_module(f"upsample_{i}", nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest"))

        elif mtype == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum(output_filters[l+1] if l > 0 else output_filters[l] for l in layers)
            modules.add_module(f"route_{i}", EmptyLayer())

        elif mtype == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module(f"shortcut_{i}", EmptyLayer())

        elif mtype == "yolo":
            mask_ids = [int(x) for x in module_def["mask"].split(",")]
            all_anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(all_anchors[i], all_anchors[i+1]) for i in range(0, len(all_anchors), 2)]
            selected = [anchors[i] for i in mask_ids]
            yolo = YOLOLayer(selected, int(module_def["classes"]), int(hyperparams["height"]))
            modules.add_module(f"yolo_{i}", yolo)

        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Used as a placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super().__init__()


class YOLOLayer(nn.Module):
    """YOLO detection layer for bounding box prediction"""
    def __init__(self, anchors, num_classes, img_dim):
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes  # x, y, w, h, conf + class
        self.image_dim = img_dim
        self.ignore_thres = 0.5

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        nB, _, nG, _ = x.size()  # Batch, Channels, Grid (assume square)
        nA = self.num_anchors
        stride = self.image_dim / nG

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = (
            x.view(nB, nA, self.bbox_attrs, nG, nG)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        x = torch.sigmoid(prediction[..., 0])  # center x
        y = torch.sigmoid(prediction[..., 1])  # center y
        w = prediction[..., 2]  # width
        h = prediction[..., 3]  # height
        pred_conf = torch.sigmoid(prediction[..., 4])  # objectness score
        pred_cls = torch.sigmoid(prediction[..., 5:])  # class scores

        grid_x = torch.arange(nG, device=x.device).repeat(nG, 1).view([1, 1, nG, nG])
        grid_y = grid_x.permute(0, 1, 3, 2)

        scaled_anchors = FloatTensor([(aw / stride, ah / stride) for aw, ah in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view(1, nA, 1, 1)
        anchor_h = scaled_anchors[:, 1:2].view(1, nA, 1, 1)

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h

        if targets is not None:
            # Generate ground truth targets
            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes.cpu(),
                pred_conf.cpu(),
                pred_cls.cpu(),
                targets.cpu(),
                scaled_anchors.cpu(),
                nA,
                self.num_classes,
                nG,
                self.ignore_thres,
                self.image_dim
            )

            nProposals = (pred_conf > 0.5).sum().item()
            recall = float(nCorrect / nGT) if nGT else 1.0
            precision = float(nCorrect / nProposals) if nProposals else 0.0

            # Convert ground truth tensors
            mask, conf_mask = mask.type(ByteTensor), conf_mask.type(ByteTensor)
            tx, ty, tw, th = (v.type(FloatTensor) for v in (tx, ty, tw, th))
            tconf, tcls = tconf.type(FloatTensor), tcls.type(LongTensor)

            # Compute individual loss terms
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = (
                self.bce_loss(pred_conf[conf_mask - mask], tconf[conf_mask - mask]) +
                self.bce_loss(pred_conf[mask], tconf[mask])
            )
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))

            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return total_loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), recall, precision

        # Inference mode
        output = torch.cat([
            pred_boxes.view(nB, -1, 4) * stride,
            pred_conf.view(nB, -1, 1),
            pred_cls.view(nB, -1, self.num_classes)
        ], dim=-1)
        return output


class Darknet(nn.Module):
    def __init__(self, config_path, img_size=416):
        super().__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

    def forward(self, x, targets=None):
        is_training = targets is not None
        outputs = []
        self.losses = defaultdict(float)
        layer_outputs = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            module_type = module_def["type"]

            if module_type in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_type == "route":
                layers = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[layer_idx] for layer_idx in layers], dim=1)
            elif module_type == "shortcut":
                from_idx = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[from_idx]
            elif module_type == "yolo":
                if is_training:
                    x, *loss_values = module[0](x, targets)
                    for name, loss_val in zip(self.loss_names, loss_values):
                        self.losses[name] += loss_val
                else:
                    x = module(x)
                outputs.append(x)
            layer_outputs.append(x)

        self.losses["recall"] /= 3
        self.losses["precision"] /= 3

        if is_training:
            return sum(outputs)
        else:
            return torch.cat(outputs, dim=1)

    def load_weights(self, weights_path, cutoff=-1):
        if weights_path.endswith("darknet53.conv.74"):
            cutoff = 75

        with open(weights_path, "rb") as fp:
            header = np.fromfile(fp, dtype=np.int32, count=5)
            self.header_info = header
            self.seen = header[3]
            weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def.get("batch_normalize"):
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()
                    # Load BN parameters
                    bn_layer.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias))
                    ptr += num_b
                    bn_layer.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight))
                    ptr += num_b
                    bn_layer.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean))
                    ptr += num_b
                    bn_layer.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var))
                    ptr += num_b
                else:
                    # Load Conv bias
                    num_b = conv_layer.bias.numel()
                    conv_layer.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias))
                    ptr += num_b
                # Load Conv weights
                num_w = conv_layer.weight.numel()
                conv_layer.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight))
                ptr += num_w

    def save_weights(self, path, cutoff=-1):
        with open(path, "wb") as fp:
            self.header_info[3] = self.seen
            self.header_info.tofile(fp)

            for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
                if module_def["type"] == "convolutional":
                    conv_layer = module[0]
                    if module_def.get("batch_normalize"):
                        bn_layer = module[1]
                        bn_layer.bias.data.cpu().numpy().tofile(fp)
                        bn_layer.weight.data.cpu().numpy().tofile(fp)
                        bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                        bn_layer.running_var.data.cpu().numpy().tofile(fp)
                    else:
                        conv_layer.bias.data.cpu().numpy().tofile(fp)
                    conv_layer.weight.data.cpu().numpy().tofile(fp)
