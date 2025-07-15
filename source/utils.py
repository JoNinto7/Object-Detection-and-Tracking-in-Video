from __future__ import division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def initialize_weights(module):
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
    elif "BatchNorm2d" in classname:
        nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(module.bias.data, 0.0)


def load_class_names(filepath):
    with open(filepath, "r") as file:
        names = file.read().strip().split("\n")
    return names


def compute_average_precision(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        box1_x1, box1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        box1_y1, box1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        box2_x1, box2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        box2_y1, box2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1 + 1, min=0) * torch.clamp(inter_y2 - inter_y1 + 1, min=0)
    box1_area = (box1_x2 - box1_x1 + 1) * (box1_y2 - box1_y1 + 1)
    box2_area = (box2_x2 - box2_x1 + 1) * (box2_y2 - box2_y1 + 1)

    iou = inter_area / (box1_area + box2_area - inter_area + 1e-16)
    return iou


def bbox_iou_numpy(box1, box2):
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    iw = np.maximum(0, np.minimum(np.expand_dims(box1[:, 2], 1), box2[:, 2]) - np.maximum(np.expand_dims(box1[:, 0], 1), box2[:, 0]))
    ih = np.maximum(0, np.minimum(np.expand_dims(box1[:, 3], 1), box2[:, 3]) - np.maximum(np.expand_dims(box1[:, 1], 1), box2[:, 1]))
    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + box2_area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)
    return (iw * ih) / ua


def non_max_suppression(predictions, num_classes, conf_thresh=0.5, nms_thresh=0.4):
    corner_preds = predictions.new(predictions.shape)
    corner_preds[:, :, 0] = predictions[:, :, 0] - predictions[:, :, 2] / 2
    corner_preds[:, :, 1] = predictions[:, :, 1] - predictions[:, :, 3] / 2
    corner_preds[:, :, 2] = predictions[:, :, 0] + predictions[:, :, 2] / 2
    corner_preds[:, :, 3] = predictions[:, :, 1] + predictions[:, :, 3] / 2
    predictions[:, :, :4] = corner_preds[:, :, :4]

    outputs = [None] * len(predictions)

    for img_idx, img_pred in enumerate(predictions):
        mask = img_pred[:, 4] >= conf_thresh
        img_pred = img_pred[mask]
        if not img_pred.size(0):
            continue

        class_conf, class_pred = torch.max(img_pred[:, 5:5 + num_classes], dim=1, keepdim=True)
        detections = torch.cat((img_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        unique_classes = detections[:, -1].cpu().unique()
        if predictions.is_cuda:
            unique_classes = unique_classes.cuda()

        img_output = []
        for cls in unique_classes:
            cls_detections = detections[detections[:, -1] == cls]
            _, conf_sort_idx = torch.sort(cls_detections[:, 4], descending=True)
            cls_detections = cls_detections[conf_sort_idx]

            while cls_detections.size(0):
                img_output.append(cls_detections[0].unsqueeze(0))
                if len(cls_detections) == 1:
                    break
                ious = bbox_iou(img_output[-1], cls_detections[1:])
                cls_detections = cls_detections[1:][ious < nms_thresh]

        if img_output:
            outputs[img_idx] = torch.cat(img_output)

    return outputs


def build_targets(pred_boxes, pred_conf, pred_cls, targets, anchors, num_anchors, num_classes, grid_size, ignore_thresh, img_dim):
    nB, nA, nC, nG = targets.size(0), num_anchors, num_classes, grid_size

    mask = torch.zeros(nB, nA, nG, nG)
    conf_mask = torch.ones(nB, nA, nG, nG)
    tx, ty, tw, th = [torch.zeros(nB, nA, nG, nG) for _ in range(4)]
    tconf = torch.zeros(nB, nA, nG, nG, dtype=torch.uint8)
    tcls = torch.zeros(nB, nA, nG, nG, nC, dtype=torch.uint8)

    nGT, nCorrect = 0, 0

    for b in range(nB):
        for t in range(targets.size(1)):
            if targets[b, t].sum() == 0:
                continue

            nGT += 1
            gx, gy = targets[b, t, 1] * nG, targets[b, t, 2] * nG
            gw, gh = targets[b, t, 3] * nG, targets[b, t, 4] * nG
            gi, gj = int(gx), int(gy)

            gt_box = torch.FloatTensor([[0, 0, gw, gh]])
            anchor_shapes = torch.FloatTensor(np.hstack((np.zeros((len(anchors), 2)), np.array(anchors))))
            anchor_ious = bbox_iou(gt_box, anchor_shapes)

            conf_mask[b, anchor_ious > ignore_thresh, gj, gi] = 0

            best_n = np.argmax(anchor_ious)

            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1

            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)

            target_label = int(targets[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            gt_box = torch.FloatTensor([[gx, gy, gw, gh]])
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)

            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            if iou > 0.5 and pred_label == target_label and pred_conf[b, best_n, gj, gi] > 0.5:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


def to_categorical(y, num_classes):
    return torch.from_numpy(np.eye(num_classes, dtype=np.uint8)[y])
