import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from tqdm import tqdm
import gc
from sklearn.metrics import classification_report

from utils import non_max_suppression, bbox_iou_numpy, compute_average_precision

def trainepoch(model, device, optimizer, scheduler, trainloader, tensor_type=torch.cuda.FloatTensor, update_every=16, freeze_backbone=False):
    model.train()
    scheduler.step()

    if freeze_backbone:
        for name, param in model.named_parameters():
            if int(name.split('.')[1]) < 75:
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            if int(name.split('.')[1]) < 75:
                param.requires_grad = True

    optimizer.zero_grad()

    for batch_idx, (img_paths, images, labels) in enumerate(tqdm(trainloader, desc="Training")):
        images = images.type(tensor_type)
        labels = labels.type(tensor_type)

        loss = model(images, labels)
        loss.backward()

        if ((batch_idx + 1) % update_every == 0) or (batch_idx + 1 == len(trainloader)):
            optimizer.step()
            optimizer.zero_grad()

        del images, labels
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[End of Epoch] Total Loss: {loss.item():.4f}")
    return model

def validate(model, device, valloader , tensor_type=torch.cuda.FloatTensor, num_classes=8):
    model.eval()
    model.train(False)

    all_detections = []
    all_annotations = []
    y_true_all = []
    y_pred_all = []

    for batch_idx, (_, images, labels) in enumerate(tqdm(valloader, desc="Validation")):
        images = images.type(tensor_type)

        with torch.no_grad():
            outputs = model(images)
            outputs = non_max_suppression(outputs, num_classes, conf_thresh=0.8, nms_thresh=0.4)

        for output, annotation in zip(outputs, labels):
            all_detections.append([np.array([]) for _ in range(num_classes)])
            if output is not None:
                pred_boxes = output[:, :5].cpu().numpy()
                scores = output[:, 4].cpu().numpy()
                pred_labels = output[:, -1].cpu().numpy()

                y_pred_all.extend(pred_labels.astype(int))

                sort_idx = np.argsort(scores)
                pred_labels = pred_labels[sort_idx]
                pred_boxes = pred_boxes[sort_idx]

                for label in range(num_classes):
                    all_detections[-1][label] = pred_boxes[pred_labels == label]

            all_annotations.append([np.array([]) for _ in range(num_classes)])
            if any(annotation[:, -1] > 0):
                annotation_labels = annotation[annotation[:, -1] > 0, 0].numpy()
                y_true_all.extend(annotation_labels.astype(int))

                annotation_boxes = annotation[annotation[:, -1] > 0, 1:]

                gt_boxes = np.empty_like(annotation_boxes)
                gt_boxes[:, 0] = annotation_boxes[:, 0] - annotation_boxes[:, 2] / 2
                gt_boxes[:, 1] = annotation_boxes[:, 1] - annotation_boxes[:, 3] / 2
                gt_boxes[:, 2] = annotation_boxes[:, 0] + annotation_boxes[:, 2] / 2
                gt_boxes[:, 3] = annotation_boxes[:, 1] + annotation_boxes[:, 3] / 2
                gt_boxes *= 416

                for label in range(num_classes):
                    all_annotations[-1][label] = gt_boxes[annotation_labels == label, :]

        del images, labels
        gc.collect()
        torch.cuda.empty_cache()

    average_precisions = {}
    for label in range(num_classes):
        true_positives = []
        scores = []
        num_annotations = 0

        for idx in range(len(all_annotations)):
            detections = all_detections[idx][label]
            annotations = all_annotations[idx][label]

            num_annotations += annotations.shape[0]
            detected_annotations = []

            for *bbox, score in detections:
                scores.append(score)

                if annotations.shape[0] == 0:
                    true_positives.append(0)
                    continue

                overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                assigned = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned]

                if max_overlap >= 0.5 and assigned not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned)
                else:
                    true_positives.append(0)

        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        tp = np.array(true_positives)
        fp = 1 - tp

        indices = np.argsort(-np.array(scores))
        tp = tp[indices]
        fp = fp[indices]

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        recall = tp / num_annotations
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        average_precisions[label] = compute_average_precision(recall, precision)

    ap_list = list(average_precisions.values())
    mAP = np.mean(ap_list)
    ap_list.append(mAP)

    print(f"Validation mAP: {mAP:.4f}")

    if len(y_true_all) > 0 and len(y_pred_all) > 0:
        print("\nClassification Report:")
        print(classification_report(y_true_all, y_pred_all, zero_division=0))

    return model, mAP

def trainmodel(model, device, optimizer, scheduler, trainloader, valloader, save_dir, max_epochs=30, tensor_type=torch.cuda.FloatTensor, update_every=16, freeze_backbone=False, freeze_epoch=-1):
    best_mAP = 0.0

    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")

        if freeze_backbone and freeze_epoch != -1 and epoch + 1 >= freeze_epoch:
            model = trainepoch(model, device, optimizer, scheduler, trainloader, tensor_type, update_every, freeze_backbone=True)
        else:
            model = trainepoch(model, device, optimizer, scheduler, trainloader, tensor_type, update_every, freeze_backbone=False)

        model, current_mAP = validate(model, device, valloader, tensor_type)

        if current_mAP > best_mAP:
            best_mAP = current_mAP
            print("Saving current best epoch to model...")
            model.save_weights(os.path.join(save_dir, "trained_model.pth"))
