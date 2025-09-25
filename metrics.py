import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

from tqdm import tqdm
import json

from FileCrawler import FileCrawler




def compute_precision_recall(results, label_path, penalize_extra_correct=True):
  # Ensure paths are valid
  label_path = Path(label_path)
  
  # Extract predictions
  pred_boxes = results.boxes.xyxy.cpu().numpy()  # Predicted bounding boxes [x1, y1, x2, y2]
  pred_scores = results.boxes.conf.cpu().numpy()  # Confidence scores
  pred_classes = results.boxes.cls.cpu().numpy()  # Predicted classes
  
  # Read ground truth labels (YOLO format: class x_center y_center width height)
  gt_boxes = []
  gt_classes = []
  with open(label_path, 'r') as f:
    for line in f:
      parts = line.strip().split()
      if len(parts) < 5:
        continue
      cls = int(parts[0])
      x_center, y_center, width, height = map(float, parts[1:5])
      # Convert YOLO format to [x1, y1, x2, y2]
      img = results[0].orig_img
      img_h, img_w = img.shape[:2]
      x1 = (x_center - width / 2) * img_w
      y1 = (y_center - height / 2) * img_h
      x2 = (x_center + width / 2) * img_w
      y2 = (y_center + height / 2) * img_h
      gt_boxes.append([x1, y1, x2, y2])
      gt_classes.append(cls)
  
  gt_boxes = np.array(gt_boxes)
  gt_classes = np.array(gt_classes)
  
  # Compute IoU between predicted and ground truth boxes
  def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_g, y1_g, x2_g, y2_g = box2
    
    # Calculate intersection coordinates
    xx1 = max(x1, x1_g)
    yy1 = max(y1, y1_g)
    xx2 = min(x2, x2_g)
    yy2 = min(y2, y2_g)
    
    # Compute areas
    inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_g - x1_g) * (y2_g - y1_g)
    
    # Compute IoU
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou
  
  # Match predictions to ground truth
  iou_threshold = 0.5
  true_positives = 0
  false_positives = 0
  false_negatives = len(gt_boxes)  # Initially assume all GT boxes are missed
  matched_gt_indices = set()  # Track which ground truth boxes are matched
  
  for pred_idx, (pred_box, pred_cls, pred_conf) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
    best_iou = 0
    best_gt_idx = -1
    
    for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
      if pred_cls == gt_cls:  # Match classes
        iou = compute_iou(pred_box, gt_box)
        if iou > best_iou:
          best_iou = iou
          best_gt_idx = gt_idx
    
    if best_iou >= iou_threshold:
      if penalize_extra_correct:
        # Only count as TP if the GT box hasn't been matched yet
        if best_gt_idx not in matched_gt_indices:
          true_positives += 1
          matched_gt_indices.add(best_gt_idx)
          false_negatives -= 1  # This GT box was detected
        else:
          false_positives += 1  # Extra correct prediction
      else:
        # Count as TP regardless of whether GT box was already matched
        true_positives += 1
        if best_gt_idx not in matched_gt_indices:
          matched_gt_indices.add(best_gt_idx)
          false_negatives -= 1  # This GT box was detected
    else:
      false_positives += 1  # Incorrect prediction
  
  # Compute precision and recall
  precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
  recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
  
  return precision, recall



def load_yolo_labels(label_path, img_width, img_height):
  """Load YOLO format labels and convert to absolute coordinates [x_min, y_min, w, h]."""
  gt_boxes = []
  with open(label_path, 'r') as f:
    for line in f:
      class_id, cx, cy, w, h = map(float, line.strip().split())
      # Convert from normalized YOLO format to absolute coordinates
      cx, cy, w, h = cx * img_width, cy * img_height, w * img_width, h * img_height
      x_min = cx - w / 2
      y_min = cy - h / 2
      gt_boxes.append({'bbox': [x_min, y_min, w, h], 'category_id': int(class_id)})
  return gt_boxes

def iou(box1, box2):
  """Calculate IoU between two boxes [x_min, y_min, w, h]."""
  x1, y1, w1, h1 = box1
  x2, y2, w2, h2 = box2
  x1_max, y1_max = x1 + w1, y1 + h1
  x2_max, y2_max = x2 + w2, y2 + h2

  inter_x_min = max(x1, x2)
  inter_y_min = max(y1, y2)
  inter_x_max = min(x1_max, x2_max)
  inter_y_max = min(y1_max, y2_max)

  inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
  union_area = (w1 * h1) + (w2 * h2) - inter_area
  return inter_area / union_area if union_area > 0 else 0

def compute_ap(recall, precision):
  """Compute Average Precision from recall and precision arrays."""
  recall = np.concatenate(([0], recall, [1]))
  precision = np.concatenate(([0], precision, [0]))
  for i in range(len(precision) - 1, 0, -1):
    precision[i - 1] = max(precision[i - 1], precision[i])
  i = np.where(recall[1:] != recall[:-1])[0]
  ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
  return ap

def compute_map(gt_boxes, pred_boxes, iou_thresholds=[0.5], penalize_extra_correct=True):
  """Compute mAP for a single class or multiple classes."""
  aps = []
  classes = set([box['category_id'] for box in gt_boxes] + [box['category_id'] for box in pred_boxes])
  
  for cls in classes:
    cls_gt = [box for box in gt_boxes if box['category_id'] == cls]
    cls_pred = [box for box in pred_boxes if box['category_id'] == cls]
    for iou_th in iou_thresholds:
      # Sort predictions by confidence
      cls_pred = sorted(cls_pred, key=lambda x: x['score'], reverse=True)
      gt_count = len(cls_gt)
      tp = np.zeros(len(cls_pred))
      fp = np.zeros(len(cls_pred))
      matched = set()

      for i, pred in enumerate(cls_pred):
        best_iou = 0
        best_gt_idx = -1
        for j, gt in enumerate(cls_gt):
          if j in matched:
            continue
          iou_score = iou(pred['bbox'], gt['bbox'])
          if iou_score > best_iou:
            best_iou = iou_score
            best_gt_idx = j
        if best_iou >= iou_th and best_gt_idx >= 0:
          if best_gt_idx not in matched:
            tp[i] = 1
            matched.add(best_gt_idx)
          else:
            fp[i] = 1
        else:
          if penalize_extra_correct:
            fp[i] = 1

      # Compute precision and recall
      tp = np.cumsum(tp)
      fp = np.cumsum(fp)
      recall = tp / max(gt_count, 1)
      precision = tp / np.maximum(tp + fp, 1e-9)
      ap = compute_ap(recall, precision)
      aps.append(ap)

  map50 = np.mean([aps[i] for i in range(0, len(aps), len(iou_thresholds))]) if aps else 0
  map50_95 = np.mean(aps) if len(iou_thresholds) > 1 else None
  return map50, map50_95

def run_yolo_predictions(image_path, model):
  # Load image
  img = cv2.imread(image_path)
  img_height, img_width = img.shape[:2]

  # Run predictions
  # results = model(img)[0]  # Get first result (single image)
  results = model.predict(
    img,
    # imgsz=640,
    # persist=True,
    # stream=True,
    # show=True,
    show=False,
    # conf=0.00001,
    # conf=0.5,
    conf=0.1,
    # iou=0.9,
    iou=0.3,
    agnostic_nms=True,
    verbose=False,
    # verbose=True,
    # tracker='custom_botsort.yaml'
  )[0]
  
  
  # Extract predictions
  pred_boxes = []
  for box in results.boxes:
    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
    conf = box.conf.cpu().numpy()
    cls = int(box.cls.cpu().numpy())
    pred_boxes.append({
      'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
      'category_id': cls,
      'score': conf
    })
  
  return results, pred_boxes, img_width, img_height

def main(image_path, label_path, model):
  # Load ground truth labels
  img = cv2.imread(image_path)
  img_width, img_height = img.shape[:2]
  gt_boxes = load_yolo_labels(label_path, img_width, img_height)
  
  # Run YOLO predictions
  results, pred_boxes, _, _ = run_yolo_predictions(image_path, model)
  
  # Compute mAP50 and mAP50:95
  iou_thresholds = [0.5]  # For mAP50 only
  # For mAP50:95, use: iou_thresholds = np.arange(0.5, 1.0, 0.05)
  
  precision, recall = compute_precision_recall(results, label_path, penalize_extra_correct=True)
  precisionNoPenalty, recallNoPenalty = compute_precision_recall(results, label_path, penalize_extra_correct=False)
  
  map50, _ = compute_map(gt_boxes, pred_boxes, iou_thresholds, penalize_extra_correct=True)
  map50NoPenalty, _ = compute_map(gt_boxes, pred_boxes, iou_thresholds, penalize_extra_correct=False)
  
  iou_thresholds = np.arange(0.5, 1.0, 0.05)  # For mAP50:95
  _, map50_95 = compute_map(gt_boxes, pred_boxes, iou_thresholds, penalize_extra_correct=True)
  _, map50_95NoPenalty = compute_map(gt_boxes, pred_boxes, iou_thresholds, penalize_extra_correct=False)

  return map50, map50NoPenalty, map50_95, map50_95NoPenalty, precision, recall, precisionNoPenalty, recallNoPenalty


if __name__ == "__main__":
  projectName = 'ApplesM5'

  #point the script at the corresponding trained best.pt s
  
  modelPaths = [
    ('real',                      rf'./ai\runs\detect\{projectName}_12n-detect-100_real_0\weights\best.pt'),
    ('synetic-train+real-val',    rf'./ai\runs\detect\{projectName}_12n-detect-100_synetic-train+real-val_0\weights\best.pt'),
    ('synetic+bg-train+real-val', rf'./ai\runs\detect\{projectName}_12n-detect-100_synetic+bg-train+real-val_0\weights\best.pt'),
    ('synetic+real',              rf'./ai\runs\detect\{projectName}_12n-detect-100_synetic+real_0\weights\best.pt'),
  ]
  
  pathValsDataset = fr'W:\synetic\Marketing\datasets\{projectName}\real\yolo\images\vals'
  directoryNameContainsFilterSet = set([])
  nameContainsFilterSet = set([])
  extensionFilterSet = set(['.png', '.jpg'])
  fileCrawlerVals = FileCrawler(pathValsDataset, directoryNameContainsFilterSet, nameContainsFilterSet, extensionFilterSet)

  mAPs = []
  for modelName, modelPath in tqdm(modelPaths):
    """Run YOLOv8 predictions on an image and return bounding boxes."""
    # Load YOLOv8 model
    model = YOLO(modelPath, task='detect')
    
    mAP50s = []
    mAP50_95s = []
    mAP50NoPenaltys = []
    mAP50_95NoPenaltys = []
        
    precisions = []
    recalls = []
    precisionsNoPenaltys = []
    recallsNoPenaltys = []

    for yoloFile in tqdm(fileCrawlerVals._filesArr):
      pathYoloLabelFile = yoloFile._path.replace('/yolo/images/', '/yolo/labels/').replace(yoloFile._extension, '.txt')

      mAP50, mAP50NoPenalty, mAP50_95, mAP50_95NoPenalty, precision, recall, precisionNoPenalty, recallNoPenalty = main(yoloFile._path, pathYoloLabelFile, model)
      if mAP50 is not None:
        mAP50s.append(mAP50)
      if mAP50NoPenalty is not None:
        mAP50NoPenaltys.append(mAP50NoPenalty)

      if mAP50_95 is not None:
        mAP50_95s.append(mAP50_95)
      if mAP50_95NoPenalty is not None:
        mAP50_95NoPenaltys.append(mAP50_95NoPenalty)

      precisions.append(precision)
      recalls.append(recall)
      precisionsNoPenaltys.append(precisionNoPenalty)
      recallsNoPenaltys.append(recallNoPenalty)


    mAP50s = np.array(mAP50s)
    mAP50_95s = np.array(mAP50_95s)
    mAP50NoPenaltys = np.array(mAP50NoPenaltys)
    mAP50_95NoPenaltys = np.array(mAP50_95NoPenaltys)
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    precisionsNoPenaltys = np.array(precisionsNoPenaltys)
    recallsNoPenaltys = np.array(recallsNoPenaltys)

    mAPs.append(
      (
        modelName,
        (
          f'mAP50: {np.mean(mAP50s):.4f}', f'mAP50-95: {np.mean(mAP50_95s):.4f}', f'mAP50-np: {np.mean(mAP50NoPenaltys):.4f}', f'mAP50-95-np: {np.mean(mAP50_95NoPenaltys):.4f}',
          f'precision: {np.mean(precisions):.4f}', f'recall: {np.mean(recalls):.4f}', f'precision-np: {np.mean(precisionsNoPenaltys):.4f}', f'recall-np: {np.mean(recallsNoPenaltys):.4f}',
        )
      )
    )


  jsonString = json.dumps(mAPs, sort_keys=True, indent=2)

  print(f'\n{projectName}\n', jsonString)
