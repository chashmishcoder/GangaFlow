import os
import numpy as np
import matplotlib.pyplot as plt

test_images_path = "F:/GangaFlow/dataset/test/images"  
pred_dir = "F:/GangaFlow/runs/detect/predict"  
ground_truth_dir = "F:/GangaFlow/dataset/test/labels"  

# IoU calculation function
def calculate_iou(pred_bbox, gt_bbox):
    x1, y1, w1, h1 = pred_bbox
    x2, y2, w2, h2 = gt_bbox
    x1_min, y1_min, x1_max, y1_max = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min, x2_max, y2_max = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2

   
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection = inter_width * inter_height

    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    # IoU calculation
    iou = intersection / union if union != 0 else 0
    return iou

def read_label_file(file_path):
    bboxes = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            bboxes.append([x_center, y_center, width, height])
    return bboxes

ious = []
for img_name in os.listdir(test_images_path):
    if img_name.endswith(".jpg") or img_name.endswith(".png"):  # assuming images are .jpg or .png
        gt_file = os.path.join(ground_truth_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        pred_file = os.path.join(pred_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        if os.path.exists(gt_file) and os.path.exists(pred_file):
            # Read ground truth and prediction labels
            gt_bboxes = read_label_file(gt_file)
            pred_bboxes = read_label_file(pred_file)

            for gt_bbox in gt_bboxes:
                max_iou = 0
                for pred_bbox in pred_bboxes:
                    iou = calculate_iou(pred_bbox, gt_bbox)
                    max_iou = max(max_iou, iou)

                ious.append(max_iou)

plt.figure(figsize=(10, 6))
plt.hist(ious, bins=20, edgecolor='black', alpha=0.7)
plt.title("IoU Distribution")
plt.xlabel("IoU")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
plt.savefig("iou_curve.png")
