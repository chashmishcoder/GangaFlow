import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('yolov8s.pt')  # Update with the correct path to your model file

# Path to the test images and labels
test_images_path = 'F:/GangaFlow/dataset/test/images'  # Update with your test image path
test_labels_path = 'F:/GangaFlow/dataset/test/labels'  # Update with your test label path

# Function to evaluate the model
def evaluate_model(test_images_path, test_labels_path):
    # Get sorted lists of images and labels
    test_images = sorted([f for f in os.listdir(test_images_path) if f.endswith('.jpg')])
    test_labels = sorted([f for f in os.listdir(test_labels_path) if f.endswith('.txt')])

    true_labels = []  # Ground truth labels
    predicted_labels = []  # Predicted labels

    for image_name, label_name in zip(test_images, test_labels):
        image_path = os.path.join(test_images_path, image_name)
        label_path = os.path.join(test_labels_path, label_name)

        # Read the ground truth label (assuming the label is 'polluted' or 'non-polluted')
        with open(label_path, 'r') as f:
            ground_truth = f.read().strip()

        # Debugging: Print the ground truth for each image
        print(f"Ground truth for {image_name}: {ground_truth}")
        
        # Only append valid labels
        if ground_truth in ["polluted", "non-polluted"]:
            true_labels.append(ground_truth)
        else:
            print(f"Warning: Invalid label found in {label_name}, skipping this image.")

        # Perform inference on the image
        image = cv2.imread(image_path)
        results = model(image)

        # Get the result for the first image
        result = results[0]

        # Accessing the boxes and labels
        pred_boxes = result.boxes.xywh.cpu().numpy()  # Get the predicted bounding boxes in xywh format

        # Extract predicted class IDs and labels
        pred_class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Get the class IDs for the predicted boxes
        pred_labels = [result.names[class_id] for class_id in pred_class_ids]  # Map class IDs to label names

        # Debugging: Print the predicted labels for each image
        print(f"Predicted labels for {image_name}: {pred_labels}")

        # Check if a "polluted" label is detected and assign as predicted label
        if "polluted" in pred_labels:
            predicted_labels.append("polluted")
        else:
            predicted_labels.append("non-polluted")

   
    print(f"True labels: {true_labels}")
    print(f"Predicted labels: {predicted_labels}")


    if not true_labels:
        print("No valid true labels found. Exiting evaluation.")
        return

    if not predicted_labels:
        print("No valid predicted labels found. Exiting evaluation.")
        return

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")


    cm = confusion_matrix(true_labels, predicted_labels, labels=["polluted", "non-polluted"])
    print("Confusion Matrix:")
    print(cm)


    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(len(cm)))
    ax.set_yticks(np.arange(len(cm)))
    ax.set_xticklabels(["polluted", "non-polluted"])
    ax.set_yticklabels(["polluted", "non-polluted"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.savefig('confusion_matrix.png')  
    plt.show()

evaluate_model(test_images_path, test_labels_path)
