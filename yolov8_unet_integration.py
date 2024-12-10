import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segmentation_models_pytorch import Unet

# Load YOLOv8 Model
yolo_model = YOLO("yolov8n.pt")  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load U-Net Model
unet_model = Unet(encoder_name="resnet34", encoder_weights=None, classes=1, activation=None)
unet_model.load_state_dict(torch.load("unet_ganga_water_quality.pth", map_location=device))
unet_model.to(device)
unet_model.eval()

# Process YOLO Predictions
def process_yolo_predictions(image, results):
    cropped_regions = []
    for result in results:  # Iterate through results generator
        if result.boxes is not None:  # Check if any boxes were detected
            for box in result.boxes.xyxy:  # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box)
                cropped_region = image[y1:y2, x1:x2]  # Crop the detected region
                cropped_regions.append(cropped_region)
    return cropped_regions

# Load and Process Image
test_image_path = "F:/GangaFlow/dataset/test/images/high_resolution_image16_png.rf.1ac69be3258e2e6c0feed42cac092d52.jpg"
image = cv2.imread(test_image_path)

if image is None:
    raise ValueError(f"Could not read image from path: {test_image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# YOLO Inference
yolo_results = yolo_model.predict(image_rgb, device=device, stream=True)

# Process YOLO Results
cropped_regions = process_yolo_predictions(image_rgb, yolo_results)

if cropped_regions:
    for i, region in enumerate(cropped_regions):
        resized_region = cv2.resize(region, (256, 256))  # Resize to match U-Net input size
        input_tensor = torch.from_numpy(resized_region).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        input_tensor = input_tensor.to(device)

        # U-Net Inference
        with torch.no_grad():
            mask = unet_model(input_tensor)
            mask = mask.sigmoid().squeeze().cpu().numpy()  # Apply sigmoid activation

        # Resize mask back to the original cropped region size
        mask_resized = cv2.resize(mask, (region.shape[1], region.shape[0]))

        # Overlay mask on the cropped region
        overlay = (region * 0.5 + np.expand_dims(mask_resized, axis=-1) * 255 * 0.5).astype(np.uint8)

        # Save the overlay to disk
        output_path = f"output_region_{i+1}.png"
        cv2.imwrite(output_path, overlay)
        print(f"Saved overlay for Region {i+1} at {output_path}")
else:
    print("No regions to process. Exiting.")
