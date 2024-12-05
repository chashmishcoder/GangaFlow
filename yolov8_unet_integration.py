import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segmentation_models_pytorch import Unet


yolo_model = YOLO("yolov8n.pt")  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet_model = Unet(encoder_name="resnet34", encoder_weights=None, classes=1, activation=None)
unet_model.load_state_dict(torch.load("unet_ganga_water_quality.pth", map_location=device))
unet_model.to(device)
unet_model.eval()

def process_yolo_predictions(image, results):
    cropped_regions = []
    probabilities = []
    boxes = []
    for result in results:  
        if result.boxes is not None:  
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                confidence = float(box.conf)  
                cropped_region = image[y1:y2, x1:x2]  
                cropped_regions.append(cropped_region)
                probabilities.append(confidence)
                boxes.append((x1, y1, x2, y2))
    return cropped_regions, probabilities, boxes


test_image_path = "F:/GangaFlow/dataset/test/images/test1.jpg"
image = cv2.imread(test_image_path)

if image is None:
    raise ValueError(f"Could not read image from path: {test_image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

yolo_results = yolo_model.predict(image_rgb, device=device, stream=True)

cropped_regions, probabilities, boxes = process_yolo_predictions(image_rgb, yolo_results)

if cropped_regions:
    for i, (region, prob, box) in enumerate(zip(cropped_regions, probabilities, boxes)):
        resized_region = cv2.resize(region, (256, 256))  
        input_tensor = torch.from_numpy(resized_region).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        input_tensor = input_tensor.to(device)

        # U-Net Inference
        with torch.no_grad():
            mask = unet_model(input_tensor)
            mask = mask.sigmoid().squeeze().cpu().numpy()  

        mask_resized = cv2.resize(mask, (region.shape[1], region.shape[0]))

        
        pollution_threshold = 0.5  
        is_polluted = mask_resized.mean() > pollution_threshold
        label = "Polluted" if is_polluted else "Non-Polluted"

        
        overlay = (region * 0.5 + np.expand_dims(mask_resized, axis=-1) * 255 * 0.5).astype(np.uint8)

        
        x1, y1, x2, y2 = box
        color = (0, 0, 255) if is_polluted else (0, 255, 0)  # Red for polluted, green for non-polluted
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  
        probability_text = f"{prob * 100:.2f}% {label}"
        cv2.putText(
            image, probability_text, (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )

        
        overlay_path = f"output_region_{i+1}.png"
        cv2.imwrite(overlay_path, overlay)
        print(f"Saved overlay for Region {i+1} at {overlay_path}")


    output_image_path = "output_with_bounding_boxes.png"
    cv2.imwrite(output_image_path, image)
    print(f"Saved image with bounding boxes and labels at {output_image_path}")
else:
    print("No regions to process. Exiting.")
