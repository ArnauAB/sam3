####################################
# Imports
####################################
import torch
import numpy as np
import cv2
import os
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


####################################
# Paths
####################################
input_folder = "/home/usuaris/imatge/arnau.aban/Dataset/"
output_folder = "results_prompt_especial"

# Create results folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


####################################
# Load model
####################################
print("Loading SAM3 model...")

model = build_sam3_image_model()
processor = Sam3Processor(model)

print("Model loaded")


####################################
# Get list of images
####################################
image_files = [
    f for f in os.listdir(input_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

total_images = len(image_files)

print(f"Found {total_images} images")


####################################
# Process images
####################################
for i, filename in enumerate(image_files, start=1):
    
    print(f"Processed images: {i} of {total_images}")
    image_path = os.path.join(input_folder, filename)

    image = Image.open(image_path)
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt="bike lane signal")                                     

    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result = img.copy()

    for mask in masks:
        mask = mask.cpu().numpy().astype(np.uint8)
        colored_mask = np.zeros_like(result)
        colored_mask[:, :, 2] = mask * 255  # red mask
        result = cv2.addWeighted(result, 1.0, colored_mask, 0.5, 0)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(result, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, result)

print("Finished processing all images")
