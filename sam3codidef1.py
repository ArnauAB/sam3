####################################
# Imports
####################################
import torch
import numpy as np
import cv2
import os
import argparse
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


####################################
# Arguments
####################################
parser = argparse.ArgumentParser(description="Process images with SAM3")

parser.add_argument("--input", type=str, required=True, help="Input folder")
parser.add_argument("--prompt", type=str, default="bike lane", help="Text prompt")
parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold")

args = parser.parse_args()

input_folder = args.input
prompt = args.prompt
threshold = args.threshold

####################################
# Create output folder
####################################
clean_prompt = prompt.replace(" ", "_")
output_folder = f"results_{clean_prompt}"
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
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    
    filtered = [
        (m, b, s) for m, b, s in zip(masks, boxes, scores)
        if float(s) >= threshold
    ]
    if len(filtered) > 0:
        masks, boxes, scores = zip(*filtered)
    else:
        masks, boxes, scores = [], [], []

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result = img.copy()
    
    for mask in masks:
        mask = mask.cpu().numpy().astype(np.uint8)
        colored_mask = np.zeros_like(result)
        colored_mask[:, :, 2] = mask * 255
        result = cv2.addWeighted(result, 1.0, colored_mask, 0.5, 0)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(result,f"{float(score):.2f}",(x1, y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, result)

print("Finished processing all images")
