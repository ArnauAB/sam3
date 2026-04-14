import torch
import numpy as np
import cv2

#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("/home/usuaris/imatge/arnau.aban/sam3/imgbcn02.jpg")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="bike_lane")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

import numpy as np
import cv2

# Convert PIL image to numpy
img = np.array(image)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

result = img.copy()

# Draw masks
for mask in masks:
    mask = mask.cpu().numpy().astype(np.uint8)

    # Create colored mask
    colored_mask = np.zeros_like(result)
    colored_mask[:, :, 2] = mask * 255  # red mask

    # Blend mask with image
    result = cv2.addWeighted(result, 1.0, colored_mask, 0.5, 0)

# Draw bounding boxes
for box, score in zip(boxes, scores):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(result, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(result, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# Save result
cv2.imwrite("results.jpg", result)

print("Result saved as results.jpg")
