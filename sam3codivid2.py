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
parser = argparse.ArgumentParser(description="Processar vídeo amb SAM3")

parser.add_argument("--input", type=str, required=True, help="Ruta del vídeo d'entrada")
parser.add_argument("--prompt", type=str, default="bike lane", help="Text prompt")
parser.add_argument("--threshold", type=float, default=0.5, help="Llindar de puntuació")

args = parser.parse_args()

input_video_path = args.input
prompt = args.prompt
threshold = args.threshold

####################################
# Configurar entrada i sortida de vídeo
####################################
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error en obrir el vídeo: {input_video_path}")
    exit()

# Obtenir propietats del vídeo original
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Configurar el fitxer de sortida (.mp4)
output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

####################################
# Load model
####################################
print("Carregant model SAM3...")
model = build_sam3_image_model()
processor = Sam3Processor(model)
print("Model carregat")

####################################
# Processar frames
####################################
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    print(f"Processant frame: {frame_idx} de {total_frames}", end='\r')

    # Convertir de BGR (OpenCV) a RGB (PIL) per al processador
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    # Inferència
    inference_state = processor.set_image(pil_image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    
    # Filtrar per threshold
    filtered = [
        (m, b, s) for m, b, s in zip(masks, boxes, scores)
        if float(s) >= threshold
    ]
    
    if len(filtered) > 0:
        masks, boxes, scores = zip(*filtered)
    else:
        masks, boxes, scores = [], [], []

    # Dibuixar resultats sobre el frame original
    result = frame.copy()
    
    for mask in masks:
        # Convertir màscara a format OpenCV
        mask_np = mask.cpu().numpy().astype(np.uint8)
        # Crear una capa vermella per a la màscara
        colored_mask = np.zeros_like(result)
        colored_mask[:, :, 2] = mask_np * 255 # Canal vermell en BGR
        result = cv2.addWeighted(result, 1.0, colored_mask, 0.5, 0)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, f"{float(score):.2f}", (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    # Escriure el frame processat al fitxer de sortida
    out.write(result)

# Alliberar recursos
cap.release()
out.release()
print(f"\nProcés finalitzat. Vídeo desat a: {output_path}")
