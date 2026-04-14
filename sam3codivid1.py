import torch
import numpy as np
import cv2
import os
import argparse
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

parser = argparse.ArgumentParser(description="Processar múltiples vídeos amb SAM3")

parser.add_argument("--input", type=str, required=True, help="Carpeta d'entrada amb vídeos")
parser.add_argument("--prompt", type=str, default="bike lane", help="Text prompt")
parser.add_argument("--threshold", type=float, default=0.5, help="Llindar de puntuació")

args = parser.parse_args()

input_folder = args.input
prompt = args.prompt
threshold = args.threshold

clean_prompt = prompt.replace(" ", "_")
output_folder = f"results_video_{clean_prompt}"
os.makedirs(output_folder, exist_ok=True)

print("Carregant model SAM3...")
model = build_sam3_image_model()
processor = Sam3Processor(model)
print("Model carregat")

video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
video_files = [
    f for f in os.listdir(input_folder)
    if f.lower().endswith(video_extensions)
]

total_videos = len(video_files)
print(f"S'han trobat {total_videos} vídeos per processar.")

for v_idx, video_filename in enumerate(video_files, start=1):
    input_path = os.path.join(input_folder, video_filename)
    output_path = os.path.join(output_folder, f"out_{video_filename}")
                
    print(f"\n[{v_idx}/{total_videos}] Processant: {video_filename}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"No s'ha pogut obrir el vídeo: {video_filename}")
        continue

    # Propietats del vídeo                                                    
    fps = cap.get(cv2.CAP_PROP_FPS)                                                        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                                                            
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configurar l'escriptor de vídeo                                                                        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                                                            
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0: # Feedback cada 10 frames per no saturar la consola
            print(f"  -> Frame {frame_count}/{total_frames}", end='\r')
            
        # Preparar imatge per al model
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Inferència
        inference_state = processor.set_image(pil_image)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

        # Filtrar resultats
        filtered = [
            (m, b, s) for m, b, s in zip(masks, boxes, scores)
            if float(s) >= threshold
        ]

        result = frame.copy()
        if len(filtered) > 0:
            f_masks, f_boxes, f_scores = zip(*filtered)

            # Dibuixar màscares
            for mask in f_masks:
                mask_np = mask.cpu().numpy().astype(np.uint8)
                colored_mask = np.zeros_like(result)
                colored_mask[:, :, 2] = mask_np * 255 # Vermell
                result = cv2.addWeighted(result, 1.0, colored_mask, 0.5, 0)

            # Dibuixar caixes
            for box, score in zip(f_boxes, f_scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result, f"{float(score):.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out.write(result)
    #Tancar fitxers del vídeo actual    
    cap.release()
    out.release()
    print(f"\n  Done: {output_path}")

print("\n--- Tots els vídeos han estat processats ---")
