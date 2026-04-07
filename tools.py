import cv2
from PIL import Image

def get_frames(file: str):
    cap = cv2.VideoCapture(file)
    pil_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(frame_rgb)
        pil_frames.append(pil_image)
    
    cap.release()
    return pil_frames