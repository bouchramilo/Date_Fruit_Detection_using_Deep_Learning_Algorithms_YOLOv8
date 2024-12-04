
"""le fichier res.py"""


import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

def load_model(model_path):
    # Load the YOLO model
    model = YOLO(model_path)
    return model

def process_image(model, image_path):
    # Load an image and convert it to RGB
    img = Image.open(image_path).convert('RGB')
    
    # Perform the detection
    results = model(img)

    # Check if detections were made
    if results.xyxy[0].shape[0] > 0:  # Check if there are any detections
        print(f"Detected {results.xyxy[0].shape[0]} objects.")
        draw = ImageDraw.Draw(img)
        for det in results.xyxy[0]:  # results.xyxy[0] is tensor of detections
            x1, y1, x2, y2, conf, cls_id = map(int, det)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            draw.text((x1, y1 - 10), f"{results.names[int(cls_id)]} {conf:.2f}",size,  fill='red')
        # Save the annotated image
        save_path = image_path.replace('.jpg', '_annotated.jpg')
        img.save(save_path)
        print(f"Annotated image saved to {save_path}")
    else:
        print("No objects detected.")

def main():
    model_path = './best.pt'  # Path to the YOLO model weights
    image_path = 'D:/PFE/20220901_175835.jpg'  # Path to the image to process

    # Load the model
    model = load_model(model_path)

    # Process the image
    process_image(model, image_path)

if __name__ == '__main__':
    main()
