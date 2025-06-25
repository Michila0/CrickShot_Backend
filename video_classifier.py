import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import time
import os
from pathlib import Path


# Load your trained model
class ShotClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.class_names = ['drive', 'legglance-flick', 'pullshot', 'sweep']

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _create_model(self):
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 4)
        )
        return model.to(self.device)

    def predict(self, image):
        # Convert OpenCV BGR image to RGB and then to PIL Image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        # Apply transformations
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, preds = torch.max(outputs, 1) #shot classification

        return self.class_names[preds.item()]


def process_video(video_path, model_path, output_path=None):
    # Initialize classifier
    classifier = ShotClassifier(model_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare video writer if output path is specified
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read() # Extract frames
        if not ret:
            break

        # Predict shot type
        shot_type = classifier.predict(frame)

        # Calculate processing time
        current_time = time.time()
        processing_time = current_time - prev_time
        prev_time = current_time

        # Display information on frame
        cv2.putText(frame, f"Shot: {shot_type}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {1 / processing_time:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Cricket Shot Classification', frame)

        # Write frame if output path is specified
        if output_path:
            out.write(frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def classifyShot(input_path: str, output_path: str):
    model_path = "./best_model.pth"  # Your trained model

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        process_video(input_path, model_path, output_path)
        return True
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return False
