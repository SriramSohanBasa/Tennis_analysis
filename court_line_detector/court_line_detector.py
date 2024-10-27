import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(weights=None)  # Initialize a pre-trained ResNet50 model
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)  # Modify the final layer for 14 keypoints (x, y coordinates)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))  # Load the trained weights from the given path
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert input to PIL Image
            transforms.Resize((224, 224)),  # Resize image to 224x224 pixels
            transforms.ToTensor(),  # Convert PIL Image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image tensor
        ])

    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB color space
        image_tensor = self.transform(image_rgb).unsqueeze(0)  # Apply transformations and add batch dimension
        with torch.no_grad():
            outputs = self.model(image_tensor)  # Forward pass through the model
        keypoints = outputs.squeeze().cpu().numpy()  # Extract keypoints from model output
        original_h, original_w = image.shape[:2]  # Get original image dimensions
        keypoints[::2] *= original_w / 224.0  # Scale x-coordinates back to original image size
        keypoints[1::2] *= original_h / 224.0  # Scale y-coordinates back to original image size

        return keypoints

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])  # Get x-coordinate of keypoint
            y = int(keypoints[i+1])  # Get y-coordinate of keypoint
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Add keypoint number label
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Draw circle at keypoint location
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)  # Draw keypoints on each frame
            output_video_frames.append(frame)  # Add annotated frame to output list
        return output_video_frames
