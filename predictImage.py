import torch
import torchvision.transforms as transforms
from PIL import Image


class ImageProcessor:
    def __init__(self, model):
        self.model = model

    def process_image(self, image_path, transform):
        # Load the image
        image = Image.open(image_path)

        # Apply the transformation to the image
        input_data = transform(image).unsqueeze(0)  # Add batch dimension

        # Pass the input through the model
        with torch.no_grad():
            self.model.eval()  # Set the model to evaluation mode
            output = self.model(input_data)

        return output

