import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision

from predictImage import ImageProcessor

num_classes = 2


class CatAndDogClassifier:
    def __init__(self):
        # Model
        self.model = torchvision.models.googlenet(weights="DEFAULT")
        self.model.fc = nn.Linear(in_features=1024, out_features=num_classes)

        parameters = torch.load("my_model_parameters.pth.tar")
        self.model.load_state_dict(parameters["state_dict"])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        self.model.eval()

    def predict(self, image_path):
        # Create an instance of ImageProcessor with your model
        image_processor = ImageProcessor(self.model)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        output_tensor = image_processor.process_image(image_path, transform)
        _, predictions = output_tensor.max(1)
        animal = predictions.numpy()[0]
        print('This is ' + ('dog' if animal == 1 else 'cat'))



