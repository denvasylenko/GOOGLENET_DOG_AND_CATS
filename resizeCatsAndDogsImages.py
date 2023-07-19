import os
from PIL import Image


class ResizedCatsAndDogsDataset:
    def __init__(self, root_dir, resized_dir, resize_shape):
        self.root_dir = root_dir
        self.resized_dir = resized_dir
        self.resize_shape = resize_shape

        if not os.path.exists(self.resized_dir):
            self.create_resized_images()

    def create_resized_images(self):
        os.makedirs(self.resized_dir, exist_ok=True)

        files = os.listdir(self.root_dir)
        for file_name in files:
            if file_name.endswith('.jpg'):
                image_path = os.path.join(self.root_dir, file_name)
                resized_image_path = os.path.join(self.resized_dir, file_name)

                with Image.open(image_path) as image:
                    resized_image = image.resize(self.resize_shape)
                    resized_image.save(resized_image_path)

        print(f"Resized images saved in '{self.resized_dir}'.")
