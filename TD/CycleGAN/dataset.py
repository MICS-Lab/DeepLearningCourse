import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class GlassesNoGlassesDataset(Dataset):
    def __init__(self, root_glasses, root_no_glasses, transform=None):
        self.root_glasses = root_glasses
        self.root_no_glasses = root_no_glasses
        self.transform = transform

        self.glasses_images = os.listdir(root_glasses)
        self.no_glasses_images = os.listdir(root_no_glasses)


    def __len__(self):
        return max(len(self.glasses_images), len(self.no_glasses_images))

    def __getitem__(self, index):
        image_glass_path = None
        image_glass = None
        if index > len(self.glasses_images):
            image_glass_path = random.choice(self.glasses_images)
        else:
            image_glass_path = self.glasses_images[index]
        image_glass = np.array(
            Image.open(os.path.join(self.root_glasses, image_glass_path)).convert("RGB")
        )

        image_no_glass_path = None
        image_no_glass = None
        if index > len(self.no_glasses_images):
            image_no_glass_path = random.choice(self.no_glasses_images)
        else:
            image_no_glass_path = self.no_glasses_images[index]
        image_no_glass = np.array(
            Image.open(os.path.join(self.root_no_glasses, image_no_glass_path)).convert("RGB")
        )

        if self.transform:
            image_glass = self.transform(image_glass)
            image_no_glass = self.transform(image_no_glass)

        return image_glass, image_no_glass


if __name__ == "__main__":
    dataset = GlassesNoGlassesDataset(
        "data/train/glasses",
        "data/train/no_glasses",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(0.5, 0.5)
        ])
    )
    print(dataset[0])
