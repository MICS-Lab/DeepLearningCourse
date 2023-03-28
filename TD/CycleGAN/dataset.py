import numpy as np
import os
import random
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
        image_glass = None
        if index > len(self.glasses_images):
            image_glass = random.choice(self.glasses_images)
        else:
            image_glass = self.glasses_images[index]

        image_no_glass = None
        if index > len(self.no_glasses_images):
            image_no_glass = random.choice(self.no_glasses_images)
        else:
            image_no_glass = self.no_glasses_images[index]

        return image_glass, image_no_glass


if __name__ == "__main__":
    dataset = GlassesNoGlassesDataset("data/train/glasses", "data/train/no_glasses")
    print(dataset[0])
