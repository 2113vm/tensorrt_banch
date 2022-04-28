import torch
import cv2

from albumentations import Resize, Compose
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import Normalize
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class FlowerDataset(Dataset):
    def __init__(self, image_path: str):
        self.image_pathes = sorted(list(Path(image_path).glob("*.jpg")))
        self.transforms = Compose([
            Resize(224, 224, interpolation=cv2.INTER_NEAREST),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __getitem__(self, idx):
        image_path = self.image_pathes[idx]
        image = cv2.imread(str(image_path))[:, :, ::-1]
        image = self.transforms(image=image)["image"]
        return image

    def __len__(self):
        return len(self.image_pathes)


if __name__ == "__main__":

    IMAGE_DIR = "./data"

    dataset = FlowerDataset(IMAGE_DIR)
    print(dataset[0].shape)
