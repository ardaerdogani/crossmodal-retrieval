import csv
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import BertTokenizer

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MAX_TOKEN_LENGTH = 64
BATCH_SIZE = 64


class Flickr8kDataset(Dataset):
    def __init__(self, captions_file: str, images_dir: str):
        self.images_dir = images_dir

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.samples = []  # list of (image_filename, caption)
        with open(captions_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                image_filename = row[0]
                caption = row[1]
                self.samples.append((image_filename, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_filename, caption = self.samples[idx]
        image_path = os.path.join(self.images_dir, image_filename)

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        return image, input_ids, attention_mask


def get_dataloader(captions_file: str, images_dir: str, batch_size: int = BATCH_SIZE, shuffle: bool = True):
    dataset = Flickr8kDataset(captions_file, images_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
