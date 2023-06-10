import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class PascalVOC2012Dataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        phase='train',
        transform=None,
        with_info=False,
    ):
        super().__init__()

        self.image_dir = os.path.join(dataset_dir, "voc2012/JPEGImages")
        self.label_dir = os.path.join(dataset_dir, "voc2012/SegmentationClass")
        self.df_files = pd.DataFrame(
            [f[:-4] for f in os.listdir(self.label_dir)],
            columns=["file_name"],
        )
        self.transform = transform
        self.with_info = with_info

    def __len__(self):
        return len(self.df_files)

    def __getitem__(self, index):
        file_name = self.df_files.iloc[index]["file_name"]
        image = self._get_image(file_name)
        label = self._get_label(file_name)

        image, label = self.transform(image, label)

        if self.with_info:
            return image, label, file_name

        return image, label

    def _get_image(self, file_name):
        image_path = os.path.join(self.image_dir, f"{file_name}.jpg")
        try:
            image = cv2.imread(image_path)
        except:
            print(f"Image does not exist : {image_path}")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        return np.array(image)

    def _get_label(self, file_name):
        label_path = os.path.join(self.label_dir, f"{file_name}.png")
        try:
            label = Image.open(label_path)
        except:
            print(f"Label does not exist : {label_path}")
            return

        label.convert('P')
        label = np.asarray(label)
        label = np.where(label == 255, 0, label)
        label = cv2.resize(label, (224, 224))

        bg = np.where(label == 0, 1, 0)
        obj = np.where(label != 0, 1, 0)

        mask = np.stack([bg, obj], axis=-1)

        return mask
