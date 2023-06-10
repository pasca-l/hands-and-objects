import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class EgoHOSObjnessClsDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        phase='train',
        transform=None,
        with_info=False,
    ):
        super().__init__()

        self.frame_dir = os.path.join(dataset_dir, f"egohos/{phase}/image/")
        self.label_dir = os.path.join(dataset_dir, f"egohos/{phase}/label/")
        self.df_files = pd.DataFrame(
            [f[:-4] for f in os.listdir(self.frame_dir)],
            columns=["file_name"]
        )
        self.transform = transform
        self.with_info = with_info

    def __len__(self):
        return len(self.df_files)

    def __getitem__(self, index):
        file_name = self.df_files.iloc[index]["file_name"]
        frame = self._get_frame(file_name)
        label = self._get_label(file_name)

        frame, label = self.transform(frame, label)

        if self.with_info:
            return frame, label, file_name

        return frame, label

    def _get_frame(self, file_name):
        frame_path = os.path.join(self.frame_dir, f"{file_name}.jpg")
        try:
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
        except:
            print(f"Image does not exist : {frame_path}")
            return

        return np.array(frame)

    def _get_label(self, file_name):
        label_path = os.path.join(self.label_dir, f"{file_name}.png")
        try:
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, (224, 224))
        except:
            print(f"Label does not exist : {label_path}")
            return

        mask = self._create_mask(label)
        return mask

    def _create_mask(self, label):
        masks = []

        bg = np.where(
            (label == 0),
            1.0, 0.0
        )
        objects = np.where(
            (label == 3) | (label == 4) | (label == 5) | (label == 6) |
            (label == 7) | (label == 8),
            1.0, 0.0
        )
        hands = np.where(
            (label == 1) | (label == 2),
            1.0, 0.0
        )

        # masks.append(bg)
        masks.append(objects)
        masks.append(hands)

        return np.array(masks).transpose(1,2,0)
