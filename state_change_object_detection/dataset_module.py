import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl


class StateChgObjDataModule(pl.LightningDataModule):
    """
    Fetches total of 3 frames per state change: pre, pnr, and post frames. Labels are given by a object bounding box information.

    Returns:
        [
            frames: [batch, frame_num, height, width, channel],
            labels: {
                "mask": ['pre_frame': 0, 'pnr_frame': 1, 'post_frame': 2],
                "labels": [
                    "left_hand": 0, "right_hand": 1,
                    "object_of_change": 2, "tool": 3
                ],
                "boxes": [x1, x2, y1, y2] (corners) or
                         [cx, cy, w, h] (COCO)
            },
            info: dict of additional information (optional)
        ]
    """

    def __init__(self, data_dir, json_dict, model_name, batch_size, label_mode):
        super().__init__()
        self.data_dir = data_dir
        self.json_dict = json_dict
        self.batch_size = batch_size
        self.label_mode = label_mode

        preprocessor = StateChgObjDataPreprocessor(model_name)
        self.transform = preprocessor()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = StateChgObjDataset(
                data_dir=self.data_dir,
                flatten_json=self.json_dict['train'],
                transform=self.transform,
                label_mode=self.label_mode
            )
            self.val_data = StateChgObjDataset(
                data_dir=self.data_dir,
                flatten_json=self.json_dict['val'],
                transform=self.transform,
                label_mode=self.label_mode
            )

        if stage == "test":
            self.test_data = None

        if stage == "predict":
            self.predict_data = None

    def collate_fn(self, batch):
        frames, labels = zip(*batch)
        return torch.stack(frames, dim=0), labels

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.collate_fn
        )


class StateChgObjDataset(Dataset):
    def __init__(self, data_dir, flatten_json, transform, label_mode):
        self.frame_dir = data_dir
        self.flatten_json = flatten_json
        self.transform = transform
        self.label_mode = label_mode

    def __len__(self):
        return len(self.flatten_json)

    def __getitem__(self, index):
        info = self.flatten_json[index]
        labels = self._get_labels(info)
        frames = self._get_frames(info)
        frames = self.transform(frames)

        return frames, labels

    def _get_frames(self, info):
        frames = []
        for frame_type in ['pre_frame', 'pnr_frame', 'post_frame']:
            frame_path = f"{self.frame_dir}{info['clip_uid']}/" +\
                         f"{info[f'{frame_type}_num_clip']}.png"
            try:
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                image = np.expand_dims(frame, axis=0).astype(np.float32)
            except:
                print(f"Image does not exist : {frame_path}")
                return

            frames.append(image)

        return np.concatenate(frames)

    def _get_labels(self, info):
        objs = {
            "frame_type": [],
            "labels": [],
            "boxes": []
        }
        obj_cls_dict = {
            "left_hand": 0,
            "right_hand": 1,
            "object_of_change": 2,
            "tool": 3,
        }

        for i, frame in enumerate(['pre_frame', 'pnr_frame', 'post_frame']):
            for objects in info[f"{frame}_objects"]:
                objs["frame_type"].append(i)

                bbox_cls = obj_cls_dict[objects["object_type"]]
                objs["labels"].append(bbox_cls)

                if self.label_mode == 'corners':
                    bbox_x1 = objects["bbox_x"]
                    bbox_x2 = bbox_x1 + objects["bbox_width"]
                    bbox_y1 = objects["bbox_y"]
                    bbox_y2 = bbox_y1 + objects["bbox_height"]
                    objs["boxes"].append([bbox_x1, bbox_y1, bbox_x2, bbox_y2])

                elif self.label_mode == 'COCO':
                    bbox_cx = objects["bbox_x"] + objects["bbox_width"] / 2
                    bbox_cy = objects["bbox_y"] + objects["bbox_height"] / 2
                    bbox_w = objects["bbox_width"]
                    bbox_h = objects["bbox_height"]
                    objs["boxes"].append([bbox_cx, bbox_cy, bbox_w, bbox_h])

        objs["frame_type"] = np.array(objs["frame_type"])
        objs["labels"] = np.array(objs["labels"])
        objs["boxes"] = np.array(objs["boxes"])

        return objs


class StateChgObjDataPreprocessor():
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self):
        return self._simple_transform()

    def _simple_transform(self):
        transform = transforms.Compose([
            transforms.Lambda(
                lambda x: torch.as_tensor(x, dtype=torch.float)
            ),
            transforms.Normalize([0.45],[0.225])
        ])

        return transform
