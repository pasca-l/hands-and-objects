import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl

sys.path.append("../utils")
from json_handler import JsonHandler
from video_extractor import Extractor


class StateChgObjDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, ann_dir, ann_task_name='fho_scod',
                 extract_options=[], batch_size=1):
        super().__init__()
        self.task_name = ann_task_name
        self.extracts = extract_options
        self.batch_size = batch_size
        self.path_dict = {
            "data_dir": data_dir,
            "train_json": f"{ann_dir}{ann_task_name}_train.json",
            "val_json": f"{ann_dir}{ann_task_name}_val.json",
        }
        self.json_handler = JsonHandler(ann_task_name)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = StateChgObjDataset(
                'train', self.task_name, self.extracts, **self.path_dict)
            self.val_data = StateChgObjDataset(
                'val', self.task_name, self.extracts, **self.path_dict)

        if stage == "test" or stage is None:
            self.test_data = None

        if stage == "predict":
            self.predict_data = None
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            pin_memory=True
        )


class StateChgObjDataset(Dataset):
    def __init__(self, phase, task_name, extraction, **path_dict):
        self.data_dir = path_dict["data_dir"]
        self.json_file = path_dict[f"{phase}_json"]
        self.frame_dir = f"{self.data_dir}clip_arrays/"

        self.transform = FrameTransform()

        json_handler = JsonHandler(task_name)
        self.flatten_json = json_handler(self.json_file)

        if extraction:
            extractor = Extractor(self.data_dir, self.flatten_json)
            for option in extraction:
                getattr(extractor, option)()

    def __len__(self):
        return len(self.flatten_json)

    def __getitem__(self, index):
        info = self.flatten_json[index]

        frames = self._get_frames(info)
        frames = torch.as_tensor(frames).permute(3, 0, 1, 2)
        frames = self.transform(frames)

        labels = self._get_labels(info)

        return frames, labels, info

    def _get_frames(self, info):
        frames = []
        for frame_type in ['pre_frame', 'pnr_frame', 'post_frame']:
            frame_path = f"{self.frame_dir}{info['clip_uid']}.npz"
            try:
                frame_arrays = np.load(frame_path)
                frame = frame_arrays[f"arr_{info[f'{frame_type}_num_clip']}"]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                image = np.expand_dims(frame, axis=0).astype(np.float32)
            except:
                print(f"Image does not exist : {frame_path}")
                return
            frames.append(image)

        return np.concatenate(frames)

    def _get_labels(self, info):
        object_labels = []
        obj_cls_dict = {
            "left_hand": 0,
            "right_hand": 1,
            "object_of_change": 2,
            "tool": 3,
        }

        for frame_type in ['pre_frame', 'pnr_frame', 'post_frame']:
            width = info[f"{frame_type}_width"]
            height = info[f"{frame_type}_height"]

            temp = []
            for objects in info[f"{frame_type}_objects"]:
                bbox_x = objects["bbox_x"] / width
                bbox_dx = objects["bbox_width"] / width
                bbox_y = objects["bbox_y"] / height
                bbox_dy = objects["bbox_height"] / height

                bbox_cls = obj_cls_dict[objects["object_type"]]
                temp.append([bbox_cls, bbox_x, bbox_y, bbox_dx, bbox_dy])

            object_labels.append(temp)

        return object_labels


class FrameTransform():
    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.Normalize([0.45],[0.225])
        ])

    def __call__(self, frame):
        return self.data_transform(frame)