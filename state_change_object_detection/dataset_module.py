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
                 batch_size=1):
        super().__init__()
        self.ann_task_name = ann_task_name
        self.path_dict = {
            "data_dir": data_dir,
            "train_json": f"{ann_dir}{ann_task_name}_train.json",
            "val_json": f"{ann_dir}{ann_task_name}_val.json",
        }
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            json_handler = JsonHandler(self.ann_task_name)
            self.train_data =\
                StateChgObjDataset('train', json_handler, **self.path_dict)
            self.val_data =\
                StateChgObjDataset('val', json_handler, **self.path_dict)

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
    def __init__(self, phase, json_handler, extraction=False, **path_dict):
        self.data_dir = path_dict["data_dir"]
        self.json_file = path_dict[f"{phase}_json"]
        # self.action_frame_dir = f"{self.data_dir}action_frames/"

        self.transform = FrameTransform()

        self.flatten_json = json_handler(self.json_file)
        # if extraction:
        #     extractor = Extractor(self.data_dir, self.flatten_json)
        #     extractor.extract_action_clip_frame()

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
            frame_path = f"{self.action_frame_dir}{info['clip_uid']}" +\
                         f"/{info[f'{frame_type}_num']}.png"
            try:
                image = self._load_frame(frame_path)
            except:
                print(f"Image does not exist : {frame_path}")
                return
            frames.append(image)

        return np.concatenate(frames)

    def _get_labels(self, info):
        object_labels = []
        obj_cls_dict = {
            "object_of_change": 0,
            "left_hand": 1,
            "right_hand": 2
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

    def _load_frame(self, frame_path):
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = np.expand_dims(frame, axis=0).astype(np.float32)

        return frame


class FrameTransform():
    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.Normalize([0.45],[0.225])
        ])

    def __call__(self, frame):
        return self.data_transform(frame)