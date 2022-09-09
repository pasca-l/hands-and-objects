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


class PNRTempLocDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, ann_dir, ann_task_name='fho_hands',
                 batch_size=1):
        super().__init__()
        self.ann_task_name = ann_task_name
        self.path_dict = {
            "data_dir": data_dir,
            "train_json": f"{ann_dir}{ann_task_name}_train.json",
            "val_json": f"{ann_dir}{ann_task_name}_val.json",
            "test_json": f"{ann_dir}{ann_task_name}_test_unannotated.json"
        }
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            json_handler = JsonHandler(self.ann_task_name)
            self.train_data =\
                PNRTempLocDataset('train', json_handler, **self.path_dict)
            self.val_data =\
                PNRTempLocDataset('val', json_handler, **self.path_dict)

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


class PNRTempLocDataset(Dataset):
    def __init__(self, phase, json_handler, extraction=False, **path_dict):
        self.data_dir = path_dict["data_dir"]
        self.json_file = path_dict[f"{phase}_json"]
        self.clip_dir = f"{self.data_dir}clips/"
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

        # video_path = f"{self.data_dir}{info['clip_uid']}.mp4"
        # video = cv2.VideoCapture(video_path)
        # info["original_fps"] = video.get(cv2.CAP_PROP_FPS)

        frames, labels, fps, frame_nums = self._sample_clip_with_label(info)
        frames = torch.as_tensor(frames, dtype=torch.float).permute(3, 0, 1, 2)
        frames = self.transform(frames)

        info["sample_frame_num"] = frame_nums
        info["effective_fps"] = fps

        return frames, labels, info

    def _sample_clip_with_label(self, info):
        # fps = info["original_fps"]
        fps = 30
        start_frame = info["clip_start_frame"]
        end_frame = info["clip_end_frame"]
        pnr_frame = info["clip_pnr_frame"]

        random_start_frame, random_end_frame =\
            self._random_clipping(
                pnr_frame, start_frame, end_frame, min_ratio=0.6)
        sample_frame_num, frame_pnr_dist =\
            self._sample_out_frames(
                pnr_frame, random_start_frame, random_end_frame, 32)

        # frames = []
        # for frame_num in sample_frame_num:
        #     frame_path = f"{self.action_frame_dir}{info['clip_uid']}" +\
        #                  f"/{frame_num}.png"
        #     try:
        #         image = self._load_frame(frame_path)
        #     except:
        #         print(f"Image does not exist : {frame_path}")
        #         return
        #     frames.append(image)

        frames = self._load_frames(sample_frame_num, info)

        keyframe_idx = np.argmin(frame_pnr_dist)
        onehot_label = np.zeros(len(sample_frame_num))
        onehot_label[keyframe_idx] = 1

        clipped_seconds = (random_end_frame - random_start_frame) / fps
        effective_fps = len(sample_frame_num) / clipped_seconds

        return (
            np.concatenate(frames),
            np.array(onehot_label, dtype='float32'),
            effective_fps,
            sample_frame_num
        )

    def _random_clipping(self, pnr, start_frame, end_frame, min_ratio):
        max_len = end_frame - start_frame
        min_len = int((end_frame - start_frame) * min_ratio)
        random_size = np.random.randint(min_len, max_len, 1)
        random_pivot = np.random.randint(0, max_len - random_size, 1)
        random_start_frame = random_pivot + start_frame
        random_end_frame = random_pivot + start_frame + random_size

        offset = 0
        if pnr < random_start_frame:
            offset = pnr - random_start_frame
        if pnr > random_end_frame:
            offset = pnr - random_end_frame
        random_start_frame += offset
        random_end_frame += offset

        return int(random_start_frame), int(random_end_frame)

    def _sample_out_frames(self, pnr, start_frame, end_frame, to_total_frames):
        num_frames = end_frame - start_frame
        sample_rate = num_frames // to_total_frames
        sample_frame_num, frame_pnr_dist = [], []

        if pnr - start_frame < end_frame - pnr:
            frame_list = range(start_frame, end_frame + 1)
        else:
            frame_list = range(end_frame, start_frame + 1, -1)[::-1]

        for counter, frame_num in enumerate(frame_list):
            if counter % sample_rate == 0:
                sample_frame_num.append(frame_num)
                frame_pnr_dist.append(np.abs(frame_num - pnr))

        return (
            sample_frame_num[:to_total_frames], 
            frame_pnr_dist[:to_total_frames]
        )

    def _load_frames(self, frame_nums, info):
        video_path = f"{self.clip_dir}{info['clip_uid']}.mp4"
        video = cv2.VideoCapture(video_path)
        info["original_fps"] = video.get(cv2.CAP_PROP_FPS)

        if not video.isOpened():
            print(f"Video cannot be opened : {video_path}")
            return

        frames = []
        counter = 1

        while True:
            ret, frame = video.read()
            if ret == False:
                break
            if counter in sorted(frame_nums):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frame = np.expand_dims(frame, axis=0).astype(np.float32)
                frames.append(frame)

            counter += 1

        video.release()
        return frames

    # def _load_frame(self, frame_path):
    #     frame = cv2.imread(frame_path)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     frame = cv2.resize(frame, (224, 224))
    #     frame = np.expand_dims(frame, axis=0).astype(np.float32)

    #     return frame


class FrameTransform():
    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.Normalize([0.45],[0.225])
        ])

    def __call__(self, frame):
        return self.data_transform(frame)