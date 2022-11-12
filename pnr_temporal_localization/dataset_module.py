import importlib
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl


class PNRTempLocDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir, json_dict, model_name):
        super().__init__()
        self.data_dir = data_dir
        self.json_dict = json_dict
        self.batch_size = batch_size

        preprocessor = PNRTempLocDataPreprocessor()
        self.transform = preprocessor(model_name)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = PNRTempLocDataset(
                data_dir=self.data_dir,
                flatten_json=self.json_dict['train'],
                transform=self.transform
            )
            self.val_data = PNRTempLocDataset(
                data_dir=self.data_dir,
                flatten_json=self.json_dict['val'],
                transform=self.transform
            )

        if stage == "test":
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


class PNRTempLocDataset(Dataset):
    def __init__(self, data_dir, flatten_json, transform):
        self.frame_dir = f"{data_dir}frames/"
        self.flatten_json = flatten_json
        self.transform = transform

    def __len__(self):
        return len(self.flatten_json)

    def __getitem__(self, index):
        info = self.flatten_json[index]
        frames, labels = self._sample_clip_with_label(info)
        frames = self.transform(frames)

        return frames, labels, info

    def _sample_clip_with_label(self, info):
        start_frame = info["clip_start_frame"]
        end_frame = info["clip_end_frame"]
        pnr_frame = info["clip_pnr_frame"]

        random_start_frame, random_end_frame =\
            self._random_clipping(
                pnr_frame, start_frame, end_frame, min_ratio=0.6)
        sample_frame_num, frame_pnr_dist =\
            self._sample_out_frames(
                pnr_frame, random_start_frame, random_end_frame, 32)

        frames = self._load_frames(sample_frame_num, info)

        keyframe_idx = np.argmin(frame_pnr_dist)
        onehot_label = np.zeros(len(sample_frame_num))
        onehot_label[keyframe_idx] = 1

        # include values for metric
        info["total_frame_num"] = random_end_frame - random_start_frame
        info["fps"] = 30

        return (
            np.concatenate(frames),
            np.array(onehot_label, dtype='float32')
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
        frames = []

        for num in frame_nums:
            frame_path = f"{self.frame_dir}{info['clip_uid']}/{num}.png"

            try:
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frame = np.expand_dims(frame, axis=0).astype(np.float32)
            except:
                print(f"Image does not exist : {frame_path}")
                return

            frames.append(frame)

        return frames


class PNRTempLocDataPreprocessor():
    def __init__(self):
        pass

    def __call__(self, preprocess_name):
        if preprocess_name == 'hand_salience':
            return self._hand_map_transform()
        else:
            return self._simple_transform()

    def _simple_transform(self):
        transform = transforms.Compose([
            transforms.Lambda(
                lambda x: torch.as_tensor(x, dtype=torch.float)
                            .permute(3,0,1,2)
            ),
            transforms.Normalize([0.45],[0.225])
        ])

        return transform

    def _hand_map_transform(self):
        module = importlib.import_module('data_preprocess.hand_map')
        hand_saliency_map = module.AddHandMapTransform()
        transform = transforms.Compose([
            transforms.Lambda(hand_saliency_map),
            transforms.Lambda(
                lambda x: torch.as_tensor(x, dtype=torch.float)
                            .permute(3,0,1,2)
            ),
            transforms.Normalize([0.45],[0.225])
        ])

        return transform
