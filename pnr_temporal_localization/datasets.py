import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FrameTransform():
    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.Normalize([0.45],[0.225])
        ])

    def __call__(self, frame):
        return self.data_transform(frame)


class PNRTempLocDataset(Dataset):
    def __init__(self, phase='train', transform=None, ann_dir=None,
                 ann_task_name='fho_hands', data_dir=None):
        self.phase = phase
        self.train_json_file = f"{ann_dir}{ann_task_name}_train.json"
        self.val_json_file = f"{ann_dir}{ann_task_name}_val.json"
        self.test_json_file = f"{ann_dir}{ann_task_name}_test_unannotated.json"
        if self.phase == 'train':
            self.json_file = self.train_json_file
        elif self.phase == 'val':
            self.json_file = self.val_json_file
        elif self.phase == 'test':
            self.json_file = self.test_json_file

        self.transform = transform

        self.flatten_json = self._unpack_json()

    def __len__(self):
        return len(self.flatten_json)

    def __getitem__(self, index):
        return

    def _unpack_json(self):
        flatten_json_list = list()

        json_data = json.load(open(self.json_file, 'r'))
        for data in tqdm(json_data['clips'], desc='Preparing data'):
            for frame_data in data['frames']:
                json_dict = {
                    "clip_id": data['clip_id'],
                    "clip_uid": data['clip_uid'],
                    "video_uid": data['video_uid'],
                    "video_start_sec": frame_data['action_start_sec'],
                    "video_end_sec": frame_data['action_end_sec'],
                    "video_start_frame": frame_data['action_start_frame'],
                    "video_end_frame": frame_data['action_end_frame'],
                    "clip_start_sec": frame_data['action_clip_start_sec'],
                    "clip_end_sec": frame_data['action_clip_end_sec'],
                    "clip_start_frame": frame_data['action_clip_start_frame'],
                    "clip_end_frame": frame_data['action_clip_end_frame'],
                }

                frame_alias_dict = {
                    'pre_45': "pre45",
                    'pre_30': "pre30",
                    'pre_15': "pre15",
                    'pre_frame': "pre",
                    'post_frame': "post",
                    'pnr_frame': "pnr"
                }
                for frame_type, alias in frame_alias_dict.items():
                    try:
                        temp_data = frame_data[frame_type]
                    except KeyError:
                        temp_dict = {
                            f"video_{alias}_frame": "",
                            f"clip_{alias}_frame": "",
                            f"{alias}_hands": "",
                        }
                    else:
                        temp_dict = {
                            f"video_{alias}_frame": temp_data['frame'],
                            f"clip_{alias}_frame": temp_data['clip_frame'],
                            f"{alias}_hands": temp_data['boxes'],
                        }
                    json_dict |= temp_dict

                flatten_json_list.append(json_dict)

        return flatten_json_list
