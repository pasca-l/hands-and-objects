import os
import json
from tqdm import tqdm
import cv2

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
                 ann_task_name='fho_hands', clip_dir=None):
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

        self.clip_dir = clip_dir
        self.action_clip_dir = f"{clip_dir}actions/"
        os.makedirs(self.action_clip_dir, exist_ok=True)
        self.action_frame_dir = f"{clip_dir}action_frames/"
        os.makedirs(self.action_frame_dir, exist_ok=True)

        self.transform = transform

        self.flatten_json = self._unpack_json()

        # for info in tqdm(self.flatten_json, desc='Trimming clip near action'):
        #     self._trim_around_action(info)
        for info in tqdm(self.flatten_json, desc='Extracting frames'):
            self._extract_action_clip_frame(info)

    def __len__(self):
        return len(self.flatten_json)

    def __getitem__(self, index):
        info = self.package[index]

        start_frame = info["clip_start_frame"]
        end_frame = info["clip_end_frame"]
        video_path = f"{self.action_clip_dir}{info['clip_uid']}" +\
                          f"_{start_frame}_{end_frame}.mp4"
        video = cv2.VideoCapture(video_path)
        original_fps = video.get(cv2.CAP_PROP_FPS)



        return

    def _unpack_json(self):
        """
        Unpacks annotation json file to list of dicts.
        """
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
                            f"video_{alias}_frame": None,
                            f"clip_{alias}_frame": None,
                            f"{alias}_hands": None,
                        }
                    else:
                        temp_dict = {
                            f"video_{alias}_frame": temp_data['frame'],
                            f"clip_{alias}_frame": temp_data['clip_frame'],
                            f"{alias}_hands": temp_data['boxes'],
                        }
                    json_dict |= temp_dict

                flatten_json_list.append(json_dict)

        print(f"Contained {len(flatten_json_list)} actions.")
        return flatten_json_list

    def _trim_around_action(self, info):
        """
        Trims video to 8s clips containing action.
        """
        start_frame = info["clip_start_frame"]
        end_frame = info["clip_end_frame"]
        video_save_path = f"{self.action_clip_dir}{info['clip_uid']}" +\
                          f"_{start_frame}_{end_frame}.mp4"

        if os.path.exists(video_save_path):
            return "Video clip already exists"

        video = cv2.VideoCapture(f"{self.clip_dir}{info['clip_uid']}.mp4")
        fps = video.get(cv2.CAP_PROP_FPS)
        v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        v_size = (v_width, v_height)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_save_path, fourcc, fps, v_size)

        for i in range(end_frame + 1):
            ret, frame = video.read()
            if ret == True and start_frame <= i:
                writer.write(frame)

        writer.release()
        video.release()

    def _extract_action_clip_frame(self, info):
        """
        Saves all frames of 8s clips containing action.
        """
        start_frame = info["clip_start_frame"]
        end_frame = info["clip_end_frame"]

        frame_save_dir = f"{self.action_frame_dir}{info['clip_uid']}/"
        os.makedirs(frame_save_dir, exist_ok=True)

        video = cv2.VideoCapture(f"{self.clip_dir}{info['clip_uid']}.mp4")

        for i in range(end_frame + 1):
            ret, frame = video.read()
            if ret == True and start_frame <= i:
                frame_save_path = f"{frame_save_dir}{i}.png"
                if os.path.exists(frame_save_path):
                    continue
                cv2.imwrite(frame_save_path, frame)

        video.release()

    def _sample_clip_with_label(self, info, to_total_frames=10):
        random_start_frame, random_end_frame = _random_clipping(info, 5, 8)
        sample_frame_num, frame_pnr_dist = _sample_out_frames()

    def _random_clipping(self, info, min_len, max_len=8):
        return random_start_frame, random_end_frame
