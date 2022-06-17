import os
import json
from tqdm import tqdm
import cv2
import numpy as np

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
    def __init__(self, phase='train', ann_dir=None, clip_dir=None,
                 ann_task_name='fho_hands'):
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

        self.transform = FrameTransform()

        self.flatten_json = self._unpack_json()

        # self._trim_around_action()
        # self._extract_action_clip_frame()

    def __len__(self):
        return len(self.flatten_json)

    def __getitem__(self, index):
        info = self.flatten_json[index]

        # video_path = f"{self.clip_dir}{info['clip_uid']}.mp4"
        # video = cv2.VideoCapture(video_path)
        # info["original_fps"] = video.get(cv2.CAP_PROP_FPS)
        # print(video_path, video.get(cv2.CAP_PROP_FPS))

        frames, labels, fps, frame_nums = self._sample_clip_with_label(info)
        frames = torch.as_tensor(frames).permute(3, 0, 1, 2)
        frames = self.transform(frames)

        info["sample_frame_num"] = frame_nums
        info["effective_fps"] = fps

        return frames, labels, info

    def _unpack_json(self):
        """
        Unpacks annotation json file to list of dicts.
        The target annotation file is "fho_hands_PHASE.json".
        """
        flatten_json_list = []

        json_data = json.load(open(self.json_file, 'r'))
        for data in tqdm(json_data['clips'], desc='Preparing data'):
            for frame_data in data['frames']:
                # ------------
                try:
                    frame_data['pnr_frame']
                except KeyError:
                    continue
                # ------------

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
                    # ---------
                    "clip_pnr_frame": frame_data['pnr_frame']['clip_frame']
                }

                # frame_alias_dict = {
                #     'pre_45': "pre45",
                #     'pre_30': "pre30",
                #     'pre_15': "pre15",
                #     'pre_frame': "pre",
                #     'post_frame': "post",
                #     'pnr_frame': "pnr"
                # }
                # for frame_type, alias in frame_alias_dict.items():
                #     try:
                #         temp_data = frame_data[frame_type]
                #     except KeyError:
                #         temp_dict = {
                #             f"video_{alias}_frame": None,
                #             f"clip_{alias}_frame": None,
                #             f"{alias}_hands": None,
                #         }
                #     else:
                #         temp_dict = {
                #             f"video_{alias}_frame": temp_data['frame'],
                #             f"clip_{alias}_frame": temp_data['clip_frame'],
                #             f"{alias}_hands": temp_data['boxes'],
                #         }
                #     json_dict |= temp_dict

                flatten_json_list.append(json_dict)

        print(f"Contained {len(flatten_json_list)} actions.")
        return flatten_json_list

    def _trim_around_action(self):
        """
        Trims video to 8s clips containing action.
        """
        for info in tqdm(self.flatten_json, desc='Trimming clip near action'):
            start_frame = info["clip_start_frame"]
            end_frame = info["clip_end_frame"]
            video_save_path = f"{self.action_clip_dir}{info['clip_uid']}" +\
                              f"_{start_frame}_{end_frame}.mp4"

            if os.path.exists(video_save_path):
                continue

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

    def _extract_action_clip_frame(self):
        """
        Saves all frames of 8s clips containing action.
        """
        frame_dict = {}
        for info in tqdm(self.flatten_json, desc='Finding frames to extract'):
            frame_dict.setdefault(info['clip_uid'], set())

            start_frame = info["clip_start_frame"]
            end_frame = info["clip_end_frame"]
            frame_dict[info['clip_uid']] |=\
                {i for i in range(start_frame, end_frame + 1)}

        existing_frame_dirs = [d for d in os.listdir(self.action_frame_dir)
                               if os.path.isdir(f"{self.action_frame_dir}{d}")]
        for d in tqdm(existing_frame_dirs,
                      desc='Excluding existing frames to extract'):
            try:
                frame_dict[d] -=\
                    {int(f[:-4]) for f in os.listdir(
                        f"{self.action_frame_dir}{d}")}
            except KeyError:
                continue

        for clip_id, frame_nums in tqdm(frame_dict.items(),
                                        desc='Extracting frames'):
            if len(frame_nums) == 0:
                continue

            frame_save_dir = f"{self.action_frame_dir}{clip_id}/"
            os.makedirs(frame_save_dir, exist_ok=True)

            video = cv2.VideoCapture(f"{self.clip_dir}{clip_id}.mp4")

            counter = 1
            while True:
                ret, frame = video.read()
                if ret == False:
                    break
                if counter in frame_nums:
                    frame_save_path = f"{frame_save_dir}{counter}.png"
                    cv2.imwrite(frame_save_path, frame)
                counter += 1

            video.release()

    def _load_frame(self, frame_path):
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = np.expand_dims(frame, axis=0).astype(np.float32)

        return frame

    def _sample_clip_with_label(self, info):
        # fps = info["original_fps"]
        pnr_frame = info["clip_pnr_frame"]
        start_frame = info["clip_start_frame"]
        end_frame = info["clip_end_frame"]

        random_start_frame, random_end_frame, random_size, random_pivot =\
            self._random_clipping(
                pnr_frame, start_frame, end_frame, min_ratio=0.6)
        frame_range, sample_frame_num, frame_pnr_dist =\
            self._sample_out_frames(
                pnr_frame, random_start_frame, random_end_frame, 5)

        print(start_frame, frame_range, random_size, random_pivot, sample_frame_num, end_frame)

        frames = []
        for frame_num in sample_frame_num:
            frame_path = f"{self.action_frame_dir}{info['clip_uid']}" +\
                         f"/{frame_num}.png"
            try:
                image = self._load_frame(frame_path)
            except:
                print(f"Image does not exist : {frame_path}")
                return
            frames.append(image)

        keyframe_idx = np.argmin(frame_pnr_dist)
        onehot_label = np.zeros(len(sample_frame_num))
        onehot_label[keyframe_idx] = 1

        clipped_seconds = (random_end_frame - random_start_frame) / 30
        effective_fps = len(sample_frame_num) / clipped_seconds

        return (np.concatenate(frames), np.array(onehot_label),
                    effective_fps, sample_frame_num)

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

        return int(random_start_frame), int(random_end_frame), random_size, random_pivot

    def _sample_out_frames(self, pnr, start_frame, end_frame, to_total_frames):
        num_frames = end_frame - start_frame
        sample_rate = num_frames // to_total_frames
        sample_frame_num, frame_pnr_dist = [], []

        if pnr - start_frame < end_frame - pnr:
            frame_range = range(start_frame, end_frame + 1)
        else:
            frame_range = range(end_frame, start_frame + 1, -1)[::-1]

        for counter, frame_num in enumerate(frame_range):
            if counter % sample_rate == 0:
                sample_frame_num.append(frame_num)
                frame_pnr_dist.append(np.abs(frame_num - pnr))

        return (frame_range, sample_frame_num[:to_total_frames], 
                frame_pnr_dist[:to_total_frames])