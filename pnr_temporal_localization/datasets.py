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

        # self._trim_around_action()
        self._extract_action_clip_frame()


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
        info["original_fps"] = original_fps

        frames, labels, fps, frame_nums = self._sample_clip_with_label(info)
        frames = torch.as_tensor(frames).permute(3, 0, 1, 2)
        frames = self.transform(frames)

        info["sample_frame_num"] = frame_nums

        return frames, labels, None, fps, info

    def _unpack_json(self):
        """
        Unpacks annotation json file to list of dicts.
        The target annotation file is "fho_hands_PHASE.json".
        """
        flatten_json_list = []

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

        print(len(frame_dict))

        for clip_id, frame_nums in tqdm(frame_dict.items(),
                                        desc='Extracting frames'):
            frame_save_dir = f"{self.action_frame_dir}{clip_id}/"
            os.makedirs(frame_save_dir, exist_ok=True)

            video = cv2.VideoCapture(f"{self.clip_dir}{clip_id}.mp4")

            for i in range(end_frame + 1):
                ret, frame = video.read()
                if ret == True and i in frame_nums:
                    frame_save_path = f"{frame_save_dir}{i}.png"
                    if os.path.exists(frame_save_path):
                        continue
                    print(frame_save_path)
                    cv2.imwrite(frame_save_path, frame)

            video.release()

    def _load_frame(self, frame_path):
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = np.expand_dims(frame, axis=0).astype(np.float32)

        return frame

    def _sample_clip_with_label(self, info):
        random_start_frame, random_end_frame = _random_clipping(info, 5, 8)
        sample_frame_num, frame_pnr_dist =\
            _sample_out_frames(info, random_start_frame, random_end_frame)

        frames = []
        for frame_num in sample_frame_num:
            frame_path = f"{self.action_frame_dir}{info['clip_uid']}" +\
                         f"/{frame_num}.png"
            image = self._load_frame(frame_num)
            frames.append(image)

        keyframe_idx = np.argmin(frame_pnr_dist)
        onehot_label = np.zeros(len(sample_frame_num))
        onehot_label[keyframe_idx] = 1

        clipped_seconds = (random_end_frame - random_start_frame) / fps
        effective_fps = len(sample_frame_num) / clipped_seconds

        return (np.concatenate(frames), np.array(onehot_label),
                    effective_fps, sample_frame_num)

    def _random_clipping(self, info, min_len, max_len=8):
        fps = info["original_fps"]
        random_size = np.random.randint(0, (max_len - min_len) * fps, 1)
        random_pivot = np.random.randint(0, random_size, 1)
        random_start_frame = random_pivot
        random_end_frame = (max_len * fps) - (random_size - random_pivot)

        return random_start_frame, random_end_frame

    def _sample_out_frames(self, info, start_frame, end_frame, to_total_frames=10):
        pnr_frame = info['video_pnr_frame']
        num_frames = end_frame - start_frame
        sample_rate = num_frames // to_total_frames
        sample_frame_num, frame_pnr_dist = [], []

        for frame_count in range(start_frame, end_frame + 1):
            if frame_count % sample_rate == 0:
                sample_frame_num.append(frame_count)
                frame_pnr_dist.append(np.abs(frame_count - pnr_frame))

        return sample_frame_num, frame_pnr_dist
