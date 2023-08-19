import os
import sys
import git
import cv2
import numpy as np
from torch.utils.data import Dataset

git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
sys.path.append(f"{git_root}/utils/datasets/ego4d")
from handler import AnnotationHandler
from extractor import VideoExtractor


class Ego4DKeypointEstDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        task='fho_oscc-pnr',
        phase='train',
        transform=None,
        with_info=False,
        extract=False,
        image_level=True,
    ):
        super().__init__()

        self.frame_dir = os.path.join(dataset_dir, "ego4d/v2/frames")
        self.transform = transform
        self.with_info = with_info
        self.image_level = image_level

        self.classes = {
            "other": 0,
            "pnr": 1,
        }

        handler = AnnotationHandler(dataset_dir, task, phase)
        self.ann_len = len(handler)
        self.man_df, self.ann_df = handler(with_center=self.image_level)

        if extract:
            extractor = VideoExtractor(self.ann_df, dataset_dir)
            extractor.extract_frames()

    def __len__(self):
        return self.ann_len

    def __getitem__(self, index):
        info = self.ann_df[index]

        frame_nums = self._select_frames(info)

        frames = self._get_frames(info, frame_nums)
        labels = self._get_labels(info, frame_nums)

        if self.with_info:
            return frames, labels, info

        return frames, labels

    def _get_frames(self, info, frame_nums):
        video_uid = info.select("video_uid").item()

        frames = []
        for num in frame_nums:
            frame_path = os.path.join(self.frame_dir, video_uid, f"{num}.jpg")
            try:
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
            except:
                raise Exception(f"Image does not exist at: {frame_path}")

            frames.append(frame)

        return np.array(frames)

    def _get_labels(self, info, frame_nums, class_num=2):
        pnr = info.select("parent_pnr_frame").item()

        labels = []
        for num in frame_nums:
            if num == pnr:
                label = self.classes["pnr"]
            else:
                label = self.classes["other"]

            labels.append(label)

        return np.array(*labels)

    def _select_frames(self, info):
        frame_nums = []

        if self.image_level:
            frame_num = info.select("center").item()
            frame_nums.append(frame_num)

        # else:
        #     start = info.select("parent_start_frame").item()
        #     end = info.select("parent_end_frame").item()

        #     frame_nums.append(*[i for i in range(start, end + 1)])

        return frame_nums
