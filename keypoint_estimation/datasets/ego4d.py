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


class Ego4DKeypointEstDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        task='fho_oscc-pnr',
        phase='train',
        transform=None,
        with_info=False,
        selection='center',  #['center', 'segsec', 'segratio'],
        sample_num=16,
    ):
        super().__init__()

        self.frame_dir = os.path.join(dataset_dir, "ego4d/v2/frames")
        self.video_dir = os.path.join(dataset_dir, "ego4d/v2/full_scale")
        self.transform = transform
        self.with_info = with_info
        self.selection = selection
        self.sample = 1 if selection == 'center' else sample_num

        self.classes = {
            "other": 0,
            "pnr": 1,
        }

        handler = AnnotationHandler(
            dataset_dir, task, phase, selection, sample_num
        )
        self.ann_len = len(handler)
        self.ann_df = handler()

    def __len__(self):
        return self.ann_len

    def __getitem__(self, index):
        info = self.ann_df[index]

        frame_nums = self._select_frames(info)

        frames = self._get_frames(info, frame_nums)
        labels = self._get_labels(info, frame_nums)

        frames, labels = self.transform(frames, labels)

        if self.with_info:
            return frames, labels, info

        return frames, labels

    def _get_frames(self, info, frame_nums):
        video_uid = info.select("video_uid").item()

        frames = []
        for num in frame_nums:
            # get image file, image should be extracted from video beforehand,
            # as cv2.VideoCapture cannot be used with parallel computing
            frame_path = os.path.join(self.frame_dir, video_uid, f"{num}.jpg")

            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
            else:
                raise Exception(f"No path at: {frame_path}")

            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

        return np.array(frames)

    def _get_labels(self, info, frame_nums):
        pnr = info.select("parent_pnr_frame").item()

        if self.selection == "center":
            labels = np.where(
                frame_nums == pnr,
                self.classes["pnr"],
                self.classes["other"],
            )

        elif self.selection in ["segsec", "segratio"]:
            label_idx = info.select("label_indicies").item()
            labels = np.zeros(self.sample)
            labels[label_idx] = self.classes["pnr"]

        return labels

    def _select_frames(self, info):
        frame_nums = []

        if self.selection == "center":
            frame_num = info.select("center_frame").item()
            frame_nums.append(frame_num)

        elif self.selection in ["segsec", "segratio"]:
            start = info.select("segment_start_frame").item()
            end = info.select("segment_end_frame").item()
            frame_nums.extend(np.linspace(start, end, self.sample, dtype=int))

        return np.array(frame_nums)
