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
        self.sample_num = 1 if selection == 'center' else sample_num

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
            metalabels = info.select("sample_pnr_diff").item().to_list()
            _, metalabels = self.transform(None, metalabels)
            return frames, labels, metalabels

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
        if self.selection == "center":
            pnr = info.select("parent_pnr_frame").item()
            labels = np.where(
                frame_nums == pnr, self.classes["pnr"], self.classes["other"],
            )

        elif self.selection in ["segsec", "segratio"]:
            label_idx = info.select("label_indicies").item()
            labels = np.zeros(self.sample_num)
            labels[label_idx] = self.classes["pnr"]

        return labels

    def _select_frames(self, info):
        frame_nums = []

        if self.selection == "center":
            frame_num = info.select("center_frame").item()
            frame_nums.append(frame_num)

        elif self.selection in ["segsec", "segratio"]:
            sample_frames = info.select("sample_frames").item().to_list()
            frame_nums.extend(sample_frames)

        return np.array(frame_nums)
