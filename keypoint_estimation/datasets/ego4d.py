import os
import sys
import git
import cv2
import math
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
        with_info=None,
        extract=False,
        image_level=True,
    ):
        super().__init__()

        self.frame_dir = os.path.join(dataset_dir, "ego4d/frames")
        self.transform = transform
        self.with_info = with_info
        self.image_level = image_level

        handler = AnnotationHandler(dataset_dir, task, phase)
        self.ann_len = len(handler)
        self.man_df, self.ann_df = handler()

        if extract:
            extractor = VideoExtractor(self.ann_df, dataset_dir)
            extractor.extract_frames()

    def __len__(self):
        return self.ann_len

    def __getitem__(self, index):
        info = self.ann_df[index]
        video_uid = info.select("video_uid").item()
        frame_nums = self._select_frames(info)

        for i in self.ann_df.iter_rows(named=True):
            print(i)
            break

        print(video_uid, frame_nums)

        return info.select("parent_pnr_frame").item()

    def _get_frames(self, video_uid, frame_nums):
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

        return frames

    def _get_labels(self, video_uid, frame_nums):
        labels = []

    def _select_frames(self, info):
        frame_nums = []

        start = info.select("parent_start_frame").item()
        end = info.select("parent_end_frame").item()
        pnr = info.select("parent_pnr_frame").item()

        if self.image_level:
            if info.select("state_change").item():
                frame_num = pnr
            else:
                frame_num = math.floor((end - start) / 2) + start

            frame_nums.append(frame_num)

        return frame_nums
