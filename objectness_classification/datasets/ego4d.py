import os
import sys
import git
import cv2
import numpy as np
from torch.utils.data import Dataset

git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
sys.path.append(f"{git_root}/utils/ego4d")
from json_handler import JsonHandler
from video_extractor import Extractor


class Ego4DObjnessClsDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        task='fho_scod',
        phase='train',
        transform=None,
        with_info=False,
        label_mode='corners',  # ['corners', 'COCO']
        extract=False,
    ):
        super().__init__()

        json_handler = JsonHandler(dataset_dir, task, phase)
        self.flatten_json = json_handler()

        if extract:
            extractor = Extractor(dataset_dir, task)
            extractor.extract_frame_as_image(self.flatten_json)

        self.frame_dir = os.path.join(dataset_dir, "ego4d/frames")
        self.transform = transform
        self.label_mode = label_mode
        self.with_info = with_info

    def __len__(self):
        return len(self.flatten_json)

    def __getitem__(self, index):
        info = self.flatten_json[index]
        labels = self._get_labels(info)
        frames = self._get_frames(info)

        if self.transform != None:
            frames = self.transform(frames)
            # labels = self.transform(labels)

        if self.with_info:
            info["index"] = index
            return frames, labels, info

        return frames, labels

    def _get_frames(self, info):
        frames = []
        for frame_type in ['pre_frame', 'pnr_frame', 'post_frame']:
            frame_path = os.path.join(
                self.frame_dir,
                info['clip_uid'],
                f"{info[f'{frame_type}_num_clip']}.jpg"
            )
            try:
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
            except:
                print(f"Image does not exist : {frame_path}")
                return

            frames.append(frame)

        return np.array(frames)

    def _get_labels(self, info):
        objs = {
            "frame_type": [],
            "labels": [],
            "points": []
        }
        obj_cls_dict = {
            "left_hand": 0,
            "right_hand": 1,
            "object_of_change": 2,
            "tool": 3,
        }

        for i, frame in enumerate(['pre_frame', 'pnr_frame', 'post_frame']):
            for objects in info[f"{frame}_objects"]:
                objs["frame_type"].append(i)

                bbox_cls = obj_cls_dict[objects["object_type"]]
                objs["labels"].append(bbox_cls)

                if self.label_mode == 'corners':
                    bbox_x1 = objects["bbox_x"]
                    bbox_x2 = bbox_x1 + objects["bbox_width"]
                    bbox_y1 = objects["bbox_y"]
                    bbox_y2 = bbox_y1 + objects["bbox_height"]
                    objs["points"].append([
                        (bbox_x1, bbox_y1), (bbox_x1, bbox_y2),
                        (bbox_x2, bbox_y2), (bbox_x2, bbox_y1)
                    ])

                elif self.label_mode == 'COCO':
                    bbox_cx = objects["bbox_x"] + objects["bbox_width"] / 2
                    bbox_cy = objects["bbox_y"] + objects["bbox_height"] / 2
                    bbox_w = objects["bbox_width"]
                    bbox_h = objects["bbox_height"]
                    objs["points"].append([
                        (bbox_cx - bbox_w, bbox_cy - bbox_h),
                        (bbox_cx + bbox_w, bbox_cy - bbox_h),
                        (bbox_cx - bbox_w, bbox_cy + bbox_h),
                        (bbox_cx + bbox_w, bbox_cy + bbox_h)
                    ])

        objs["frame_type"] = np.array(objs["frame_type"])
        objs["labels"] = np.array(objs["labels"])
        objs["points"] = np.array(objs["points"])

        mask = self._create_mask(info['clip_uid'], objs)
        return mask

    def _create_mask(self, clip_uid, objs):
        frame_path = os.path.join(
            self.frame_dir,
            clip_uid,
            "sample.jpg"
        )
        frame = cv2.imread(frame_path)
        h, w, _ = frame.shape

        masks = []
        for frame_type in range(3):
            label_masks = []

            # mask for state change object
            stobj_pt_idx = np.where(
                (objs["labels"] == 2) & (objs["frame_type"] == frame_type)
            )
            stobj_pts = objs["points"][stobj_pt_idx]

            stobj = cv2.fillPoly(np.zeros((h, w)), np.int32(stobj_pts), 1)
            stobj = cv2.resize(stobj, (224, 224))

            # mask for hands
            lhand_pt_idx = np.where(
                (objs["labels"] == 0) & (objs["frame_type"] == frame_type)
            )
            lhand_pts = objs["points"][lhand_pt_idx]
            rhand_pt_idx = np.where(
                (objs["labels"] == 1) & (objs["frame_type"] == frame_type)
            )
            rhand_pts = objs["points"][rhand_pt_idx]

            hands = cv2.fillPoly(np.zeros((h, w)), np.int32(lhand_pts), 1)
            hands = cv2.fillPoly(hands, np.int32(rhand_pts), 1)
            hands = cv2.resize(hands, (224, 224))

            # mask for background
            bg = np.where(
                (stobj == 1) | (hands == 1),
                0.0, 1.0
            )

            label_masks.append(bg)
            label_masks.append(stobj)
            label_masks.append(hands)
            masks.append(label_masks)

        return np.array(masks)
