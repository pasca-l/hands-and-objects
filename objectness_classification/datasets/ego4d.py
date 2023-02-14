import os
import cv2
import numpy as np
from torch.utils.data import Dataset

sys.path.append("../../utils/ego4d")
from json_handler import JsonHandler
from video_extractor import Extractor


class Ego4DObjnessClsDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        task='fho_scod',
        phase='train',
        transform=None,
        label_mode='corners',  # ['corners', 'COCO']
        with_info=False,
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

        if self.with_info:
            return frames, labels, info

        return frames, labels

    def _get_frames(self, info):
        frames = []
        for frame_type in ['pre_frame', 'pnr_frame', 'post_frame']:
            frame_path = f"{self.frame_dir}{info['clip_uid']}/" +\
                         f"{info[f'{frame_type}_num_clip']}.jpg"
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

        mask = self._create_mask(info['clip_uid'], objs, 2)
        return mask

    def _create_mask(self, clip_uid, objs, label=2):
        frame_path = f"{self.frame_dir}{clip_uid}/sample.jpg"
        frame = cv2.imread(frame_path)
        height, width, _ = frame.shape

        masks = []
        for frame_type in range(3):
            mask = np.zeros((height, width))
            point_idx = np.where(
                (objs["labels"] == label) & (objs["frame_type"] == frame_type)
            )
            points = objs["points"][point_idx]

            mask = cv2.fillPoly(mask, np.int32(points), 1)
            mask = cv2.resize(mask, (224, 224))

            masks.append(mask)

        return np.array(masks)
