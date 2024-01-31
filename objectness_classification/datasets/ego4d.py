import os
import json
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Ego4DObjnessClsDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        task='fho_scod',
        phase='train',
        transform=None,
        with_info=False,
        label_mode='corners',  # ['corners', 'COCO']
    ):
        super().__init__()

        ann_handler = Ego4DAnnotationHandler(dataset_dir, task, phase)
        self.flatten_json = ann_handler()

        self.frame_dir = os.path.join(dataset_dir, "ego4d/frames")
        self.transform = transform
        self.label_mode = label_mode
        self.with_info = with_info

    def __len__(self):
        return len(self.flatten_json)

    def __getitem__(self, index):
        info = self.flatten_json[index]
        frames = self._get_frames(info)
        labels = self._get_labels(info)

        for i, (frame, label) in enumerate(zip(frames, labels)):
            frame, label = self.transform(frame, label)
            frames[i], labels[i] = frame, label

        frames = torch.stack(frames)
        labels = torch.stack(labels)

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

        return frames

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
            masks.append(np.array(label_masks).transpose(1,2,0))

        return masks


class Ego4DAnnotationHandler():
    def __init__(self, dataset_dir, task_name, phase):
        self.task_name = task_name
        self.ann_file = {
            "train": os.path.join(
                dataset_dir, "ego4d/annotations", f"{task_name}_train.json"
            ),
            "val": os.path.join(
                dataset_dir, "ego4d/annotations", f"{task_name}_val.json"
            ),
        }[phase]

    def __call__(self):
        """
        Unpacks annotation json file, according to the task name and phase.
        """
        if self.task_name == 'fho_hands':
            return self._fho_hands_unpack(self.ann_file)
        elif self.task_name == 'fho_scod':
            return self._fho_scod_unpack(self.ann_file)

    def _fho_hands_unpack(self, json_file, all_data=False):
        """
        The target annotation file is "fho_hands_PHASE.json".
        If all_data == True, unpacks possible pre and post frames, with hand 
        coordinates. (Some annotations might be missing; returns None).
        """
        flatten_json_list = []

        json_data = json.load(open(json_file, 'r'))
        for data in tqdm(json_data['clips'], desc='Preparing data'):

            for frame_data in data['frames']:
                # pnr frame must be included in any of the batch.
                try:
                    frame_data['pnr_frame']
                except KeyError:
                    continue

                json_dict = {
                    "clip_id": data['clip_id'],
                    "clip_uid": data['clip_uid'],
                    # "video_uid": data['video_uid'],
                    # "video_start_sec": frame_data['action_start_sec'],
                    # "video_end_sec": frame_data['action_end_sec'],
                    # "video_start_frame": frame_data['action_start_frame'],
                    # "video_end_frame": frame_data['action_end_frame'],
                    # "clip_start_sec": frame_data['action_clip_start_sec'],
                    # "clip_end_sec": frame_data['action_clip_end_sec'],
                    "clip_start_frame": frame_data['action_clip_start_frame'],
                    "clip_end_frame": frame_data['action_clip_end_frame'],
                    "clip_pnr_frame": frame_data['pnr_frame']['clip_frame']
                }

                if all_data == True:
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

        return flatten_json_list

    def _fho_scod_unpack(self, json_file):
        """
        The target annotation file is "fho_scod_PHASE.json".

        Every dict in the json list will result in the following format:
            dict
            |-  "clip_id"
            |-  "clip_uid"
            |-  "video_uid"
            |-  "clip_start_sec"
            |-  "clip_end_sec"
            |-  "clip_start_frame"
            |-  "clip_end_frame"
            |-  "(pre/pnr/post)_frame_num"
            |-  "(pre/pnr/post)_frame_num_clip"
            |-  "(pre/pnr/post)_frame_width"
            |-  "(pre/pnr/post)_frame_height"
            |-  "(pre/pnr/post)_frame_objects"
                |-  "object_type"
                |-  "structured_noun"
                |-  "instance_num"
                |-  "bbox_x"
                |-  "bbox_y"
                |-  "bbox_width"
                |-  "bbox_height"
        """
        flatten_json_list = []

        json_data = json.load(open(json_file, 'r'))
        for data in tqdm(json_data['clips'], desc='Preparing data'):
            json_dict = {
                "clip_id": data['clip_id'],
                "clip_uid": data['clip_uid'],
                # "video_uid": data['video_uid'],
                # "clip_start_sec": data['clip_parent_start_sec'],
                # "clip_end_sec": data['clip_parent_end_sec'],
                "clip_start_frame": data['clip_parent_start_frame'],
                "clip_end_frame": data['clip_parent_end_frame'],
            }

            for frame_type in ['pre_frame', 'pnr_frame', 'post_frame']:
                frame_data = data[frame_type]
                frame_dict = {
                    # f"{frame_type}_num": frame_data['frame_number'],
                    f"{frame_type}_num_clip": frame_data['clip_frame_number'],
                    f"{frame_type}_width": frame_data['width'],
                    f"{frame_type}_height": frame_data['height'],
                    f"{frame_type}_objects": []
                }

                for obj in frame_data['bbox']:
                    object_dict = {
                        "object_type": obj['object_type'],
                        "structured_noun": obj['structured_noun']
                        if obj['structured_noun'] else '',
                        # "instance_num": obj['instance_number'],
                        "bbox_x": obj['bbox']['x'],
                        "bbox_y": obj['bbox']['y'],
                        "bbox_width": obj['bbox']['width'],
                        "bbox_height": obj['bbox']['height']
                    }

                    frame_dict[f"{frame_type}_objects"].append(object_dict)

                json_dict |= frame_dict

            flatten_json_list.append(json_dict)

        return flatten_json_list
