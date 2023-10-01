import os
import json
import numpy as np
import polars as plr
import cv2
from PIL import Image

from handler import AnnotationHandler


class Visualizer:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        handler = AnnotationHandler(
            dataset_dir=dataset_dir,
            task_name="fho_oscc-pnr",
            phase="all",
            selection="center",
        )
        self.df_full = handler()

    def create_video_with_pnr_annotation(
        self, video_uid, save_as, fps_ratio=2
    ):
        pnr_frames = self.df_full.filter(
            plr.col("video_uid") == video_uid
        ).select(
            "center_frame"
        ).to_numpy().flatten()

        video_dir = os.path.join(self.dataset_dir, "ego4d/v2/full_scale")
        video_path = os.path.join(video_dir, f"{video_uid}.mp4")

        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        v_size = (v_width, v_height)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_as, fourcc, fps // fps_ratio, v_size)

        for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = video.read()
            if ret == False:
                break

            if i in pnr_frames:
                h, w, _ = frame.shape
                frame = cv2.rectangle(frame, (0,0), (w,h), (0,0,2), 3)
            writer.write(frame)

        writer.release()
        video.release()


def vis():
    # pnr at 179
    vid = "a7bffd05-bb79-45cd-8bd1-8c30c5553ddf"
    v = Visualizer(
        dataset_dir="",
    )
    v.create_video_with_pnr_annotation(
        video_uid=vid,
        save_as="./output.mp4",
    )


class SegmentationVisualizer:
    def __init__(self, df_manifest, df_train, df_val):
        self.df_manifest = df_manifest
        self.df_train = df_train
        self.df_val = df_val

        self.palette = np.array([
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
        ], dtype=np.uint8)

    def visualize_action(self, video_uid):
        df_man, df_trn, df_val = self._filter_data(video_uid)
        total_frames = df_man.to_numpy().reshape(1)[0]

        clip_range, pnrs = np.zeros(total_frames), np.zeros(total_frames)
        for start, end, pnr in df_trn.to_numpy():
            clip_range[start:end+1] = 1
            pnrs[pnr] = 1
        for start, end, pnr in df_val.to_numpy():
            clip_range[start:end+1] = 1
            pnrs[pnr] = 1

        action_list = clip_range + pnrs

        tiled_list = np.tile(action_list.astype(np.uint8), reps=(4000, 1))
        img = Image.fromarray(tiled_list).convert("P")
        img.putpalette(self.palette)
        img.save("./test.png")

        return action_list, tiled_list

    def _filter_data(self, video_uid):
        df_manifest_flt = self.df_manifest.filter(
            plr.col("video_uid") == video_uid
        ).select(
            plr.col("canonical_num_frames"),
        )
        df_train_flt = self.df_train.filter(
            (plr.col("state_change") == True)
            & (plr.col("video_uid") == video_uid)
        ).select(
            plr.col("parent_start_frame"),
            plr.col("parent_end_frame"),
            plr.col("parent_pnr_frame"),
        )
        df_val_flt = self.df_val.filter(
            (plr.col("state_change") == True)
            & (plr.col("video_uid") == video_uid)
        ).select(
            plr.col("parent_start_frame"),
            plr.col("parent_end_frame"),
            plr.col("parent_pnr_frame"),
        )

        return df_manifest_flt, df_train_flt, df_val_flt


def main():
    manifest_path = "../../../keypoint_estimation/datasets/local/manifest.csv"
    train_ann_path = "../../../keypoint_estimation/datasets/local/fho_oscc-pnr_train.json"
    val_ann_path = "../../../keypoint_estimation/datasets/local/fho_oscc-pnr_val.json"

    df_manifest = plr.read_csv(manifest_path)
    df_train = plr.read_ndjson(
        bytes("\n".join(
            [json.dumps(r) for r in json.load(open(train_ann_path, 'r'))["clips"]]
        ), 'utf-8')
    )
    df_val = plr.read_ndjson(
        bytes("\n".join(
            [json.dumps(r) for r in json.load(open(val_ann_path, 'r'))["clips"]]
        ), 'utf-8')
    )

    video_uid = df_manifest["video_uid"][1]

    visualizer = SegmentationVisualizer(df_manifest, df_train, df_val)
    visualizer.visualize_action(video_uid)


if __name__ == "__main__":
    vis()
