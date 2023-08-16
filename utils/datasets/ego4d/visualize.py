import json
import numpy as np
import polars as plr
from PIL import Image


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
    main()
