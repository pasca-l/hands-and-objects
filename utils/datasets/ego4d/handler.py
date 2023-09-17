import os
import json
import numpy as np
import polars as pl


class AnnotationHandler:
    def __init__(self, dataset_dir, task_name, phase, image_level):
        data_dir = os.path.join(dataset_dir, "ego4d/v2/annotations")
        self.manifest_file = os.path.join(data_dir, "manifest.csv")
        self.ann_file = {
            "train": os.path.join(data_dir, f"{task_name}_train.json"),
            "val": os.path.join(data_dir, f"{task_name}_val.json")
        }
        self.phase = phase
        self.image_level = image_level

    def __call__(self):
        df_train = self._unpack_json_to_df("train")
        df_val = self._unpack_json_to_df("val")

        df = self._create_split(df_train, df_val)[self.phase]

        if self.image_level:
            df = self._add_center_frame_column(df)
            return df
        else:
            df = self._add_parent_num_frames_column(df)
            df = self._format_ann_to_video_segments(df)
            return df

    def __len__(self):
        df = self.__call__()
        return df.select(pl.count()).item()

    def _unpack_manifest_to_df(self):
        return pl.read_csv(self.manifest_file)

    def _unpack_json_to_df(self, phase):
        df = pl.read_ndjson(
            bytes("\n".join(
                [json.dumps(r) for r in json.load(
                    open(self.ann_file[phase], 'r')
                )["clips"]]
            ), 'utf-8')
        )

        return df

    def _create_split(self, df_train, df_val):
        df_full = pl.concat([df_train, df_val], how="align")

        vids = df_full.select(
            "video_uid"
        ).unique(
            maintain_order=True,
        )

        train_vids = vids.sample(
            fraction=0.7,
            seed=42,
        )
        val_vids = vids.join(
            train_vids,
            on="video_uid",
            how="anti",
        ).sample(
            fraction=0.2/0.3,
            seed=42,
        )
        test_vids = vids.join(
            train_vids,
            on="video_uid",
            how="anti",
        ).join(
            val_vids,
            on="video_uid",
            how="anti",
        )

        dfs = {
            "train": df_full.join(train_vids, on="video_uid", how="semi"),
            "val": df_full.join(val_vids, on="video_uid", how="semi"),
            "test": df_full.join(test_vids, on="video_uid", how="semi"),
        }

        return dfs

    def _add_center_frame_column(self, df):
        df_with_center = df.with_columns(
            pl.when(
                pl.col("state_change") == True
            ).then(
                pl.col("parent_pnr_frame")
            ).otherwise(
                ((pl.col("parent_end_frame") - \
                  pl.col("parent_start_frame")) / 2).floor() + \
                pl.col("parent_start_frame")
            ).cast(pl.Int64).alias("center_frame")
        )

        return df_with_center

    def _add_parent_num_frames_column(self, df):
        man_df = self._unpack_manifest_to_df()
        df_with_total_num_frames = df.join(
            man_df.select(
                ["video_uid", "canonical_num_frames"],
            ),
            on="video_uid",
            how="inner",
        ).rename(
            {"canonical_num_frames": "parent_num_frames"},
        )

        return df_with_total_num_frames

    def _format_ann_to_video_segments(self, df, seg_sec=8, fps=30):
        video_uids = []
        parent_frame_num = []
        segment_start_frame = []
        keyframes = []

        iterator = df.select(
            "video_uid"
        ).unique(
            maintain_order=True,
        ).iter_rows()

        for [vid] in iterator:
            frame_num = df.select(
                pl.when(
                    pl.col("video_uid") == vid
                ).then(
                    pl.col("parent_num_frames")
                )
            ).unique().drop_nulls().item()

            pnr_frames = df.select(
                pl.when(
                    pl.col("video_uid") == vid
                ).then(
                    pl.col("parent_pnr_frame")
                )
            ).drop_nulls().to_numpy().flatten()

            start_frames = [
                i for i in range(1, frame_num-seg_sec*fps, seg_sec*fps)
            ]

            # recording values into lists
            record_num = len(start_frames)
            video_uids.extend([vid for _ in range(record_num)])
            parent_frame_num.extend([frame_num for _ in range(record_num)])
            keyframes.extend([
                np.where((pnr_frames > f) & (pnr_frames <= f+seg_sec*fps))[0]
                for f in start_frames
            ])
            segment_start_frame.extend(start_frames)

        df = pl.DataFrame({
            "video_uid": video_uids,
            "parent_pnr_frame": keyframes,
            "parent_frame_num": parent_frame_num,
            "segment_start_frame": segment_start_frame,
        }).with_columns(
            (
                pl.col("segment_start_frame") + seg_sec * fps - 1
            ).alias("segment_end_frame"),
            # pl.when(
            #     pl.col("parent_pnr_frame").list.lengths() == 0
            # ).then(False).otherwise(True).alias("state_change")
        )

        return df
