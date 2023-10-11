import os
import json
import math
import numpy as np
import polars as pl


class AnnotationHandler:
    def __init__(self, dataset_dir, task_name, phase, selection, sample_num=16):
        data_dir = os.path.join(dataset_dir, "ego4d/v2/annotations")
        self.manifest_file = os.path.join(data_dir, "manifest.csv")
        self.ann_file = {
            "train": os.path.join(data_dir, f"{task_name}_train.json"),
            "val": os.path.join(data_dir, f"{task_name}_val.json")
        }
        self.phase = phase
        self.selection = selection
        self.sample_num = sample_num

        df_train = self._unpack_json_to_df("train")
        df_val = self._unpack_json_to_df("val")
        self.df_full = pl.concat([df_train, df_val], how="align")

    def __call__(self):
        df = self._process_df(self.df_full)
        df = self._create_split(df)[self.phase]
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

    def _process_df(self, df):
        if self.selection == "center":
            df = self._add_center_frame_column(df)

        elif self.selection in ["segsec", "segratio"]:
            df = self._add_parent_num_frames_column(df)
            df = self._format_ann_to_video_segments(df, self.selection)
            df = self._add_sample_frames_and_labels(df, self.sample_num)

        return df

    def _create_split(self, df):
        vids = df.select(
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
            "train": df.join(train_vids, on="video_uid", how="semi"),
            "val": df.join(val_vids, on="video_uid", how="semi"),
            "test": df.join(test_vids, on="video_uid", how="semi"),
            "all": df,
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

    def _format_ann_to_video_segments(
        self, df, selection, seg_sec=8, seg_ratio=100, fps=30
    ):
        video_uids = []
        parent_frame_num = []
        segment_start_frame = []
        segment_end_frame = []
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

            if selection == "segsec":
                step = seg_sec * fps
            elif selection == "segratio":
                step = math.ceil(frame_num / seg_ratio)

            start_frames = [i for i in range(1, frame_num - step, step)]
            end_frames = [i + step - 1 for i in start_frames]

            # recording values into lists
            record_num = len(start_frames)
            video_uids.extend([vid for _ in range(record_num)])
            parent_frame_num.extend([frame_num for _ in range(record_num)])
            keyframes.extend([
                pnr_frames[np.where((pnr_frames >= f) & (pnr_frames < f+step))]
                for f in start_frames
            ])
            segment_start_frame.extend(start_frames)
            segment_end_frame.extend(end_frames)

        df = pl.DataFrame({
            "video_uid": video_uids,
            "parent_pnr_frame": keyframes,
            "parent_frame_num": parent_frame_num,
            "segment_start_frame": segment_start_frame,
            "segment_end_frame": segment_end_frame,
        })
        # .with_columns(
        #     pl.when(
        #         pl.col("parent_pnr_frame").list.lengths() == 0
        #     ).then(False).otherwise(True).alias("state_change")
        # )

        return df

    def _add_sample_frames_and_labels(self, df, sample_num):
        df_with_sample_frames_and_labels = df.with_columns(
            pl.struct(
                ["segment_start_frame", "segment_end_frame"],
            ).apply(
                lambda c: np.linspace(
                    c["segment_start_frame"], c["segment_end_frame"],
                    sample_num, dtype=int,
                ).tolist(),
            ).alias("sample_frames"),
        ).with_columns(
            pl.struct(
                ["parent_pnr_frame", "sample_frames"]
            ).apply(
                lambda c: [
                    np.argmin(np.abs(np.array(c["sample_frames"]) - i))
                    for i in c["parent_pnr_frame"]
                ]
            ).alias("label_indicies"),
        )

        return df_with_sample_frames_and_labels

    def create_path_elements(self):
        df = self.__call__()

        if self.selection == "center":
            df_elem = df.select(
                "video_uid", "center_frame"
            )

        if self.selection in ["segsec", "segratio"]:
            df_elem = df.select(
                "video_uid", "sample_frames"
            ).explode(
                "sample_frames"
            )

        return df_elem
