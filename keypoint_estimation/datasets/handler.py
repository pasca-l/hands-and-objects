import os
import json
import math
import numpy as np
import polars as pl


class KeypointEstAnnotationHandler:
    def __init__(
        self, dataset_dir, task_name, selection, sample_num, seg_arg,
        neg_ratio, fps=30, fast_load=False,
    ):
        data_dir = os.path.join(dataset_dir, "ego4d/v2/annotations")
        self.manifest_file = os.path.join(data_dir, "manifest.csv")
        self.ann_file = {
            "train": os.path.join(data_dir, f"{task_name}_train.json"),
            "val": os.path.join(data_dir, f"{task_name}_val.json"),
            "processed": os.path.join(data_dir, f"{task_name}_processed.json"),
        }
        self.selection = selection
        self.sample_num = sample_num
        self.seg_arg = seg_arg
        self.neg_ratio = neg_ratio
        self.fps = fps
        self.fast_load = fast_load

        self.df_full = self._load_full_df()

    def __call__(self, phase="train"):
        df = self._create_split(self.df_full)[phase]
        df = self._adjust_posneg_ratio(df, self.neg_ratio)
        return df

    def _load_full_df(self):
        if self.fast_load:
            print("Loading data from preprocessed annotation ...")
            df_full = pl.read_ndjson(self.ann_file["processed"])

        else:
            df_train = self._unpack_json_to_df("train")
            df_val = self._unpack_json_to_df("val")
            df_full = self._process_df(
                pl.concat([df_train, df_val], how="align")
            )

            # save created full DataFrame
            df_full.write_ndjson(self.ann_file["processed"])

        return df_full

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
        # add "parent_num_frames" (total number of frames in original video)
        df = self._add_parent_num_frames_column(df)

        # filter negative samples, for later use
        neg_df = df.filter(
            pl.col("state_change") == False
        )

        # rearrange to new dataset
        df_video = self._create_video_info(df)
        df_seg = self._create_segments(df)
        df = self._create_samples(df_video, df_seg)

        # expand dataframe for data points
        df = df.select(
            "video_uid", "parent_frame_num", "sample_frames",
        ).explode(
            "sample_frames",
        ).with_columns(
            df["parent_pnr_frame"].explode(),
            df["segment_start_frame"].explode(),
            df["segment_end_frame"].explode(),
            df["hard_label"].explode(),
            df["soft_label"].explode(),
            # df["sample_pnr_diff"].explode(),
        ).with_columns(
            pl.when(
                pl.col("parent_pnr_frame").list.lengths() == 0
            ).then(False).otherwise(True).alias("state_change")
        )

        # replace negative samples
        df = self._replace_negative_samples(df, neg_df)
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
        df = df.with_columns(
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

        return df

    def _add_parent_num_frames_column(self, df):
        man_df = self._unpack_manifest_to_df()
        df_added = df.join(
            man_df.select(
                ["video_uid", "canonical_num_frames"],
            ),
            on="video_uid",
            how="inner",
        ).rename(
            {"canonical_num_frames": "parent_frame_num"},
        )

        return df_added

    def _create_video_info(self, df):
        print("Creating video level information ...")

        # create list of pnr frames per video uid
        df = df.groupby(
            "video_uid", "parent_frame_num",
        ).agg(
            pl.col("parent_pnr_frame").drop_nulls(),
        )

        # add global nearest pnr
        # data must be filtered so that there is at least 1 pnr in the video
        df = df.filter(
            # filtering for global nearest pnr calculation
            pl.col("parent_pnr_frame").list.lengths() > 0,
        ).with_columns(
            pl.struct(
                ["parent_frame_num", "parent_pnr_frame"],
            ).apply(
                # np.arange()[:,np.newaxis] - np.array(),
                # creates arrays containing distance from a given pivot
                # eg. a = np.arange(1, 6), b = np.array([0, 3])
                #     a[:,np.newaxis] - b with np.abs()
                #     >> np.array([
                #           [1, 2, 3, 4, 5],
                #           [2, 1, 0, 1, 2]
                #        ])
                #     -> np.min() vertically, would give the nearest PNR dist
                lambda c: np.min(
                    np.abs(
                        np.arange(1, c["parent_frame_num"]+1)[:,np.newaxis] \
                        - np.array(c["parent_pnr_frame"])
                    ),
                    axis=1,
                ).tolist(),
            ).alias("nearest_pnr_diff"),
        )

        return df

    def _create_segments(self, df):
        print("Creating segments ...")

        df = df.select(
            "video_uid", "parent_frame_num",
        ).unique(
            maintain_order=True,
        ).with_columns(
            pl.when(
                self.selection == "segsec"
            ).then(
                pl.lit(self.seg_arg * self.fps).alias("step")
            ).when(
                self.selection == "segratio"
            ).then(
                (pl.col("parent_frame_num") / self.seg_arg).ceil()
                .cast(pl.Int32).alias("step")
            )
        )

        df = df.with_columns(
            pl.struct(
                ["parent_frame_num", "step"],
            ).apply(
                lambda c: [
                    i for i in
                    range(1, c["parent_frame_num"] - c["step"], c["step"])
                ],
            ).alias("segment_start_frame")
        ).with_columns(
            pl.struct(
                ["step", "segment_start_frame"],
            ).apply(
                lambda c: [i + c["step"] - 1 for i in c["segment_start_frame"]],
            ).alias("segment_end_frame")
        )

        return df

    def _create_samples(self, df_video, df_seg):
        print("Creating samples ...")
        df = df_video.join(df_seg, on=["video_uid", "parent_frame_num"])

        # sample frame numbers within a given range
        df = df.with_columns(
            pl.struct(
                ["segment_start_frame", "segment_end_frame"],
            ).apply(
                lambda c: [
                    np.linspace(
                        start, end, self.sample_num, dtype=int
                    ).tolist()
                    for start, end in zip(
                        c["segment_start_frame"], c["segment_end_frame"]
                    )
                ]
            ).alias("sample_frames")
        )

        # add hard labels
        df = df.with_columns(
            pl.struct(
                ["parent_pnr_frame", "sample_frames"],
            ).apply(
                lambda c: [
                    np.array(c["sample_frames"]).flatten()[
                        np.argmin(np.abs(
                            np.array(c["sample_frames"]).flatten() - pnr),
                        )
                    ]
                    # some PNR are annotated out of the range of segments,
                    # PNR is considered only up to the max value of sample
                    for pnr in (
                        np.array(c["parent_pnr_frame"])[
                            np.array(c["parent_pnr_frame"]) < \
                            np.array(c["sample_frames"]).max()
                        ]
                    )
                ]
            ).alias("label_frames")
        )
        df = df.with_columns(
            pl.struct(
                ["sample_frames", "label_frames"]
            ).apply(
                lambda c: [
                    np.isin(np.array(sample), c["label_frames"]) * 1
                    for sample in c["sample_frames"]
                ]
            ).alias("hard_label")
        )

        # prepare 1d gauss distribution with -3σ<x<3σ fit within acceptance
        mu, sigma, amp = 0, 1, 1
        acceptance = 5
        x = np.linspace(-3 * sigma, 3 * sigma, acceptance)
        gauss = amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        # add soft labels
        df = df.with_columns(
            pl.struct(
                ["hard_label"],
            ).apply(
                # convolute hard label with gauss distribution,
                # and clip with amplitude 1, for neighboring effect
                lambda c: np.clip(
                    np.convolve(
                        np.array(c["hard_label"]).flatten(), gauss, mode="same"
                    ), 0, 1
                ).reshape(
                    np.array(c["hard_label"]).shape
                ).tolist()
            ).alias("soft_label"),
        )

        # add global nearest pnr frame
        df = df.with_columns(
            pl.struct(
                ["nearest_pnr_diff", "sample_frames"],
            ).apply(
                lambda c: np.array(c["nearest_pnr_diff"])[
                    np.array(c["sample_frames"]) - 1
                ].tolist()
            ).alias("sample_pnr_diff")
        )

        # divide parent_pnr_frame into segment range
        df = df.with_columns(
            pl.struct(
                ["parent_pnr_frame", "segment_start_frame", "segment_end_frame"]
            ).apply(
                lambda c: [
                    np.array(c["parent_pnr_frame"])[
                        np.where(
                            (np.array(c["parent_pnr_frame"]) >= start) & \
                            (np.array(c["parent_pnr_frame"]) <= end)
                        )
                    ] for start, end in zip(
                        c["segment_start_frame"], c["segment_end_frame"]
                    )
                ],
            )
        )

        return df

    def _replace_negative_samples(self, df, neg_df):
        # TODO: may need to change code
        # 1. adjusting the length of negatives (to get even sample interval)
        # 2. addition of "sample_pnr_diff"? -> currently unavailable, if video contains no PNR
        neg_df = neg_df.select(
            pl.col("video_uid"),
            pl.col("parent_frame_num"),
            (pl.col("parent_start_frame")+1).alias("segment_start_frame"),
            (pl.col("parent_end_frame")+1).alias("segment_end_frame"),
            pl.col("state_change"),
        ).with_columns([
            pl.col("video_uid").apply(lambda _: []).cast(pl.List(pl.Int64)).alias("parent_pnr_frame"),
            pl.col("video_uid").apply(lambda c: np.zeros(self.sample_num).tolist()).cast(pl.List(pl.Int64)).alias("hard_label"),
            pl.col("video_uid").apply(lambda c: np.zeros(self.sample_num).tolist()).alias("soft_label"),
            pl.struct(
                ["segment_start_frame", "segment_end_frame"],
            ).apply(
                lambda c: np.linspace(
                    c["segment_start_frame"], c["segment_end_frame"],
                    self.sample_num, dtype=int
                ).tolist()
            ).alias("sample_frames"),
        ])
        neg_df = neg_df.select(df.columns)

        df = pl.concat([
            df.filter(pl.col("state_change") == True),
            neg_df,
        ])

        return df

    def _adjust_posneg_ratio(self, df, neg_ratio):
        if neg_ratio == None:
            return df

        df_pos = df.filter(
            pl.col("state_change") == True
        )
        df_neg = df.filter(
            pl.col("state_change") == False
        )

        pos_count = df_pos.height
        neg_count = int(pos_count * neg_ratio)

        df = df_pos.vstack(df_neg.sample(n=neg_count, seed=0))

        return df

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
