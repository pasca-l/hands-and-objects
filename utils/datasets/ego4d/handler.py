import os
import json
import polars as pl


class AnnotationHandler:
    def __init__(self, dataset_dir, task_name, phase, data_as="dataframe"):
        data_dir = os.path.join(dataset_dir, "ego4d/v2/annotations")
        self.manifest_file = os.path.join(data_dir, "manifest.csv")
        self.ann_file = {
            "train": os.path.join(data_dir, f"{task_name}_train.json"),
            "val": os.path.join(data_dir, f"{task_name}_val.json")
        }
        self.phase = phase
        self.data_as = data_as

    def __call__(self):
        if self.data_as == "dataframe":
            df_train = self._unpack_json_to_df("train")
            df_val = self._unpack_json_to_df("val")

            df = self._create_split_with_test(df_train, df_val)[self.phase]

            return self._unpack_manifest_to_df(), df

    def __len__(self):
        if self.data_as == "dataframe":
            _, df = self.__call__()
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

    def _create_split_with_test(self, df_train, df_val):
        df_full = pl.concat([df_train, df_val], how="align")

        test_vids = df_full.select("video_uid").unique(
            maintain_order=True,
        ).sample(
            fraction=0.1, seed=42
        )

        dfs = {
            "train": df_train.join(test_vids, on="video_uid", how="anti"),
            "val": df_val.join(test_vids, on="video_uid", how="anti"),
            "test": df_full.join(test_vids, on="video_uid", how="semi"),
        }

        return dfs
