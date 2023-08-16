import os
import json
import polars as pl


class AnnotationHandler:
    def __init__(self, dataset_dir, task_name, phase, data_as="dataframe"):
        self.manifest_file = os.path.join(
            dataset_dir, "ego4d/annotations", "manifest.csv"
        )
        self.ann_file = {
            "train": os.path.join(
                dataset_dir, "ego4d/annotations", f"{task_name}_train.json"
            ),
            "val": os.path.join(
                dataset_dir, "ego4d/annotations", f"{task_name}_val.json"
            )
        }[phase]
        self.data_as = data_as

        # testing files
        self.manifest_file = "/Users/sy/program/git/hands-and-objects/keypoint_estimation/datasets/local/manifest.csv"
        self.ann_file = "/Users/sy/program/git/hands-and-objects/keypoint_estimation/datasets/local/fho_oscc-pnr_train.json"

    def __call__(self):
        if self.data_as == "dataframe":
            return self._unpack_manifest_to_df(), self._unpack_json_to_df()

    def __len__(self):
        if self.data_as == "dataframe":
            return self._unpack_json_to_df().select(pl.count()).item()

    def _unpack_manifest_to_df(self):
        return pl.read_csv(self.manifest_file)

    def _unpack_json_to_df(self):
        df = pl.read_ndjson(
            bytes("\n".join(
                [json.dumps(r) for r in json.load(
                    open(self.ann_file, 'r')
                )["clips"]]
            ), 'utf-8')
        )

        return df
