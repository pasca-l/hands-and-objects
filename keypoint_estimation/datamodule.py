import os
import lightning as L
from torch.utils.data import DataLoader

from datasets import (
    KeypointEstAnnotationHandler,
    Ego4DKeypointEstDataset,
    KeypointEstDataPreprocessor,
)


class KeypointEstDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir,
        dataset_mode="ego4d",
        batch_size=4,
        transform_mode="base",
        with_info=False,
        selection="center",  #["center", "segsec", "segratio"],
        sample_num=1,
        seg_arg=None,
        neg_ratio=None,
        fast_load=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_dir = dataset_dir
        self.dataset_mode = dataset_mode
        self.batch_size = batch_size
        self.with_info = with_info
        self.selection = selection
        self.sample_num = 1 if selection == "center" else sample_num

        self.num_workers=os.cpu_count()

        self.transform = KeypointEstDataPreprocessor(
            transform_mode=transform_mode
        )

        self.handler = KeypointEstAnnotationHandler(
            dataset_dir=dataset_dir,
            task_name="fho_oscc-pnr",
            selection=selection,
            sample_num=self.sample_num,
            seg_arg=seg_arg if selection in ["segsec", "segratio"] else None,
            neg_ratio=neg_ratio,
            fast_load=fast_load,
        )

    def setup(self, stage=None):
        if self.dataset_mode == "ego4d":
            if stage == "fit" or stage is None:
                self.train_data = Ego4DKeypointEstDataset(
                    dataset_dir=self.dataset_dir,
                    ann_df=self.handler(phase="train"),
                    transform=self.transform,
                    with_info=self.with_info,
                    selection=self.selection,
                    sample_num=self.sample_num,
                )
                self.val_data = Ego4DKeypointEstDataset(
                    dataset_dir=self.dataset_dir,
                    ann_df=self.handler(phase="val"),
                    transform=self.transform,
                    with_info=self.with_info,
                    selection=self.selection,
                    sample_num=self.sample_num,
                )

            if stage == "test":
                self.test_data = Ego4DKeypointEstDataset(
                    dataset_dir=self.dataset_dir,
                    ann_df=self.handler(phase="test"),
                    transform=self.transform,
                    with_info=self.with_info,
                    selection=self.selection,
                    sample_num=self.sample_num,
                )

            if stage == "predict":
                self.predict_data = None

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
