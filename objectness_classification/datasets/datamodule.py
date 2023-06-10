import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ego4d import Ego4DObjnessClsDataset
from egohos import EgoHOSObjnessClsDataset
from oxford_iiit_pet import PetSegmentDataset
from pascal_voc2012 import PascalVOC2012Dataset
from transform import ObjnessClsDataPreprocessor


class ObjnessClsDataModule(pl.LightningDataModule):
    """
    Fetches frames and labeled masks.
        ego4d: 
            Fetches total of 3 frames per state change: pre, pnr, and post frames. Labels are given by mask of background (0) and bounding box level state-changing object (1).
        egohos:
            Fetches frames from few egocentric dataset source. Labels are given by a mask of background (0), and pixel level foreground (any object that is interacted by the photographer) (1), and pixel level hands (2).

    Returns:
        [
            frames: [batch, frame_num, height, width, channel],
            labels: [batch, frame_num, height, width, channel],
            info: dict of additional information (optional)
        ]
    """

    def __init__(
        self,
        dataset_dir,
        dataset_mode='ego4d',  # ['ego4d', 'egohos']
        batch_size=4,
        transform_mode='base',  # ['base', 'display', 'aug[num]']
        with_info=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_dir = dataset_dir
        self.dataset_mode = dataset_mode
        self.batch_size = batch_size
        self.with_info = with_info

        self.num_workers=os.cpu_count()

        self.transform = ObjnessClsDataPreprocessor(
            transform_mode=transform_mode
        )

    def setup(self, stage=None):
        if self.dataset_mode == 'ego4d':
            if stage == "fit" or stage is None:
                self.train_data = Ego4DObjnessClsDataset(
                    dataset_dir=self.dataset_dir,
                    transform=self.transform,
                    with_info=self.with_info,
                )
                self.val_data = Ego4DObjnessClsDataset(
                    dataset_dir=self.dataset_dir,
                    phase='val',
                    transform=self.transform,
                    with_info=self.with_info,
                )

            if stage == "test":
                self.test_data = None

            if stage == "predict":
                self.predict_data = None

        elif self.dataset_mode == 'egohos':
            if stage == "fit" or stage is None:
                self.train_data = EgoHOSObjnessClsDataset(
                    dataset_dir=self.dataset_dir,
                    transform=self.transform,
                    with_info=self.with_info,
                )
                self.val_data = EgoHOSObjnessClsDataset(
                    dataset_dir=self.dataset_dir,
                    phase='val',
                    transform=self.transform,
                    with_info=self.with_info,
                )

            if stage == "test":
                self.test_data = [
                    EgoHOSObjnessClsDataset(
                        dataset_dir=self.dataset_dir,
                        phase='test_indomain',
                        transform=self.transform,
                        with_info=self.with_info,
                    ),
                    EgoHOSObjnessClsDataset(
                        dataset_dir=self.dataset_dir,
                        phase='test_outdomain',
                        transform=self.transform,
                        with_info=self.with_info,
                    )
                ]

            if stage == "predict":
                self.predict_data = None

        elif self.dataset_mode == 'pet':
            if stage == "fit" or stage is None:
                self.train_data = PetSegmentDataset(
                    dataset_dir=self.dataset_dir,
                    transform=self.transform,
                    with_info=self.with_info,
                )
                self.val_data = PetSegmentDataset(
                    dataset_dir=self.dataset_dir,
                    phase='valid',
                    transform=self.transform,
                    with_info=self.with_info,
                )

            if stage == "test":
                self.test_data = PetSegmentDataset(
                    dataset_dir=self.dataset_dir,
                    phase='test',
                    transform=self.transform,
                    with_info=self.with_info,
                )

            if stage == "predict":
                self.predict_data = None

        elif self.dataset_mode == 'voc2012':
            if stage == "fit" or stage is None:
                self.train_data = PascalVOC2012Dataset(
                    dataset_dir=self.dataset_dir,
                    transform=self.transform,
                    with_info=self.with_info,
                )
                self.val_data = None

            if stage == "test":
                self.test_data = None

            if stage == "predict":
                self.predict_data = None

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True
            )
            for dataset in self.test_data
        ]
