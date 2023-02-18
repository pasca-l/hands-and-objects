import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ego4d import Ego4DObjnessClsDataset
from egohos import EgoHOSObjnessClsDataset
from transform import ObjnessClsDataPreprocessor


class ObjnessClsDataModule(pl.LightningDataModule):
    """
    Fetches total of 3 frames per state change: pre, pnr, and post frames.
    Labels are given by a mask of background (0), and foreground (1).

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
        with_transform=True,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_mode = dataset_mode
        self.batch_size = batch_size

        if with_transform:
            self.transform = ObjnessClsDataPreprocessor()
        else:
            self.transform = None

    def setup(self, stage=None):
        if self.dataset_mode == 'ego4d':
            if stage == "fit" or stage is None:
                self.train_data = Ego4DObjnessClsDataset(
                    dataset_dir=self.dataset_dir,
                    transform=self.transform,
                )
                self.val_data = Ego4DObjnessClsDataset(
                    dataset_dir=self.dataset_dir,
                    phase='val',
                    transform=self.transform,
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
                )
                self.val_data = EgoHOSObjnessClsDataset(
                    dataset_dir=self.dataset_dir,
                    phase='val',
                    transform=self.transform,
                )

            if stage == "test":
                self.test_data = None

            if stage == "predict":
                self.predict_data = None

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True
        )
