from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import ObjnessClsDataset


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
        data_dir,
        json_dict,
        transform=None,
        batch_size=4,
        label_mode='corners', # ['corners', 'COCO']
        with_info=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.json_dict = json_dict
        self.transform = transform
        self.batch_size = batch_size
        self.label_mode = label_mode
        self.with_info = with_info

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = ObjnessClsDataset(
                data_dir=self.data_dir,
                flatten_json=self.json_dict['train'],
                transform=self.transform,
                label_mode=self.label_mode,
                with_info=self.with_info,
            )
            self.val_data = ObjnessClsDataset(
                data_dir=self.data_dir,
                flatten_json=self.json_dict['val'],
                transform=self.transform,
                label_mode=self.label_mode,
                with_info=self.with_info,
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
