import os
from torch.utils.data import Dataset

from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset


class PetSegmentDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        phase='train',
        transform=None,
        with_info=False,
    ):
        super().__init__()

        self.dataset = SimpleOxfordPetDataset(
            root=os.path.join(dataset_dir, "oxford_iiit_pet"),
            mode=phase,
        )
        self.transform = transform
        self.with_info = with_info

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_dict = self.dataset[index]
        image = data_dict['image']
        mask = data_dict['mask']
        trimap = data_dict['trimap']

        if self.transform != None:
            image = self.transform(image)

        if self.with_info:
            return image, mask, trimap

        return image, mask
