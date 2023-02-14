import torch
from torchvision import transforms


class ObjnessClsDataPreprocessor():
    def __init__(
        self,
    ):
        pass

    def __call__(self):
        return self._simple_transform()

    def _simple_transform(self):
        transform = transforms.Compose([
            transforms.Lambda(
                lambda x: torch.as_tensor(x, dtype=torch.float)
            ),
            transforms.Normalize([0.45], [0.225])
        ])

        return transform
