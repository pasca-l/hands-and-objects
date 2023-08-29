import torch
from torchvision import transforms


class KeypointEstDataPreprocessor:
    def __init__(self, transform_mode="base", label_mode="classes"):
        self.transform_mode = transform_mode
        self.label_mode = label_mode

    def __call__(self, frame, label):
        frame = self._frame_transform(frame)
        label = self._label_transform(label)

        return frame, label

    def _frame_transform(self, frame):
        if self.transform_mode == "base":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        elif self.transform_mode == "display":
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        else:
            transform = lambda x: torch.tensor(x)

        return transform(frame)

    def _label_transform(self, label):
        if self.label_mode == "image":
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        else:
            transform = lambda x: torch.tensor(x)

        return transform(label)
