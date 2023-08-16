from torchvision import transforms


class KeypointEstDataPreprocessor:
    def __init__(self, transform_mode='base'):
        self.transform_mode = transform_mode

    def __call__(self, frame, label):
        frame = self._frame_transform(frame)
        label = self._label_transform(label)

        return frame, label

    def _frame_transform(self, frame):
        if self.transform_mode == 'display':
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        return transform(frame)

    def _label_transform(self, label):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        return transform(label)
