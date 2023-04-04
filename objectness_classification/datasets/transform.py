import torch
from torchvision import transforms


class ObjnessClsDataPreprocessor:
    def __call__(self, x):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
#     transform = transforms.Compose([
#         transforms.Lambda(
#             lambda x: torch.as_tensor(x, dtype=torch.float)
#         ),
#         transforms.Normalize([0.45], [0.225])
#     ])

        return transform(x)
