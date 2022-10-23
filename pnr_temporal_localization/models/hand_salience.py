import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mediapipe as mp


class System():
    def __init__(self):
        self.model = HandSalience()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        self.label_transform = IdentityTransform()


class HandSalience(nn.Module):
    def __init__(self):
        super().__init__()
        self.hand_saliency_map = AddHandMapTransform()
        resnet3d = torch.hub.load('facebookresearch/pytorchvideo',
                                  'slow_r50', pretrained=True)
        resnet3d_modules = nn.ModuleList([*list(resnet3d.blocks.children())])
        self.backbone = nn.Sequential(*resnet3d_modules[:-1])
        self.backbone[0].conv = nn.Conv3d(
                                    4, 64, kernel_size=(1,7,7), stride=(1,2,2),
                                    padding=(0,3,3), bias=False
                                )
        self.pool = nn.AvgPool3d(kernel_size=(1,7,7), stride=1, padding=0)
        self.drop = nn.Dropout(p=0.5, inplace=False)
        self.proj = nn.Linear(in_features=2048, out_features=1, bias=True)


    def forward(self, x):
        batch_size = x.shape[0]

        x = self.hand_saliency_map(x)
        x = self.backbone(x)

        x = self.pool(x)
        x = self.drop(x)
        x = x.permute((0, 2, 3, 4, 1))
        x = self.proj(x)
        x = x.permute((0, 4, 1, 2, 3))
        x = x.view(batch_size, -1)

        return x


class AddHandMapTransform():
    def __init__(self):
        self.image_mode = True
        self.hand_num = 2
        self.detection_conf = 0.7

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=self.image_mode,
            max_num_hands=self.hand_num,
            min_detection_confidence=self.detection_conf
        )

    def __call__(self, frames):
        batch_num, c, frame_num, h, w = frames.shape
        output = torch.empty((batch_num, c + 1, frame_num, h, w))

        for batch in range(batch_num):
            for i in range(frame_num):
                frame = frames[batch,:,i,:,:]
                saliency_map = self.hand_saliency_map(frame)
                output[batch,:,i,:,:] = torch.cat((frame, saliency_map), 0)

        return output

    def _twoD_gaussian(self, x, y, coord, sigma, amp):
        function = amp * np.exp(-0.5 * (((x - coord[0]) / sigma) ** 2 +
                                ((y - coord[1]) / sigma) ** 2))
        return function

    def _min_max(self, x):
        min = x.min(keepdims=True)
        max = x.max(keepdims=True)
        result = (x-min)/(max-min)
        return result

    def hand_saliency_map(self, frame):
        image = frame.permute((1,2,0)).detach().numpy().astype('uint8')

        h, w, _ = image.shape
        coords = []
        map = np.zeros_like(image[:,:,0], dtype='f8')
        x = np.arange(0, map.shape[1], 1)
        y = np.arange(0, map.shape[0], 1)
        X, Y = np.meshgrid(x, y)

        result = self.hands.process(image)

        if result.multi_hand_landmarks is None:
            return torch.from_numpy(map).unsqueeze(0)

        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                coords.append((landmark.x * w, landmark.y * h))

        for coord in coords:
            map += self._twoD_gaussian(X, Y, coord, 20, 1)
        map = self._min_max(map) * 255

        return torch.from_numpy(map).unsqueeze(0)


class IdentityTransform():
    def __init__(self):
        pass

    def __call__(self, batch):
        return batch[1]