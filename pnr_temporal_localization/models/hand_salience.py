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
        self.resnet3d = torch.hub.load('facebookresearch/pytorchvideo',
                                  'slow_r50', pretrained=True)

    def forward(self, x):
        x = self.hand_saliency_map(x)
        print(x)
        # x = self.resnet3d(x)

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
        images = frames.detach().numpy().transpose(0,2,3,4,1).astype('uint8')

        coords = self.detect_hand_coord(images)
        self.coords_to_map()

        return coords

    def detect_hand_coord(self, images):
        coords = []
        batch_size = images.shape[0]
        frame_num = images.shape[1]
        h, w = images.shape[2], images.shape[3]

        for batch in range(batch_size):
            for i in range(frame_num):
                results = self.hands.process(images[batch,:,i,:,:])

                if results.multi_hand_landmarks is None:
                    return coords

                else:
                    return None

    def coords_to_map(self):
        pass


class IdentityTransform():
    def __init__(self):
        pass

    def __call__(self, batch):
        return batch[1]