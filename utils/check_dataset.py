import cv2
import numpy as np


class DatasetChecker():
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self):
        self.dataset.setup()
        data = next(iter(self.dataset.train_dataloader()))

        img = data[0][0][0].numpy()
        mask = data[1][0][0].numpy()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('image.jpg', img)
        cv2.imwrite('mask.jpg', mask * 255)
        cv2.imwrite('attention.jpg', img * np.stack([mask, mask, mask], axis=2))
