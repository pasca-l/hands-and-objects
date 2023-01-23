import os
import cv2
import numpy as np
import torch
from tqdm import tqdm


class Checker():
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

        self.dataset.setup()
        self.train_dataloader = iter(self.dataset.train_dataloader())
        self.val_dataloader = iter(self.dataset.val_dataloader())

    def check_dataset(self):
        # clip_uid = "f36f1ca0-3567-4ef5-99b5-bc0e436c9217" # drawing
        # clip_uid = "e8e54409-495b-49e5-b325-556c812d6ff4" # dish washing
        clip_uid = "3dead068-318a-49c5-8036-e176e50cbf50"
        for data in tqdm(
            self.train_dataloader,
            desc=f'Checking data: {clip_uid}'
        ):
            if data[2]['clip_uid'][0] == clip_uid:
                pre_frame = int(data[2]['pre_frame_num_clip'])
                pnr_frame = int(data[2]['pnr_frame_num_clip'])
                post_frame = int(data[2]['post_frame_num_clip'])
                
                img_pre = data[0][0][0].numpy()
                img_pnr = data[0][0][1].numpy()
                img_post = data[0][0][2].numpy()
                mask_pnr = data[1][0][1].numpy()

                img_pre = cv2.cvtColor(img_pre, cv2.COLOR_RGB2BGR)
                img_pnr = cv2.cvtColor(img_pnr, cv2.COLOR_RGB2BGR)
                img_post = cv2.cvtColor(img_post, cv2.COLOR_RGB2BGR)

                test_img_dir = f"./test_data/{data[2]['clip_uid'][0]}/"
                os.makedirs(test_img_dir, exist_ok=True)
                # cv2.imwrite(f'{test_img_dir}pre_{pre_frame}.jpg', img_pre)
                cv2.imwrite(f'{test_img_dir}pnr_{pnr_frame}.jpg', img_pnr)
                # cv2.imwrite(f'{test_img_dir}post_{pos_frame}.jpg', img_post)
                cv2.imwrite(f'{test_img_dir}mask_{pnr_frame}.jpg', mask_pnr * 255)
                # cv2.imwrite(f'{test_img_dir}attention_{pnr_frame}.jpg', img_pnr * np.stack([mask_pnr, mask_pnr, mask_pnr], axis=2))

    def check_model(self):
        img = self.single_train_data[0]
        print(f"Train data shape: {img.shape}")
        train_output = self.model(img)
        processed_output = torch.argmax(torch.sigmoid(train_output[0]), axis=0)
        print(f"Train output: {processed_output}, with shape {processed_output.shape}")
        print(processed_output.unsqueeze(-1).shape)
        cv2.imwrite('test.jpg', processed_output.unsqueeze(-1).numpy())

        # print(f"Val data shape: {self.single_val_data.shape}")
        # val_output = self.model(self.single_val_data)
        # print(f"Val output: {val_output}, with shape {val_output.shape}")
