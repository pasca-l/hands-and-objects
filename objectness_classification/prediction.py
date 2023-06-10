import os
import sys
import git
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

import segmentation_models_pytorch as smp

git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")

sys.path.append(f"{git_root}/objectness_classification/")
from seed import set_seed
sys.path.append(f"{git_root}/objectness_classification/datasets")
from datamodule import ObjnessClsDataModule


def get_prediction(model, weight_path, input):
    model.load_state_dict(
        torch.load(weight_path, map_location=torch.device('cpu')),
        # strict=False,
    )
    model.eval()

    with torch.no_grad():
        out = model(input)

    return out.sigmoid().detach().numpy()[0]


def main():
    set_seed()

    home = os.path.expanduser("~")
    dataset_dir = os.path.join(home, "Documents/datasets")

    dataset = ObjnessClsDataModule(
        dataset_dir=dataset_dir,
        dataset_mode='egohos',
        batch_size=1,
        with_transform=False,
        with_info=True,
    )
    dataset.setup(stage="test")
    dataloader = iter(dataset.test_dataloader()[0])

    model1 = smp.Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        # activation="sigmoid",
    )
    model2 = smp.Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        # activation="sigmoid",
    )

    model1_path = os.path.join(git_root, "objectness_classification/logs/lightning_logs/2023-04-18T01:10:25.004901/unet.pth")
    # model2_path = os.path.join(git_root, "objectness_classification/logs/lightning_logs/2023-04-18T01:47:53.652015/unet.pth")
    model2_path = os.path.join(git_root, "/Users/shionyamadate/Downloads/2cls/unet.pth")
    model3_path = os.path.join(git_root, "/Users/shionyamadate/Downloads/2cls_centercrop/unet.pth")
    model4_path = os.path.join(git_root, "/Users/shionyamadate/Downloads/2cls_randomaffine(translate=0.9,scale=1-1.2)/unet.pth")
    model5_path = os.path.join(git_root, "/Users/shionyamadate/Downloads/2cls_randomresizedcrop/unet.pth")

    threshold = 0.5

    for i, (frames, labels, info) in enumerate(dataloader):
        # if i not in [151]:#, 4, 35, 134, 414, 416]:
        #     print(i)
        #     continue

        img = frames.permute(0,2,3,1)[0]
        mask = labels[0]

        input = frames.float()

        out1 = get_prediction(model1, model1_path, input)
        out2 = get_prediction(model2, model2_path, input)
        out3 = get_prediction(model2, model3_path, input)
        out4 = get_prediction(model2, model4_path, input)
        out5 = get_prediction(model2, model5_path, input)

        fig = plt.figure()
        fig.add_subplot(3, 3, 1)
        plt.imshow(img)
        fig.add_subplot(3, 3, 2)
        plt.imshow(mask[0])
        plt.gray()
        fig.add_subplot(3, 3, 3)
        plt.imshow(mask[1])
        plt.gray()

        # fig.add_subplot(3, 3, 4)
        # heatmap1 = cv2.applyColorMap(
        #     (out1[0] * 255).astype(np.uint8),
        #     cv2.COLORMAP_JET,
        # )
        # heatmap1 = cv2.cvtColor(heatmap1, cv2.COLOR_BGR2RGB)
        # plt.imshow(heatmap1)

        # fig.add_subplot(3, 3, 5)
        # heatmap2 = cv2.applyColorMap(
        #     (out2[0] * 255).astype(np.uint8),
        #     cv2.COLORMAP_JET,
        # )
        # heatmap2 = cv2.cvtColor(heatmap2, cv2.COLOR_BGR2RGB)
        # plt.imshow(heatmap2)

        # fig.add_subplot(3, 3, 6)
        # heatmap3 = cv2.applyColorMap(
        #     (out2[1] * 255).astype(np.uint8),
        #     cv2.COLORMAP_JET,
        # )
        # heatmap3 = cv2.cvtColor(heatmap3, cv2.COLOR_BGR2RGB)
        # plt.imshow(heatmap3)

        fig.add_subplot(3, 3, 4)
        binary1 = out2[0] > threshold
        plt.imshow(binary1)
        plt.gray()

        fig.add_subplot(3, 3, 5)
        binary2 = out3[0] > threshold
        plt.imshow(binary2)
        plt.gray()

        fig.add_subplot(3, 3, 7)
        binary3 = out4[0] > threshold
        plt.imshow(binary3)
        plt.gray()

        fig.add_subplot(3, 3, 8)
        binary4 = out5[0] > threshold
        plt.imshow(binary4)
        plt.gray()

        # plt.show()
        dir = "./images_test"
        plt.savefig(f'{dir}/{i}.png')

        # plt.imsave(f'{dir}/{i}_img.png', img.detach().numpy())
        # plt.imsave(f'{dir}/{i}_mask1.png', mask[0].detach().numpy())
        # plt.imsave(f'{dir}/{i}_mask2.png', mask[1].detach().numpy())
        # plt.imsave(f'{dir}/{i}_binary1.png', binary1)
        # plt.imsave(f'{dir}/{i}_binary2.png', binary2)
        # plt.imsave(f'{dir}/{i}_binary3.png', binary3)
        # plt.imsave(f'{dir}/{i}_heatmap1.png', heatmap1)
        # plt.imsave(f'{dir}/{i}_heatmap2.png', heatmap2)
        # plt.imsave(f'{dir}/{i}_heatmap3.png', heatmap3)

        plt.clf()
        plt.close()


if __name__ == "__main__":
    main()