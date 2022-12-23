import argparse
import importlib
import pytorch_lightning as pl
import cv2
import torch
import numpy as np
import torchvision
import torch.nn.functional as nnf
import matplotlib.pyplot as plt

from system import StateChgObjDetector


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default="fho_scod",
                        choices=["fho_scod"])
    parser.add_argument('-d', '--data_dir', type=str, 
                        default='/home/ubuntu/data/ego4d/')
    parser.add_argument('-a', '--ann_dir', type=str,
                        default='/home/ubuntu/data/ego4d/annotations/')
    parser.add_argument('-m', '--model', type=str, default="faster_rcnn",
                        choices=["faster_rcnn", "finetune_resnet"])

    return parser.parse_args()


def main():
    args = option_parser()

    module = importlib.import_module(f'models.{args.model}')
    system = module.System()
    detector = StateChgObjDetector(
        sys=system
    )

    video_path = "../test_data/d.mp4"
    video_save_path = "../test_data/result_d.mp4"

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # v_size = (v_width, v_height)
    v_size = (224,224)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_save_path, fourcc, fps, v_size)

    for _ in range(int(frame_count)):
        ret, frame = video.read()

        image = cv2.imread("../test_data/test.jpg")
        image = cv2.resize(image, (224,224))
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.astype(np.float32)).clone()
        img = img.unsqueeze(0).permute(0,3,1,2) # [b, c, h, w]

        model = detector.model
        model.eval()
        model.zero_grad()

        x = model.resnet.conv1(img)
        x = model.resnet.bn1(x)
        x = model.resnet.relu(x)
        x = model.resnet.maxpool(x)
        x = model.resnet.layer1(x)
        x = model.resnet.layer2(x)
        x = model.resnet.layer3(x)
        x = model.resnet.layer4(x)
        features = x.clone().detach().requires_grad_(True)

        x = model.resnet.avgpool(features)
        x = x.view(x.size(0), -1)
        output = model.resnet.fc(x)
        pred_idx = torch.argmax(output).item()

        output[0][pred_idx].backward()
        feature_vec = features.grad.view(512, 7*7)
        alpha = torch.mean(feature_vec, axis=1)

        loss = nnf.relu(torch.sum(features.squeeze(0) * alpha.view(-1,1,1), 0))
        loss = loss.detach().numpy()

        loss_min = np.min(loss)
        loss_max = np.max(loss)
        loss = (loss - loss_min)/(loss_max - loss_min)

        loss = cv2.resize(loss, (224, 224))
        loss = (loss*255).astype(np.uint8)

        heatmap = cv2.applyColorMap(loss, cv2.COLORMAP_JET)
        output = heatmap * 0.5 + image * 0.5
        cv2.imwrite("../test_data/heatmap.png", output)
        break

        writer.write(output.astype('uint8'))

    writer.release()
    video.release()


if __name__ == '__main__':
    main()
