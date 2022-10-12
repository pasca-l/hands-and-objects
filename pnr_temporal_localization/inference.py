import sys
import argparse
import importlib
import numpy as np
import cv2
import torch

from dataset_module import FrameTransform
from system import PNRLocalizer

sys.path.append("../utils")
from json_handler import JsonHandler


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_save_dir', type=str, default='./logs/')
    parser.add_argument('--model', type=str, default="cnnlstm",
                        choices=["cnnlstm", "slowfastperceiver", "bmn", 
                                 "i3d_resnet", "hand_salience"])

    return parser.parse_args()


def main():
    args = option_parser()

    transform = FrameTransform()

    video_path = "~/data/ego4d/clips/de4a85e5-1809-4547-ae11-9161aa9fdbe9.mp4"
    frames = sample_video_frames(video_path)
    frames = torch.as_tensor(frames, dtype=torch.float).permute(3, 0, 1, 2)
    frames = transform(frames)

    module = importlib.import_module(f'models.{args.model}')
    system = module.System()
    classifier = PNRLocalizer(
        sys=system
    )

    checkpoint = torch.load(f'{args.log_save_dir}{args.model}.ckpt')
    classifier.load_state_dict(checkpoint['state_dict'])
    classifier.eval()

    with torch.no_grad():
        pred = classifier(frames)


def sample_video_frames(video_path, to_total_frames=32):
    video = cv2.VideoCapture(video_path)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    sample_rate = total_frames // to_total_frames
    frame_nums = [i for i in range(total_frames) if i % sample_rate == 0]

    frames = []
    counter = 1

    while True:
        ret, frame = video.read()
        if ret == False:
            break
        if counter in sorted(frame_nums):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frame = np.expand_dims(frame, axis=0).astype(np.float32)
            frames.append(frame)

        counter += 1

    video.release()

    return frames


if __name__ == '__main__':
    main()
