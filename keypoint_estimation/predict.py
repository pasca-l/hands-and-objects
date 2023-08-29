import os
import sys
import argparse
import importlib
import datetime
import git
import torch
import lightning as L

import numpy as np
import cv2
import polars as pl
from tqdm import tqdm


git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
sys.path.append(f"{git_root}/keypoint_estimation/datasets")
from datamodule import KeypointEstDataModule
from transform import KeypointEstDataPreprocessor
sys.path.append(f"{git_root}/utils/datasets")
from seed import set_seed


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset_dir',type=str,
        default=os.path.join(
            os.path.expanduser('~'),
            'Documents/datasets',
        ),
    )
    parser.add_argument(
        '-m', '--model', type=str,
        default="resnet",
        choices=["resnet"],
    )
    parser.add_argument('-l', '--log_dir', type=str, default='./logs/')
    parser.add_argument('-e', '--exp_dir', type=str, default='')

    return parser.parse_args()


def get_model(model_name, weight_path, model_args=None):
    module = importlib.import_module(f'models.{model_name}')
    model = module.System(
        # out_channels=model_args['out_channels'],
        # mode=model_args['mode'],
    )

    model.model.load_state_dict(
        torch.load(
            weight_path,
            map_location=torch.device('cpu'),
        ),
        # strict=False,
    )

    model.eval()

    return model


def get_model_prediction(model, input):
    # model.eval()
    with torch.no_grad():
        out = model(input)

    return out.sigmoid().detach().numpy()[0]


def main():
    args = option_parser()

    set_seed()

    dataset = KeypointEstDataModule(
        dataset_dir=args.dataset_dir,
        dataset_mode='ego4d',
    )
    dataset.setup(stage="test")
    test_data = dataset.test_data

    transform = KeypointEstDataPreprocessor()

    # module = importlib.import_module(f'models.{args.model}')
    # classifier = module.System()

    classifier = get_model(
        model_name=args.model,
        weight_path="./logs/resnet101wobj-epoch10/resnet.pth",
    )

    total_average_distance = []
    average_accuracy = []

    for (vid,) in tqdm(test_data.ann_df.select("video_uid").unique(maintain_order=True).iter_rows()):
        print(vid)
        video_path = os.path.join(args.dataset_dir, f"ego4d/v2/full_scale/{vid}.mp4")
        if not os.path.exists(video_path):
            raise Exception(f"Video does not exist at: {video_path}")
        video = cv2.VideoCapture(video_path)

        outs, labels = [], []

        counter = 0
        while True:
            ret, frame = video.read()
            if ret == False:
                break

            frame = cv2.resize(frame, (224,224))
            frame, _ = transform(frame, np.array([0]))
            out = get_model_prediction(classifier, frame.unsqueeze(0))

            label = test_data.ann_df.filter(
                (pl.col("video_uid") == vid) & (pl.col("center") == counter)
            ).shape[0]

            outs.append(out.argmax())
            labels.append(label)

            counter += 1

        result = np.zeros_like(np.array(labels))
        label_indices = np.where(np.array(labels) == 1)[0]
        output_indices = np.where(np.array(outs) == 1)[0]

        for idx in output_indices:
            if len(label_indices) > 0:
                nearest_idx = label_indices[np.argmin(np.abs(label_indices - idx))]
                if nearest_idx == idx:
                    result[idx] = 0
                else:
                    result[idx] = abs(nearest_idx - idx)

        average_distance = np.average(result[output_indices])
        total_average_distance.append(average_distance)
        accuracy = 1 - np.average(np.array(outs).astype(np.uint8) ^ np.array(labels).astype(np.uint8))
        average_accuracy.append(accuracy)

        np.savetxt(f"./results/resnet101obj/{vid}_out.txt", np.array(outs))
        np.savetxt(f"./results/resnet101obj/{vid}_label.txt", np.array(labels))

    print(f"Average distance: {np.average(total_average_distance)}", f"Average accuracy: {np.average(average_accuracy)}")

    return

    log_id = datetime.datetime.now().isoformat(timespec='seconds')
    logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp_dir,
        version=log_id + "_predict",
        default_hp_metric=False,
    )
    logger.log_hyperparams(dataset.hparams | classifier.hparams)

    trainer = L.Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=10,
        logger=logger,
    )

    trainer.test(
        classifier,
        datamodule=dataset,
    )


if __name__ == '__main__':
    main()
