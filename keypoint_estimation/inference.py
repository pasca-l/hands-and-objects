import os
import argparse
import torch
import lightning as L

from datamodule import KeypointEstDataModule
from system import KeypointEstModule


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset_dir",type=str,
        default=os.path.join(
            os.path.expanduser("~"),
            "Documents/datasets",
        ),
    )
    parser.add_argument(
        "-m", "--model", type=str,
        default="vivit",
        choices=["resnet", "vivit"],
    )
    parser.add_argument("-w", "--weight_path", type=str, required=True)
    parser.add_argument("-l", "--log_dir", type=str, default="./logs/")
    parser.add_argument("-e", "--exp_dir", type=str, default="inference")

    return parser.parse_args()


def main():
    args = option_parser()

    L.seed_everything(42, workers=True)

    dataset = KeypointEstDataModule(
        dataset_dir=args.dataset_dir,
        dataset_mode="ego4d",
        batch_size=4,
        transform_mode="base",
        selection="segsec",
        sample_num=16,
        seg_arg=8,
        with_info=False,
        neg_ratio=None,
    )

    classifier = KeypointEstModule(
        model_name=args.model,
        weight_path=args.weight_path,
    )

    logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp_dir,
    )
    logger.log_hyperparams(dataset.hparams | classifier.hparams)

    trainer = L.Trainer(
        deterministic=True,
        devices="auto",
        num_nodes=1,
        logger=logger,
    )
    trainer.test(
        classifier,
        datamodule=dataset,
    )


if __name__ == "__main__":
    main()
