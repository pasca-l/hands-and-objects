import os
import argparse
import datetime
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
    parser.add_argument("-l", "--log_dir", type=str, default="./logs/")
    parser.add_argument("-e", "--exp_dir", type=str, default="")

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
        with_info=True,
    )

    classifier = KeypointEstModule(
        model_name=args.model,
    )

    log_id = datetime.datetime.now().isoformat(timespec="seconds")
    logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp_dir,
        version=log_id,
    )
    logger.log_hyperparams(dataset.hparams | classifier.hparams)

    trainer = L.Trainer(
        deterministic=True,
        # fast_dev_run=True,
        # limit_train_batches=0.01,
        # limit_val_batches=0.01,
        accelerator="auto",
        devices="auto",
        max_epochs=10,
        logger=logger,
    )

    trainer.fit(
        classifier,
        datamodule=dataset,
    )
    torch.save(
        classifier.model.state_dict(),
        f=os.path.join(args.log_dir, args.exp_dir, log_id, f"{args.model}.pth"),
    )

    trainer.test(
        classifier,
        datamodule=dataset,
    )


if __name__ == "__main__":
    main()
