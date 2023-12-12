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
        "-d", "--dataset_dir", type=str,
        default=os.path.join(
            os.path.expanduser("~"),
            "Documents/datasets",
        ),
    )
    parser.add_argument(
        "-m", "--model", type=str,
        default="vivit",
        choices=["vivit", "baseline", "i3dresnet"],
    )
    parser.add_argument("--log_dir", type=str, default="./logs/")
    parser.add_argument("--exp_dir", type=str, default="")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpus", nargs="*", type=int)
    parser.add_argument("--debug", action='store_true')

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
        fast_load=True,
    )

    classifier = KeypointEstModule(
        model_name=args.model,
        epochs=args.epochs,
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
        fast_dev_run=args.debug,
        accelerator="auto",
        devices="auto" if args.gpus is None else args.gpus,
        max_epochs=args.epochs,
        # limit_train_batches=0.01,
        # limit_val_batches=0.01,
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


if __name__ == "__main__":
    main()
