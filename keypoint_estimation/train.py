import os
import sys
import argparse
import importlib
import datetime
import git
import torch
import lightning as L

git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
sys.path.append(f"{git_root}/keypoint_estimation/datasets")
from datamodule import KeypointEstDataModule
from ego4d import Ego4DKeypointEstDataset
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


def main():
    args = option_parser()

    set_seed()

    dataset = Ego4DKeypointEstDataset(
        args.dataset_dir,
        extract=True,
    )

    print(dataset.ann_df.select("video_uid").unique()[0].item())

    return

    dataset = KeypointEstDataModule(
        dataset_dir=args.dataset_dir,
        dataset_mode='ego4d',
        batch_size=4,
        transform_mode='base',
    )

    module = importlib.import_module(f'models.{args.model}')
    classifier = module.System()

    log_id = datetime.datetime.now().isoformat(timespec='seconds')
    logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp_dir,
        version=log_id,
        default_hp_metric=False,
    )
    logger.log_hyperparams(dataset.hparams | classifier.hparams)

    trainer = L.Trainer(
        accelerator='auto',
        devices='auto',
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


if __name__ == '__main__':
    main()
