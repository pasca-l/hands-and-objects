import os
import sys
import argparse
import importlib
import git
import pytorch_lightning as pl

git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
sys.path.append(f"{git_root}/objectness_classification/datasets")
from datamodule import ObjnessClsDataModule
from system import ObjnessClassifier
from seed import set_seed


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str,
                        default='/Users/shionyamadate/Documents/datasets')
    parser.add_argument('-m', '--model', type=str, default="unet",
                        choices=["unet"])
    parser.add_argument('-c', '--ckpt_path', type=str, default='./logs/lightning_logs/2023-04-17T01:03:27.275696/checkpoints/epoch=9-step=2820.ckpt')

    return parser.parse_args()


def main():
    args = option_parser()

    set_seed()

    dataset = ObjnessClsDataModule(
        dataset_dir=args.dataset_dir,
        dataset_mode='egohos',
        batch_size=32,
        with_transform=True,
    )

    module = importlib.import_module(f'models.{args.model}')
    system = module.System()
    classifier = ObjnessClassifier(
        model=system.model,
        loss=system.loss,
        optimizer=system.optimizer,
        metric=system.metric,
    )
    classifier.load_from_checkpoint(args.ckpt_path)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        version="2023-04-17T01:03:27.275696",
    )
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        logger=logger,
    )

    trainer.test(
        classifier,
        datamodule=dataset,
        # ckpt_path=None
    )


if __name__ == '__main__':
    main()
