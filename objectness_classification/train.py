import os
import sys
import argparse
import shutil
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
    parser.add_argument('-l', '--log_save_dir', type=str, default='./logs/')
    parser.add_argument('-r', '--delete_log_dir', action='store_true')

    return parser.parse_args()


def main():
    args = option_parser()

    if args.delete_log_dir:
        shutil.rmtree(args.log_save_dir)

    set_seed()

    dataset = ObjnessClsDataModule(
        dataset_dir=args.dataset_dir,
        dataset_mode='pet',
        batch_size=16,
        with_transform=True,
    )

    module = importlib.import_module(f'models.{args.model}')
    system = module.System()
    classifier = ObjnessClassifier(
        sys=system
    )

    logger = pl.loggers.TensorBoardLogger(
        save_dir=args.log_save_dir
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        # save_weights_only=True,
        monitor="train_loss",
        mode='min',
        dirpath=args.log_save_dir,
        filename=args.model
    )
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        auto_select_gpus=True,
        max_epochs=5,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        classifier,
        datamodule=dataset,
        # ckpt_path=None
    )


if __name__ == '__main__':
    main()
