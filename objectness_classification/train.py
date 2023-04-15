import os
import sys
import argparse
import shutil
import importlib
import datetime
import git
import torch
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
    parser.add_argument('-l', '--log_dir', type=str, default='./logs/')
    parser.add_argument('-r', '--delete_log_dir', action='store_true')

    return parser.parse_args()


def main():
    args = option_parser()

    set_seed()

    dataset = ObjnessClsDataModule(
        dataset_dir=args.dataset_dir,
        dataset_mode='egohos',
        batch_size=16,
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

    if args.delete_log_dir:
        shutil.rmtree(args.log_dir)
    log_id = datetime.datetime.now().isoformat()

    logger = pl.loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        version=log_id,
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        # save_weights_only=True,
        monitor="train_loss",
        mode='min',
        dirpath=args.log_dir,
    )
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=1,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        classifier,
        datamodule=dataset,
        # ckpt_path=None
    )

    torch.save(
        classifier.model.state_dict(),
        f=os.path.join(
            args.log_dir,
            log_id,
            f"{args.model}.pth"
        )
    )


if __name__ == '__main__':
    main()
