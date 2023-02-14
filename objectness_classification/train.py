import sys
import argparse
import shutil
import importlib
import pytorch_lightning as pl

sys.path.append("./datasets")
from datasets.datamodule import ObjnessClsDataModule
from system import ObjnessClassifier

sys.path.append("../utils")
from json_handler import JsonHandler
from video_extractor import Extractor
from checker import Checker


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str,
                        default='/home/aolab/data/')
    parser.add_argument('-m', '--model', type=str, default="unet",
                        choices=["unet"])
    parser.add_argument('-l', '--log_save_dir', type=str, default='./logs/')
    parser.add_argument('-r', '--delete_log_dir', action='store_true')

    return parser.parse_args()


def main():
    args = option_parser()

    if args.delete_log_dir:
        shutil.rmtree(args.log_save_dir)

    dataset = ObjnessClsDataModule(
        dataset_dir=args.dataset_dir,
    )

    module = importlib.import_module(f'models.{args.model}')
    system = module.System()
    classifier = ObjnessClassifier(
        sys=system
    )

    checker = Checker(
        ObjnessClsDataModule(
            dataset_dir=args.dataset_dir,
        ),
        system.model,
    )
    checker.check_dataset()
    return

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
        max_epochs=10,
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
