import json
import sys
import argparse
import shutil
import importlib
import torch
import pytorch_lightning as pl

from dataset_module import PNRTempLocDataModule
from system import PNRLocalizer

sys.path.append("../utils")
from json_handler import JsonHandler
from video_extractor import Extractor


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--task', type=str, default="fho_hands",
                        choices=["fho_hands"])
    parser.add_argument('--log_save_dir', type=str, default='./logs/')
    parser.add_argument('--delete_log_dir', action='store_true')
    parser.add_argument('--ann_dir', type=str,
                        default='../../../data/ego4d/annotations/')
    parser.add_argument('--data_dir', type=str, 
                        default='../../../data/ego4d/')
    parser.add_argument('--model', type=str, default="cnnlstm",
                        choices=["cnnlstm", "slowfastperceiver", "bmn", 
                                 "i3d_resnet", "hand_salience"])

    return parser.parse_args()


def main():
    args = option_parser()

    if args.delete_log_dir:
        shutil.rmtree(args.log_save_dir)

    json_handler = JsonHandler(args.task)
    json_partial_name = f"{args.ann_dir}{args.task}"
    json_dict = {
        "train": json_handler(f"{json_partial_name}_train.json"),
        "val": json_handler(f"{json_partial_name}_val.json"),
        # "test": json_handler(f"{json_partial_name}_test_unannotated.json")
    }

    dataset = PNRTempLocDataModule(
        batch_size=4,
        data_dir=args.data_dir,
        json_dict=json_dict
    )

    module = importlib.import_module(f'models.{args.model}')
    system = module.System()
    classifier = PNRLocalizer(
        sys=system
    )

    logger = pl.loggers.TensorBoardLogger(
        save_dir=args.log_save_dir
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        save_weights_only=True,
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

    trainer.fit(classifier, dataset)

    # save pth file
    save_name = f'{args.log_save_dir}{args.model}'
    model = system.model
    checkpoint = torch.load(f'{save_name}.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    torch.save(model.state_dict(), f'{save_name}.pth')


if __name__ == '__main__':
    main()
