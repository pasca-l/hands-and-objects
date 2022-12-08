import sys
import argparse
import shutil
import importlib
import pytorch_lightning as pl

from dataset_module import StateChgObjDataModule
from system import StateChgObjDetector

sys.path.append("../utils")
from json_handler import JsonHandler
from video_extractor import Extractor


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default="fho_scod",
                        choices=["fho_scod"])
    parser.add_argument('-d', '--data_dir', type=str, 
                        default='/home/ubuntu/data/ego4d/')
    parser.add_argument('-a', '--ann_dir', type=str,
                        default='/home/ubuntu/data/ego4d/annotations/')
    parser.add_argument('-m', '--model', type=str, default="faster_rcnn",
                        choices=["faster_rcnn"])
    parser.add_argument('-l', '--log_save_dir', type=str, default='./logs/')
    parser.add_argument('-r', '--delete_log_dir', action='store_true')
    parser.add_argument('-e', '--extract_frame', action='store_true')

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
    }

    if args.extract_frame:
        extractor = Extractor(args.task, args.data_dir)
        for flatten_json in json_dict.values():
            extractor.extract_frame_as_image(flatten_json)

    dataset = StateChgObjDataModule(
        batch_size=1,
        data_dir=args.data_dir,
        json_dict=json_dict,
        model_name=args.model,
        label_mode='corners'    # 'corners' or 'COCO'
    )

    dataset.setup()
    data = iter(dataset.train_dataloader()).next()

    module = importlib.import_module(f'models.{args.model}')
    system = module.System()
    detector = StateChgObjDetector(
        sys=system
    )

    print(detector(data[0][0]))
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
        detector,
        datamodule=dataset,
        # ckpt_path=None
    )


if __name__ == '__main__':
    main()
