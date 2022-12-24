import sys
import argparse
import shutil
import importlib
import pytorch_lightning as pl

from dataset_module import ObjnessClsDataModule
from system import ObjnessClassifier

sys.path.append("../utils")
from json_handler import JsonHandler
from video_extractor import Extractor
from check_dataset import DatasetChecker


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default="fho_scod",
                        choices=["fho_scod"])
    parser.add_argument('-d', '--data_dir', type=str, 
                        default='/home/aolab/data/ego4d/')
    parser.add_argument('-m', '--model', type=str, default="unet",
                        choices=["unet"])
    parser.add_argument('-l', '--log_save_dir', type=str, default='./logs/')
    parser.add_argument('-r', '--delete_log_dir', action='store_true')
    parser.add_argument('-e', '--extract_frame', action='store_true')

    return parser.parse_args()


def main():
    args = option_parser()

    if args.delete_log_dir:
        shutil.rmtree(args.log_save_dir)

    json_handler = JsonHandler(args.task)
    json_partial_name = f"{args.data_dir}annotations/{args.task}"
    json_dict = {
        "train": json_handler(f"{json_partial_name}_train.json"),
        "val": json_handler(f"{json_partial_name}_val.json"),
    }

    if args.extract_frame:
        extractor = Extractor(args.task, args.data_dir)
        for flatten_json in json_dict.values():
            extractor.extract_frame_as_image(flatten_json)

    dataset = ObjnessClsDataModule(
        data_dir=f"{args.data_dir}frames/",
        json_dict=json_dict,
        model_name=args.model,
        batch_size=1,
        label_mode='corners'    # 'corners' or 'COCO'
    )

    # checker = DatasetChecker(dataset)
    # checker()

    module = importlib.import_module(f'models.{args.model}')
    system = module.System()
    classifier = ObjnessClassifier(
        sys=system
    )

    print(classifier.model)
    # print(detector(data[0][0]))
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
