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
        default="vivit",
        choices=["resnet", "vivit"],
    )
    parser.add_argument('-l', '--log_dir', type=str, default='./logs/')
    parser.add_argument('-e', '--exp_dir', type=str, default='')

    return parser.parse_args()


def main():
    args = option_parser()

    set_seed()

    dataset = KeypointEstDataModule(
        dataset_dir=args.dataset_dir,
        dataset_mode='ego4d',
        batch_size=4,
        transform_mode='base',
        selection='segsec',
        sample_num=16,
        with_info=True,
    )

    module = importlib.import_module(f'models.{args.model}')
    classifier = module.System()

    # # apply trained weight to model
    # param = torch.load("./logs/2cls/unet.pth", map_location=torch.device('cpu'))#["state_dict"]
    # new_param = {}
    # for k in param.keys():
    #     # if "model.unet.encoder" in k:
    #     if "encoder" in k:
    #         new_param[k[8:]] = param[k]
    # classifier.model.load_state_dict(new_param, strict=False)

    log_id = datetime.datetime.now().isoformat(timespec='seconds')
    logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp_dir,
        version=log_id,
        default_hp_metric=False,
    )
    logger.log_hyperparams(dataset.hparams | classifier.hparams)

    trainer = L.Trainer(
        # fast_dev_run=True,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
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
