import os
import sys
import argparse
import importlib
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
        "-d", "--dataset_dir",type=str,
        default=os.path.join(
            os.path.expanduser("~"),
            "Documents/datasets",
        ),
    )
    parser.add_argument(
        "-w", "--weight_path", type=str,
        default="./logs/2023-10-01T19:46:01/videomae.pth"
    )
    parser.add_argument("-l", "--log_dir", type=str, default="./logs/")
    parser.add_argument("-e", "--exp_dir", type=str, default="")

    return parser.parse_args()


def main():
    args = option_parser()

    set_seed()

    dataset = KeypointEstDataModule(
        dataset_dir=args.dataset_dir,
        dataset_mode="ego4d",
        batch_size=4,
        transform_mode="base",
        selection="segsec",
        sample_num=16,
        with_info=True,
    )

    model_name = os.path.splitext(os.path.basename(args.weight_path))[0]
    module = importlib.import_module(f"models.{model_name}")
    classifier = module.System()
    classifier.model.load_state_dict(
        torch.load(
            args.weight_path,
            # map_location=torch.device("cpu"),
        )
    )

    logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp_dir,
        version=args.weight_path.split("/")[-2],
    )

    # trainer = L.Trainer(
    #     logger=logger,
    #     devices=[0],
    #     num_nodes=1,
    # )
    # trainer.test(
    #     classifier,
    #     datamodule=dataset,
    # )

    # dataset.setup(stage="test")
    # test_dataloader = dataset.test_dataloader()
    # classifier.eval()
    # for i, (frames, labels, info) in enumerate(test_dataloader):
    #     input = frames.float()
    #     with torch.no_grad():
    #         out = classifier(input)


if __name__ == "__main__":
    main()
