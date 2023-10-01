import os
import sys
import argparse
import importlib
import git
import torch

git_root = git.Repo(os.getcwd(), search_parent_directories=True
                    ).git.rev_parse("--show-toplevel")
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
        '-w', '--weight_path', type=str,
        default="./logs/2023-10-01T19:46:01/videomae.pth"
    )

    return parser.parse_args()


class Predictor:
    def __init__(self, model_name, weight_path, model_args):
        self.model_name = model_name
        self.weight_path = weight_path
        self.model_args = model_args
        self.model = self.get_model()

    def _get_model(self):
        module = importlib.import_module(f"models.{self.model_name}")
        model = module.System(
            **self.model_args,
        )

        model.model.load_state_dict(
            torch.load(
                self.weight_path,
                # map_location=torch.device('cpu'),
            ),
        )
        return model

    def get_model_prediction(self, input):
        self.model.eval()
        with torch.no_grad():
            out = self.model(input)

        return out


def main():
    args = option_parser()

    set_seed()

    dataset = KeypointEstDataModule(
        dataset_dir=args.dataset_dir,
        dataset_mode='ego4d',
        batch_size=1,
        transform_mode='display',
        with_info=True,
    )
    dataset.setup(stage='test')
    test_dataloader = dataset.test_dataloader()

    model = Predictor(
        model_name=os.path.splitext(os.path.basename(args.weight_path))[0],
        weight_path=args.weight_path,
        model_args={

        },
    )

    for i, (frames, labels, info) in enumerate(test_dataloader):
        input = frames.float()
        out = model.get_model_prediction(input)


if __name__ == "__main__":
    main()
