import os
import sys
import argparse
import importlib
import git
import matplotlib.pyplot as plt
import torch

git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
sys.path.append(f"{git_root}/objectness_classification/datasets")
from datamodule import ObjnessClsDataModule
from seed import set_seed


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset_dir', type=str,
        default=os.path.join(
            os.path.expanduser('~'),
            'Documents/datasets/egohos_ca'
        ),
    )
    parser.add_argument(
        '-m', '--model', type=str,
        default="unet",
        choices=["unet"],
    )
    parser.add_argument(
        '-w', '--model_weight', type=str,
        default="./logs/2023-06-25T19:33:46/unet.pth",
    )

    return parser.parse_args()


def get_test_dataloader(dataset_dir):
    dataset = ObjnessClsDataModule(
        dataset_dir=dataset_dir,
        dataset_mode='egohos',
        batch_size=1,
        transform_mode='display',
        with_info=True,
    )
    dataset.setup(stage='test')
    test_dataloader = dataset.test_dataloader()

    if len(test_dataloader) > 1:
        return iter(test_dataloader[0])
    else:
        return iter(test_dataloader)


def get_model(model_name, weight_path, model_args):
    module = importlib.import_module(f'models.{model_name}')
    model = module.System(
        out_channels=model_args['out_channels'],
        mode=model_args['mode'],
    )

    model.model.load_state_dict(
        torch.load(
            weight_path,
            map_location=torch.device('cpu'),
        ),
        # strict=False,
    )
    return model


def get_model_prediction(model, input):
    model.eval()
    with torch.no_grad():
        out = model(input)

    return out.sigmoid().detach().numpy()[0]


def main():
    args = option_parser()

    set_seed()

    dataloader = get_test_dataloader(
        dataset_dir=args.dataset_dir,
    )
    model = get_model(
        model_name=args.model,
        weight_path=args.model_weight,
        model_args={
            'out_channels':2,
            'mode':'multilabel',
        },
    )

    for i, (frames, labels, info) in enumerate(dataloader):
        img = frames.permute(0,2,3,1)[0]
        mask = labels[0]

        input = frames.float()
        out = get_model_prediction(model=model, input=input)

        fig = plt.figure()
        fig.add_subplot(3,1,1)
        plt.imshow(img)
        fig.add_subplot(3,1,2)
        plt.imshow(mask[0])
        plt.gray()

        # heatmap = cv2.applyColorMap(
        #     (out[0] * 255).astype(np.uint8),
        #     cv2.COLORMAP_JET,
        # )
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        fig.add_subplot(3,1,3)
        plt.imshow(out[0])
        plt.gray()

        # plt.show()
        dir = "./images_test"
        plt.savefig(f'{dir}/{i}.png')

        # plt.imsave(f'{dir}/{i}_img.png', img.detach().numpy())
        # plt.imsave(f'{dir}/{i}_binary1.png', binary1)

        plt.clf()
        plt.close()


if __name__ == "__main__":
    main()
