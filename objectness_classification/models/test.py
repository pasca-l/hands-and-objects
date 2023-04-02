import os
import sys
import git
import matplotlib.pyplot as plt
import torch

git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")

sys.path.append(f"{git_root}/objectness_classification/")
from seed import set_seed
sys.path.append(f"{git_root}/objectness_classification/datasets")
from datamodule import ObjnessClsDataModule

def main():
    home = os.path.expanduser("~")
    dataset_dir = os.path.join(home, "Documents/datasets")

    set_seed()
    iter_counter = 0

    egohos_dataset = ObjnessClsDataModule(
        dataset_dir=dataset_dir,
        dataset_mode='egohos',
        batch_size=1,
        with_transform=False,
        with_info=True,
    )
    egohos_dataset.setup()

    egohos_train_dataloader = iter(egohos_dataset.train_dataloader())

    egohos_frames, egohos_labels, egohos_file = next(egohos_train_dataloader)
    iter_counter += 1
    print(
        f"iter counter: {iter_counter}",
        f"frames shape: {egohos_frames.shape}",
        f"file name: {egohos_file}",
        sep="\n",
    )
    egohos_img = egohos_frames[0]
    egohos_mask = egohos_labels[0]

    from unet import System

    system = System()
    model = system.model

    model_path = os.path.join(git_root, "objectness_classification/logs/unet_2cls.ckpt")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()

    output = model(egohos_frames.float())
    output = output.detach().numpy()
    cls_num, _, _ = output[0].shape
    print(
        f"model input shape: {egohos_frames.shape}",
        f"model output shape: {output.shape}",
        sep="\n",
    )

    fig = plt.figure()

    fig.add_subplot(2, 2, 1)
    plt.imshow(egohos_frames[0])

    for cls in range(cls_num):
        fig.add_subplot(2, 2, 2+cls)
        plt.imshow(output[0][cls])
        plt.gray()

    plt.show()

    plt.clf()
    plt.close()

if __name__ == "__main__":
    main()
