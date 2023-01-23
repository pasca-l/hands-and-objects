import sys
import argparse
import shutil
import importlib
import pytorch_lightning as pl

from dataset_module import StateChgObjDataModule
from system import StateChgObjDetector

sys.path.append("../utils")
from json_handler import JsonHandler


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default="fho_scod",
                        choices=["fho_scod"])
    parser.add_argument('-d', '--data_dir', type=str, 
                        default='/home/ubuntu/data/ego4d/')
    parser.add_argument('-a', '--ann_dir', type=str,
                        default='/home/ubuntu/data/ego4d/annotations/')
    parser.add_argument('-m', '--model', type=str, default="faster_rcnn",
                        choices=["faster_rcnn", "finetune_resnet"])

    return parser.parse_args()


def main():
    args = option_parser()

    json_handler = JsonHandler(args.task)
    json_partial_name = f"{args.data_dir}annotations/{args.task}"
    json_dict = {
        "train": json_handler(f"{json_partial_name}_train.json"),
        "val": json_handler(f"{json_partial_name}_val.json"),
    }

    dataset = StateChgObjDataModule(
        data_dir=f"{args.data_dir}frames/",
        json_dict=json_dict,
        model_name=args.model,
        batch_size=1,
        label_mode='corners'    # 'corners' or 'COCO'
    )

    module = importlib.import_module(f'models.{args.model}')
    system = module.System()
    detector = StateChgObjDetector(
        sys=system
    )

    import torchvision, cv2, torch
    dataset.setup()
    data = next(iter(dataset.train_dataloader()))
    img, labels = data[0], data[1]
    image = img[0][1].to(torch.uint8)

    result = detector(img[0][1].permute(2,0,1).unsqueeze(0))
    boxes, labels = result[0]['boxes'], result[0]['labels']
    cv2.imwrite("test.jpg", image.detach().numpy())


if __name__ == '__main__':
    main()
