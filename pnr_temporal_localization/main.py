import os
import argparse
import torch

from datasets import PNRTempLocDataset


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PNR', choices=['PNR'])
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--model_save_name', type=str, default='cnn_lstm.pth')

    return parser.parse_args()


def main():
    args = option_parser()

    # if args.task == 'PNR':
    #     state_change = True
    #     criterion = nn.BCELoss().cuda()
    #     save_name = 'PNR_' + args.model_save_name

    # if args.phase == 'train':
    #     train_dataset = None
    #     train_dataloader =\
    #         DataLoader(train_dataset batch_size=4, pin_memory=True,
    #                    num_workers=8, shuffle=True)

    train_dataset = PNRTempLocDataset(ann_dir="/Users/shionyamadate/Documents/ego4d_data/v1/annotations/", clip_dir='/Users/shionyamadate/Documents/ego4d_data/v1/clips/')
    print(train_dataset.flatten_json[0]["pnr_hands"])

    # model = CnnLstm(state=(state_change))


if __name__ == '__main__':
    main()
