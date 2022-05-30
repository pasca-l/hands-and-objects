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
    parser.add_argument('--ann_dir', type=str, default='/Users/shionyamadate/Documents/ego4d_data/v1/annotations/')
    parser.add_argument('--clip_dir', type=str, default='/Users/shionyamadate/Documents/ego4d_data/v1/clips/')

    return parser.parse_args()


def main():
    args = option_parser()

    # if args.task == 'PNR':
    #     state_change = True
    #     criterion = nn.BCELoss().cuda()
    #     save_name = 'PNR_' + args.model_save_name

    if args.phase == 'train':
        train_dataset =\
            PNRTempLocDataset(ann_dir=args.ann_dir, clip_dir=args.clip_dir)
        train_dataloader =\
            DataLoader(train_dataset, batch_size=4, pin_memory=True,
                       num_workers=8, shuffle=True)

    model = CnnLstm()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best_loss = 99999
    best_epoch = 0
    for epoch in range(10):
        pass


if __name__ == '__main__':
    main()
