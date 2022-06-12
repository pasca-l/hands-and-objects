import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import PNRTempLocDataset

from trainval import train, val
from evaluate import evaluate, generate_submission_file, generate_submission_file_cls
from models import CnnLstm


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

    model = CnnLstm()
    model.cuda()

    if args.phase == 'train':
        train_dataset =\
            PNRTempLocDataset(ann_dir=args.ann_dir, clip_dir=args.clip_dir)
        train_dataloader =\
            DataLoader(train_dataset, batch_size=4, pin_memory=True,
                       num_workers=8, shuffle=True)
        val_dataset =\
            PNRTempLocDataset(phase='val', ann_dir=args.ann_dir,
                              clip_dir=args.clip_dir)
        val_dataloader =\
            DataLoader(val_dataset, batch_size=1, pin_memory=True,
                       num_workers=8) 

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.BCELoss().cuda()
        best_loss = 99999
        best_epoch = 0
        for epoch in range(10):
            train(model, train_dataloader, optimizer, criterion, epoch)
            # torch.save(model.state_dict(), 'epoch_%d_'%epoch+args.save_name)

            loss, _, _ = val(model, val_dataloader, criterion, epoch)
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                torch.save(model.state_dict(), args.model_save_name)
                print('best model at epoch %d' % best_epoch)


if __name__ == '__main__':
    main()
