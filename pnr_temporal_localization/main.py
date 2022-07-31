import argparse
import pytorch_lightning as pl

from dataset_module import PNRTempLocDataModule
from models.cnnlstm import CnnLstmSys as Module
# from models.slowfastperceiver import SlowFastPreceiver as Module
# from models.bmn import BMNSys as Module
# from models.i3d_resnet import I3DResNetSys as Module
from system import PNRLocalizer


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PNR', choices=['PNR'])
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--log_save_dir', type=str, default='./logs/')
    parser.add_argument('--model_save_name', type=str, default='trained_model')
    parser.add_argument('--ann_dir', type=str,
                        default='../../../data/ego4d/annotations/')
    parser.add_argument('--data_dir', type=str, 
                        default='../../../data/ego4d/clips/')

    return parser.parse_args()


def main():
    args = option_parser()

    dataset = PNRTempLocDataModule(
        data_dir=args.data_dir,
        ann_dir=args.ann_dir,
        ann_task_name="fho_hands",
        batch_size=4
    )

    # model = Module().model
    # dataset.setup()
    # data = next(iter(dataset.train_dataloader()))
    # print(data[0].shape, data[1][0])
    # a = model(data[0])
    # print(len(a), [i.shape for i in a])
    # return

    module = Module()
    classifier = PNRLocalizer(module)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=args.log_save_dir
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        save_weights_only=True,
        monitor="train_loss",
        mode='min',
        dirpath=args.log_save_dir,
        filename=args.model_save_name
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices='auto',
        auto_select_gpus=True,
        max_epochs=10,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(classifier, dataset)


if __name__ == '__main__':
    main()
