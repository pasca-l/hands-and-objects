import argparse
import pytorch_lightning as pl

from dataset_module import PNRTempLocDataModule
# from models.cnnlstm import CnnLstm as Model
# from models.SFP.slowfastpreceiver import SlowFastPreceiver as Model
from pnr_temporal_localization.models.slowfastperceiver import SlowFastPreceiver as Model
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
    model = Model()
    print(model)
    return
    classifier = PNRLocalizer(model)

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
        accelerator='cpu',
        devices='auto',
        auto_select_gpus=True,
        max_epochs=10,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(classifier, dataset)


if __name__ == '__main__':
    main()
