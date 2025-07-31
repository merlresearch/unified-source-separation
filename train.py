# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import argparse
from pathlib import Path

import loguru
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_train import TrainingModule
from utils.config import yaml_to_parser


def main(args):

    hparams = yaml_to_parser(args.config)
    hparams = hparams.parse_args([])
    exp_name = args.config.stem

    # NOTE: this is basically only for controlling the randomness
    # of the model parameter initialization. Seeds for data loading
    # are set at setup() in the lightning module. See setup() in
    # lightning_train.py for more details.
    seed_everything(hparams.seed, workers=True)

    # some cuda configs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True

    logger = TensorBoardLogger(save_dir="exp", name="oss", version=exp_name)
    ckpt_dir = Path(logger.log_dir) / "checkpoints"

    model = TrainingModule(hparams)

    if (ckpt_dir / "last.ckpt").exists():
        # resume training from the latest checkpoint
        ckpt_path = ckpt_dir / "last.ckpt"
        loguru.logger.info(f"Resume training from {str(ckpt_path)}")
    elif getattr(hparams, "pretrained_model_path", None) is not None:
        # use pre-trained model weights
        ckpt_path = None
        if Path(hparams.pretrained_model_path.is_dir()):
            model_path = Path(hparams.pretrained_model_path) / "checkpoints"
        else:
            model_path = Path(hparams.pretrained_model_path) / "checkpoints"
        model.load_from_checkpoint2(model_path)
    else:
        # training from scratch
        ckpt_path = None

    ckpt_callback = ModelCheckpoint(**hparams.model_checkpoint)
    callbacks = [LearningRateMonitor(logging_interval="epoch"), ckpt_callback]
    if hparams.early_stopping is not None:
        callbacks.append(EarlyStopping(**hparams.early_stopping))
    if getattr(hparams, "save_checkpoint_every_n_epochs", None) is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(logger.log_dir) / "checkpoints_epoch",
                save_top_k=-1,
                every_n_epochs=hparams.save_checkpoint_every_n_epochs,
                save_last=False,
            )
        )

    trainer_conf = hparams.trainer_conf.copy()
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=False,
        devices=-1,
        **trainer_conf,
    )

    # training
    loguru.logger.info("Finished initial validation")
    trainer.fit(model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args, other_options = parser.parse_known_args()
    main(args)
