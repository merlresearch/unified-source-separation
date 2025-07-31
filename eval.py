# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import argparse
from pathlib import Path

import loguru
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_train import TrainingModule
from utils.config import yaml_to_parser


def main(args):
    if args.ckpt_path.is_dir():
        assert (args.ckpt_path / "checkpoints").exists(), f"{args.ckpt_path} must contain checkpoints directory."
        config_path = args.ckpt_path / "hparams.yaml"
        ckpt_path = args.ckpt_path / "checkpoints"
    else:
        config_path = args.ckpt_path.parent.parent / "hparams.yaml"
        ckpt_path = args.ckpt_path

    hparams = yaml_to_parser(config_path)
    hparams = hparams.parse_args([])
    hparams.single_prompt = args.single_prompt

    exp_name, name, save_dir = [config_path.parents[i].name for i in range(3)]
    logger = TensorBoardLogger(save_dir=save_dir, name=name, version=exp_name)

    seed_everything(0, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True

    trainer = Trainer(
        logger=logger,
        enable_progress_bar=True,
        deterministic=True,
        devices=1,
        precision=hparams.trainer_conf["precision"],
    )
    # testing
    loguru.logger.info("Begin Testing")

    # load pytorch-lightning model
    model = TrainingModule(hparams)

    # load averaged separation model
    model.load_from_checkpoint2(ckpt_path)

    # overwrite css-related parameters
    if args.css_segment_size is not None:
        model.model.css_segment_size = args.css_segment_size
    if args.css_shift_size is not None:
        model.model.css_shift_size = args.css_shift_size

    trainer.test(model)
    loguru.logger.info("Testing complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        required=True,
        help="Path to the checkpoint."
        "If a directory containing multiple .ckpt files is given, all the weights"
        "in .ckpt files except for last.ckpt are averaged",
    )
    parser.add_argument(
        "--css_segment_size",
        type=int,
        required=False,
        default=None,
        help="CSS segment size for long recording separation.",
    )
    parser.add_argument(
        "--css_shift_size",
        type=int,
        required=False,
        default=None,
        help="CSS shift size for long recording separation.",
    )
    parser.add_argument("--single_prompt", action="store_true")
    args, other_options = parser.parse_known_args()
    main(args)
