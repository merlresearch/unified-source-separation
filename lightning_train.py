# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import random
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import torchaudio.functional as F

from datasets.dataset import DynamicMixingDataset, FixedDataset
from datasets.sampler import CustomBatchSampler, CustomSampler
from loss_functions.loss_wrapper import LossWrapper
from nets.model_wrapper import SeparationModel
from utils.average_model_params import average_model_params
from utils.collate import collate_seq


class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        if not isinstance(hparams, Namespace):
            hparams = Namespace(hparams.model_name, **hparams.model_conf)

        self.save_hyperparameters(hparams)
        self.model = SeparationModel(
            hparams.encoder_name,
            hparams.encoder_conf,
            hparams.decoder_name,
            hparams.decoder_conf,
            hparams.model_name,
            hparams.model_conf,
            hparams.css_conf,
            hparams.variance_normalization,
        )

        # loss related
        self.loss = LossWrapper(hparams.loss_func_name, hparams.loss_func_conf, pit_loss=hparams.pit_loss)
        self.sisnr = LossWrapper("si_snr", dict(clamp_db=70), pit_loss=hparams.pit_loss)
        self.snr = LossWrapper("snr", {}, pit_loss=hparams.pit_loss)

        # define current_step for learning rate warming-up
        self.current_step = 0
        self.keep_lr_epochs = getattr(hparams, "keep_lr_epochs", 0)

        # settings for the css-style inference for long recordings
        # e.g., MSS or CASS
        self.css_datasets = hparams.css_datasets

        # IMPORTANT: audio is always resampled to 48kHz
        self.resampling_rate = 48000

        # other settings
        self.sort_prompts = getattr(self.hparams, "sort_prompts", False)
        self.p_prompt_dropout = getattr(hparams, "p_prompt_dropout", 0.0)

        # only for testing
        self.eval_with_single_prompt = getattr(hparams, "single_prompt", False)
        self.result_filename = "result_single.txt" if self.eval_with_single_prompt else "result.txt"

    def on_batch_end(self):
        # learning rate warmup
        self.warmup_lr()

    def on_train_epoch_end(self):
        # shuffle the dataset for the next epoch
        if hasattr(self.dset, "setup"):
            self.dset.setup()

    def _step(
        self,
        batch,
        return_loss_as_list=False,
        eval_metrics=["sisnr"],
        eval_with_single_prompt=False,
    ):
        # validation and test dataloader returns a list of 5 elements
        if len(batch) == 5:
            assert not self.training
            mix, ref, prompts, dset_name, sample_rate = batch

            # all the elements of dset_name and sample_rate must be the same
            assert all([dn == dset_name[0] for dn in dset_name])
            assert all([fs == sample_rate[0] for fs in sample_rate])
            dset_name = dset_name[0]
            sample_rate = sample_rate[0]
        else:
            assert self.training
            mix, ref, prompts = batch
            dset_name = None
            sample_rate = None

        if self.training and self.p_prompt_dropout > 0.0:
            prompts, ref = self._prompt_dropout(prompts, ref)

        # Resample the audio to the target sample rate (48kHz).
        # This is done here only during validation or testing, but not in training
        # because we need to resample the separated audio to the original sample rate
        # to evaluate the performance. During training, resampling is done when loading
        # the data after dynamic mixing but before stacking them to a single mini-batch
        # to align the sample rate of all the data in the same mini-batch.
        resample = not self.training and self.resampling_rate != sample_rate
        if resample:
            with torch.amp.autocast("cuda", enabled=False):
                mix = F.resample(mix, sample_rate, self.resampling_rate)

        # separation
        forward_func = self.model.css if not self.training and dset_name in self.css_datasets else self.model

        if eval_with_single_prompt:
            est, ref, prompts = self._single_prompt_inference(mix, ref, prompts, forward_func)
        else:
            est = forward_func(mix, prompts)

        # resample the separated audio to the original sample rate
        if resample:
            with torch.amp.autocast("cuda", enabled=False):
                est = F.resample(est, self.resampling_rate, sample_rate)

        # sort prompts and references, which is necessary when training a conventional
        # model for the tasks where the output order should be always the same
        # (e.g., music source separation or cinematic audio source separation)
        # because the order of prompts and references are random in each training step
        if self.sort_prompts:
            ref_tmp = torch.zeros_like(ref)
            for b in range(len(prompts)):
                idx = np.argsort(prompts[b])
                prompts[b] = [prompts[b][i] for i in idx]
                ref_tmp[b] = ref[b, idx]
            ref = ref_tmp

        # loss computation
        ref = ref.float()
        loss, _ = self.loss(est, ref, prompts, return_list=return_loss_as_list)
        metrics = dict(loss=loss)

        if "sisnr" in eval_metrics:
            sisnr, prompts = self.sisnr(est, ref, prompts, return_list=return_loss_as_list)
            metrics["sisnr"] = sisnr
        if "snr" in eval_metrics:
            snr, prompts = self.snr(est, ref, prompts, return_list=return_loss_as_list)
            metrics["snr"] = snr

        return metrics, dset_name, prompts, sample_rate, est

    def _prompt_dropout(self, prompts, ref):
        """Randomly drop some unique prompts from the mini-batch,
        as described in Sec. II-C of the paper.

        This method only drops the unique prompts, not the duplicated prompts,
        because droppping the duplicated prompts makes the problem ill-posed.
        For instance, with prompts = [["speech", "speech", "sfx", "musicbg"]],
        dropping a single "speech" would make it unclear which of the two
        speech sources to extract. Thus, the dropped prompts are chosen from
        "sfx" and/or "musicbg" in this case.
        """
        # count the number of each prompt category
        counts = [Counter(pr) for pr in prompts]
        # count the number of prompts that are not unique
        num_dup = max([sum([c for c in count.values() if c != 1]) for count in counts])
        num_max_drop = min(len(prompts[0]) - num_dup, len(prompts[0]) - 1)

        if num_max_drop > 0:
            # randomly choose number of prompts to drop
            num_drop = random.choices(
                list(range(num_max_drop + 1)),
                weights=[1.0 - self.p_prompt_dropout] + [self.p_prompt_dropout / num_max_drop] * num_max_drop,
            )[0]

            if num_drop > 0:
                new_ref = ref.new_zeros((ref.shape[0], ref.shape[1] - num_drop, *ref.shape[2:]))
                new_prompts = []

                for b, pr in enumerate(prompts):
                    unique_indices = [idx for idx, item in enumerate(pr) if counts[b][item] == 1]
                    to_remove = random.sample(unique_indices, num_drop)
                    to_keep = [i for i in range(len(pr)) if i not in to_remove]
                    assert len(to_keep) == new_ref.shape[1], (
                        to_keep,
                        new_ref.shape,
                    )

                    new_prompts.append([p for i, p in enumerate(pr) if i in to_keep])
                    for i, idx in enumerate(to_keep):
                        new_ref[b, i, :] = ref[b, idx, :]

                ref = new_ref
                prompts = new_prompts

        return prompts, ref

    def _single_prompt_inference(self, mix, ref, prompts, forward_func):
        """Inference with a single prompt.
        This method runs a for loop over the prompts and
        concatenates the separated sources.

        Similarly in `_prompt_dropout`, to avoid ill-posed conditions,
        the duplicated prompts are grouped as a single prompt.
        For instance, when prompts = [["speech", "speech", "sfx", "musicbg"]],
        this method runs for loop over [["speech", "speech"], ["sfx"], ["musicbg"]]
        instead of [["speech"], ["speech"], ["sfx"], ["musicbg"]].
        """
        assert not self.training
        assert len(prompts) == 1  # batch size must be 1

        # sort prompts and references first
        ref_tmp = torch.zeros_like(ref)
        idx = np.argsort(prompts[0])
        prompts[0] = [prompts[0][i] for i in idx]
        ref_tmp[0] = ref[0, idx]
        ref = ref_tmp.clone()

        # make list of single (or multiple "speech" or "sfx") prompt(s)
        prompt_dict = defaultdict(list)
        for p in prompts[0]:
            prompt_dict[p].append(p)

        # should be list of lists
        # something like [["speech", "speech"], ["sfx"], ["musicbg"]]
        prompts = list(prompt_dict.values())

        # forward for each prompt
        for i, p in enumerate(prompts):
            mix_clone = mix.clone()
            p = [p]
            est = forward_func(mix_clone, p)

            if i == 0:
                est_list = [est]
            else:
                est_list.append(est)
            est = torch.cat(est_list, dim=1)

        # flatten list of lists
        prompts = [[item for sublist in prompts for item in sublist]]

        return est, ref, prompts

    def training_step(self, batch, batch_idx):
        metrics = self._step(batch)[0]

        log = {}
        for metric_name, score in metrics.items():
            log[f"train/{metric_name}"] = score

        self.log_dict(
            log,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.hparams.batch_size,
        )
        self.on_batch_end()
        return metrics["loss"]

    def on_validation_epoch_start(self):
        self.validation_results = []

    def validation_step(self, batch, batch_idx):
        metrics, dset_name, prompts, *_ = self._step(batch, return_loss_as_list=True)

        # each mini-batch must have the same prompts in validation
        assert all(sublist == prompts[0] for sublist in prompts)
        prompts = prompts[0]

        # dict to store the index of each prompt
        prompts_and_index = defaultdict(list)
        for i, pr in enumerate(prompts):
            prompts_and_index[pr].append(i)

        log = {}
        for metric_name, score in metrics.items():
            score = np.array(score)  # (batch x prompts)
            for prompt, idx in prompts_and_index.items():
                log[f"{dset_name}_val/{prompt}/{metric_name}"] = score[:, idx].mean()

        # log the dataset- and source-wise average score
        self.log_dict(
            log,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.hparams.val_batch_size,
        )

        self.validation_results.append(log)

        return metrics["loss"]

    def on_validation_epoch_end(self):
        """
        Compute the average of the validation results for each dataset.
        Then average over the datasets and log.
        """
        results = defaultdict()

        for result in self.validation_results:
            # key is like "{dset_name}_val/{prompt}/{metric_name}"
            # value is the score
            for key, value in result.items():
                dset_name = key.split("/")[0].replace("_val", "")
                metric_name = key.split("/")[-1]

                if metric_name not in results:
                    results[metric_name] = defaultdict()

                if dset_name not in results[metric_name]:
                    results[metric_name][dset_name] = []
                results[metric_name][dset_name].append(value)

        # average the results for each metric
        for metric_name, result_metric in results.items():
            for dset_name, result_dset in result_metric.items():
                num_data = len(result_dset)
                score = sum(result_dset) / num_data
                results[metric_name][dset_name] = score

        # average the results for each dataset
        log = defaultdict()
        for metric_name, result in results.items():
            score = sum(result.values()) / len(result)
            log[f"val/{metric_name}"] = score

        # log the validation scores over all datasets
        self.log_dict(
            log,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )

    def on_test_epoch_start(self):
        assert not (
            Path(self.logger.log_dir) / self.result_filename
        ).exists(), f"{self.result_filename} already exists."

        self.test_results = []

    def test_step(self, batch, batch_idx):
        assert len(batch[0]) == 1, "batch size must be 1 when testing"
        metrics, dset_name, prompts, sample_rate, est = self._step(
            batch,
            return_loss_as_list=True,
            eval_metrics=["sisnr", "snr"],
            eval_with_single_prompt=self.eval_with_single_prompt,
        )
        metrics_for_logging = {}

        # each mini-batch must have the same prompts in testing
        prompts = prompts[0]

        # sort prompts for logging
        prompts_and_index = defaultdict(list)
        for i, pr in enumerate(prompts):
            prompts_and_index[pr].append(i)

        for k, v in metrics.items():
            v = np.array(v)  # (batch x prompts)
            v = v * -1 if k != "loss" else v  # negative -> positive
            metrics_for_logging[f"test/{k}"] = v.mean()
            for prompt, idx in prompts_and_index.items():
                metrics_for_logging[f"{dset_name}_test/{prompt}/{k}"] = v[:, idx].mean()
        self.log_dict(
            metrics_for_logging,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        self.test_results.append(metrics_for_logging)

        # save audio files
        output_dir = Path(self.trainer.logger.log_dir) / "audio_outputs" / dset_name
        output_dir.mkdir(exist_ok=True, parents=True)

        # save the seprarated sources
        sisnr_list = metrics["sisnr"][0]  # remove batch dim
        est = est.squeeze(0).cpu().numpy()  # remove batch dim and move to cpu
        for i in range(len(sisnr_list)):  # for loop over the number of sources
            sisnr = round(sisnr_list[i], 1)
            filename = f"data{batch_idx}_{prompts[i]}{i}_sisnr={sisnr}.wav"
            sf.write(output_dir / filename, est[i], sample_rate)

        return metrics_for_logging

    def on_test_epoch_end(self):
        results = defaultdict()
        for result in self.test_results:
            for key, value in result.items():
                if key not in results:
                    results[key] = []
                results[key].append(value)

        results = dict(sorted(results.items()))

        # write the result
        output_dir = self.logger.log_dir
        output_file = open(Path(output_dir) / self.result_filename, "w")
        for key, result in results.items():
            num_data = len(result)
            score = round(sum(result) / num_data, 2)
            output_file.write(f"{key}: {score}\n")
        output_file.close()

    def configure_optimizers(self):
        optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
        schedulers = {"reducelr": torch.optim.lr_scheduler.ReduceLROnPlateau}

        optimizer_name = self.hparams.optimizer_name
        scheduler_name = self.hparams.scheduler_name
        assert optimizer_name in optimizers and scheduler_name in schedulers

        optim_params = self.parameters()
        optimizer = optimizers[optimizer_name](optim_params, **self.hparams.optimizer_conf)
        scheduler = schedulers[scheduler_name](optimizer, **self.hparams.scheduler_conf)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }

    def _init_fn(self, worker_id):
        random.seed(self.hparams.seed + worker_id)
        np.random.seed(self.hparams.seed + worker_id)
        torch.manual_seed(self.hparams.seed + worker_id)

    def setup(self, stage):
        """This method sets a different seed for each DDP process.
        This method is called when calling trainer.fit().
        (Also called when calling, e.g., trainer.validate() but we don't do that)

        Due to our custom dataset, we need to set the seed for each process.
        Otherwise, the same data will be loaded in all the processes.
        """
        if self.trainer.global_rank is not None:
            from pytorch_lightning import seed_everything

            rank = self.trainer.global_rank
            seed = self.hparams.seed + rank
            seed_everything(seed)

    def _get_data_loader(self, partition):
        if partition == "train":
            if self.hparams.dynamic_mixing:
                # DynamicMixingDataset needs to know the number of data per epoch
                gpu_count = torch.cuda.device_count()
                num_train_steps_per_epoch = self.hparams.trainer_conf["limit_train_batches"]
                num_data_per_epoch = gpu_count * num_train_steps_per_epoch * self.hparams.batch_size
                # create the dataset and batch sampler
                dataset = DynamicMixingDataset(
                    partition,
                    prompts=self.hparams.prompts,
                    num_data_per_epoch=num_data_per_epoch,
                    **self.hparams.dataset_conf["train"],
                )
                batch_sampler = CustomBatchSampler(
                    CustomSampler(dataset.get_len()),
                    self.hparams.batch_size,
                    self.hparams.num_srcs_and_weights,
                    shuffle=True,
                    drop_last=False,
                )
                self.dset = dataset

                return torch.utils.data.DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=collate_seq,
                    num_workers=self.hparams.num_workers,
                    worker_init_fn=self._init_fn,
                )
            else:
                json_path_and_num_data = self.hparams.dataset_conf[partition].pop("json_path_and_num_data")
                dsets = [
                    FixedDataset(
                        partition,
                        json_path=path,
                        num_data=num_data,
                        **self.hparams.dataset_conf[partition],
                    )
                    for (path, num_data) in json_path_and_num_data
                ]
                dataset = torch.utils.data.ConcatDataset(dsets)
                self.dset = dataset

                return torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.hparams.batch_size,
                    collate_fn=collate_seq,
                    num_workers=self.hparams.num_workers,
                    worker_init_fn=self._init_fn,
                )
        else:
            batch_size = self.hparams.val_batch_size if partition == "valid" else 1
            json_path_and_num_data = self.hparams.dataset_conf[partition].pop("json_path_and_num_data")
            dsets = [
                FixedDataset(
                    partition,
                    json_path=cfg[0],
                    num_data=cfg[1],
                    ref_channel=cfg[2] if len(cfg) == 3 else 0,
                    **self.hparams.dataset_conf[partition],
                )
                for cfg in json_path_and_num_data
            ]
            dataset = torch.utils.data.ConcatDataset(dsets)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate_seq,
                num_workers=self.hparams.num_workers,
                worker_init_fn=self._init_fn,
            )

    def train_dataloader(self):
        return self._get_data_loader("train")

    def val_dataloader(self):
        return self._get_data_loader("valid")

    def test_dataloader(self):
        return self._get_data_loader("test")

    def lr_scheduler_step(self, scheduler, metric):
        if self.keep_lr_epochs < self.current_epoch:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def warmup_lr(self):
        # get initial learning rate at step 0
        if self.global_step == 0:
            for param_group in self.optimizers().optimizer.param_groups:
                self.peak_lr = param_group["lr"]

        self.current_step += 1
        if getattr(self.hparams, "warmup_steps", 0) >= self.global_step:
            for param_group in self.optimizers().optimizer.param_groups:
                param_group["lr"] = self.peak_lr * self.global_step / self.hparams.warmup_steps

    def load_from_checkpoint2(self, checkpoint_dir, **kwargs):
        """
        Load the model from the averaged checkpoint files.
        If the checkpoint_dir is a directory, load the averaged model from the directory.
        If the checkpoint_dir is a .ckpt file, load the model from that file.
        """
        allowed_suffix = [".ckpt", ".pth"]
        if checkpoint_dir.suffix in allowed_suffix:
            checkpoint_paths = [checkpoint_dir]
        else:
            assert checkpoint_dir.is_dir(), f"{checkpoint_dir} is not a directory."
            checkpoint_paths = [
                path
                for path in Path(checkpoint_dir).iterdir()
                if path.suffix in allowed_suffix and path.name != "last.ckpt"
            ]
            assert all(
                [checkpoint_paths[0].suffix == p.suffix for p in checkpoint_paths]
            ), "all the suffix of the pre-trained weights files must be the same"
        state_dict = average_model_params(checkpoint_paths)
        self.load_state_dict(state_dict)
