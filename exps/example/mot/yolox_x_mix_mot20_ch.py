# encoding: utf-8
"""
MOT20 混合检测配置（ch = crowded human 训练）。
作为 yolox_x_mot20_v001.py 的父类，提供 MOT20 评估关键属性：
- 模型架构: YOLOX-X (depth=1.33, width=1.25), 1 类 (person)
- test_size=(896,1600)（与 mot20_v001_full 运行的 987.1 Gflops 一致）
- mot20=True（纯 IoU 匹配的 MOT20 评估模式）
注：本项目不做训练，get_data_loader 等训练接口仅为保持官方文件结构。
"""
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.warmup_epochs = 1
        self.max_epoch = 8
        self.data_dir = os.path.join(get_yolox_datadir(), "mix_det")
        self.train_ann = "crowdhuman_train.json"
        self.val_ann = "crowdhuman_val.json"
        self.input_size = (800, 1440)
        self.test_size = (896, 1600)
        self.random_size = (18, 32)
        self.test_conf = 0.3
        self.nmsthre = 0.7
        self.mot20 = True
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            infinite_iter,
            MosaicDetection,
        )

        datadir = self.data_dir
        dataset = MOTDataset(
            data_dir=datadir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
        )
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=self.seed is None
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_sampler"] = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        # Make sure to have a shuffle=True batch sampler
        dataloader_kwargs["shuffle"] = False

        # -1 means unlimited numbers of workers
        self.dataset_len = len(self.dataset)
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            img_size=self.test_size,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader
