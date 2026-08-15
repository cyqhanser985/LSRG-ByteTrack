# encoding: utf-8
"""
V001 standard evaluation config: MOT17
- data_dir: datasets/MOT17 (V001-V021, file_name already contains the Vxxx prefix)
- name='', val_ann='eval.json'
- Inherits the backbone from yolox_x_mix_det.py (num_classes=1, test_size=(800,1440))
"""
import os
import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir

from yolox_x_mix_det import Exp as MixDetExp


class Exp(MixDetExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.dataset_name = "MOT17"
        self.val_ann = "eval.json"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), self.dataset_name),
            json_file=self.val_ann,
            img_size=self.test_size,
            name="",
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            seq_filter=getattr(self, "eval_seq", None),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": 0,  # eval: avoid pickling large datasets into workers (RAM safety)
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader
