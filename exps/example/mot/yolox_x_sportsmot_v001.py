# encoding: utf-8
"""
V001 standard evaluation config: SportsMOT
- data_dir: datasets/SportsMOT (V001-V240, file_name already contains the Vxxx prefix)
- name='', val_ann='eval.json'
- Inherits the backbone from yolox_x_mix_det.py
  (depth=1.33, width=1.25, test_size=(800,1440))
- NOTE: the official pretrained weight has a detection head with only 1 class
  (person). num_classes MUST stay 1, otherwise load_state_dict raises a shape
  mismatch. Keep it 1 even though the official annotations contain more classes.
"""
import os
import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir

from yolox_x_mix_det import Exp as MixDetExp


class Exp(MixDetExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.dataset_name = "SportsMOT"
        self.num_classes = 1  # weight head is 1-class person; do NOT change
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
