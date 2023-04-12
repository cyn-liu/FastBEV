import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.loops import TestLoop
from mmengine.runner.amp import autocast


__all__ = ['CustomVisLoop']
@LOOPS.register_module()
class CustomVisLoop(TestLoop):
    def __init__(self, iter_nums=150, **kwargs):
        super().__init__(**kwargs)
        self.iter_nums = min(len(self.dataloader), iter_nums)

    def run(self) -> None:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        results = []
        for idx, data_batch in tqdm(enumerate(self.dataloader), total=self.iter_nums, unit='frame'):
            if idx >= self.iter_nums:
                break
            results.append(self.run_iter(idx, data_batch))

        self.dataloader.dataset.vis(results, self.runner)

        # # compute metrics
        # metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        # self.runner.call_hook('after_test_epoch', metrics=metrics)
        # self.runner.call_hook('after_test')
        # return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)
        return outputs
