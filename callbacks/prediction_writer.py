import os
from pathlib import Path
from typing import List, Optional, TypeVar

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.distributed import group as _group

T = TypeVar("T")


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        if not Path(self.output_dir).exists():
            os.makedirs(self.output_dir)

        if trainer.world_size > 1:
            list_gather_step_outputs = self._gather_objects(
                trainer=trainer, obj=predictions
            )
            if trainer.is_global_zero:
                torch.save(
                    list_gather_step_outputs,
                    os.path.join(self.output_dir, "predictions.pt"),
                )
            trainer.strategy.barrier()
            return
        else:
            torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))

    def _gather_objects(self, trainer, obj: T) -> Optional[List[T]]:
        if not trainer.is_global_zero:
            dist.gather_object(
                obj=obj, object_gather_list=None, dst=0, group=_group.WORLD
            )
            return None
        else:  # global-zero only
            list_gather_obj = [
                None
            ] * trainer.world_size  # the container of gathered objects.
            dist.gather_object(
                obj=obj, object_gather_list=list_gather_obj, dst=0, group=_group.WORLD
            )
            return sum(list_gather_obj, [])


class PredictionWriterAL(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(
            predictions,
            os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"),
        )
