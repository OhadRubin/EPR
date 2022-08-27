from typing import Dict, Tuple, List
import logging

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import Average

from qdecomp_with_dependency_graphs.utils.modules import tie_models_modules

logger = logging.getLogger(__name__)


@Model.register("custom_multitask")
class MultitaskModel(Model):
    """
    Multitask model that combines multiple models, and runs each of them according to the given instance.
    Assumed to be used with `InterleavingBatchesDatasetReader` where the dataset_field_name='dataset'
    and the Models keys are the same as the corresponding datasets keys.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 models: Dict[str, Model],
                 tie_modules: List[List[str]] = [],
                 loss_weights: Dict[str, float] = None,
                 ):
        super().__init__(vocab)

        # update models as modules (for cuda, trainable, etc)
        self._models: Dict[str, Model] = models
        for name, model in self._models.items():
            self.add_module(name, model)
        self._task_to_index = {m: i for i, m in sorted(enumerate(self._models.keys()), key=lambda x: x[1])}
        self._index_to_task = {v: k for k, v in self._task_to_index.items()}

        # tie modules
        tie_models_modules(models=self._models, tie_modules_groups=tie_modules)

        self._loss_weights = loss_weights

        # tasks loss as metrics
        self._losses: Dict[str, Average] = {k: Average() for k in self._models.keys()}
        self._losses.update({"total": Average()})

    def forward(self, task: List[str], *args, **kwargs) -> Dict[str, torch.Tensor]:
        task_set = set(task)
        if len(task_set) != 1:
            raise ValueError(f"Unexpected batch - got instances from different tasks {task_set}")
        task = task_set.pop()
        model = self._models.get(task)
        outoput = model.forward(*args, **kwargs)
        if "loss" in outoput:
            if self._loss_weights:
                outoput['loss'] *= self._loss_weights.get(task, 1.0)
            loss = outoput['loss'].item()  # float (need to be serialized to json)
            self._losses[task](loss)
            self._losses["total"](loss)
        outoput['task_index'] = torch.tensor(self._task_to_index[task])
        return outoput

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        task_index = output_dict["task_index"].item()
        task = self._index_to_task[task_index]
        model = self._models[task]
        return model.make_output_human_readable(output_dict=output_dict)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        models_metrics = {}
        for model_name, model in self._models.items():
            # workaround: calling get_metrics() before any call to this model forward
            try:
                models_metrics.update({
                    f'{model_name}-{k}': v for k, v in model.get_metrics(reset=reset).items()
                })
            except Exception:
                logger.exception(f"Failed to get metrics from model {model_name}")
        models_losses = {f'{k}-loss': v.get_metric(reset=reset) for k, v in self._losses.items()}
        return {**models_metrics, **models_losses}

