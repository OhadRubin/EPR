from typing import Dict, Tuple, List, Any
import logging
from enum import Enum

from allennlp.nn import util, InitializerApplicator
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from overrides import overrides
import torch

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models.model import Model
from allennlp.training.metrics import Average, Metric, CategoricalAccuracy

from qdecomp_nlp.models.hybrid.multitask_model import MultitaskModel
from qdecomp_nlp.models.seq2seq.custom_copynet_seq2seq import CustomCopyNetSeq2Seq
from qdecomp_with_dependency_graphs.utils.helpers import rgetattr, rsetattr
from qdecomp_with_dependency_graphs.utils.modules import tie_models_modules

logger = logging.getLogger(__name__)


@Model.register("multitask_soft_rat")
class MultitaskSoftRatBasedModel(MultitaskModel):
    """
    Multitask model
    """
    def __init__(self,
                 vocab: Vocabulary,
                 tags_namespace: str,
                 zero_nones: bool = True,
                 **kwargs
                 ):
        super().__init__(vocab, **kwargs)
        assert len(self._models) == 2 and all(x in self._models for x in ['seq2seq', 'graph_parser'])
        self._seq2seq_model = self._models['seq2seq']
        self._graph_parser_model = self._models['graph_parser']

        self._seq2seq_model_forward = self._seq2seq_model.forward
        self._seq2seq_model.forward = self._seq2seq_forward

        self._zero_nones = zero_nones
        self._none_index = vocab.get_token_index('NONE', tags_namespace)

    def _seq2seq_forward(self, source_tokens: TextFieldTensors, metadata: List[Dict[str, Any]], **kwargs):
        graph_output = self._graph_parser_model(tokens=source_tokens, metadata=metadata)
        relations_probs = graph_output['arc_tag_probs']
        if self._zero_nones:
            relations_probs[:, :, self._none_index] = 0
        return self._seq2seq_model_forward(source_tokens=source_tokens,
                                           metadata=metadata,
                                           relations_probs=relations_probs,
                                           **kwargs)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = super().get_metrics(reset=reset)

        logical_form_ems = [v for k, v in metrics.items() if 'logical_form_em' in k]
        logical_form = {'maximal_logical_form_em': max(logical_form_ems)} if logical_form_ems else {}
        return {**metrics, **logical_form}
