from typing import Any, Dict, List
from overrides import overrides

import torch
from allennlp.data import TextFieldTensors
from allennlp.models import Model
from allennlp.nn import InitializerApplicator
from allennlp_models.generation.models import SimpleSeq2Seq


@Model.register("simple_seq2seq_custom")
class SimpleSeq2SeqCustom(SimpleSeq2Seq):
    def __init__(self,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 **kwargs):
        super().__init__(**kwargs)
        initializer(self)

    @overrides
    def forward(
        self,
        source_tokens: TextFieldTensors,
        target_tokens: TextFieldTensors = None,
        metadata: List[Dict[str, Any]] = None,
        **kwargs  # skip extra fields
    ) -> Dict[str, torch.Tensor]:
        output_dict = super().forward(source_tokens=source_tokens, target_tokens=target_tokens)
        if metadata:
            output_dict['metadata'] = metadata
        return output_dict

