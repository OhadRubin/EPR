"""
Based on custom_copynet_seq2seq.py
"""
import logging
from typing import Dict

from overrides import overrides
import torch

from allennlp.models.model import Model
from allennlp.nn import util

from qdecomp_nlp.models.seq2seq.custom_copynet_seq2seq import CustomCopyNetSeq2Seq

logger = logging.getLogger(__name__)


@Model.register("custom_copynet_seq2seq_for_rat")
class CustomCopyNetForRatSeq2Seq(CustomCopyNetSeq2Seq):
    def __init__(
        self,
        decomposed_dependencies: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._decomposed_dependencies = decomposed_dependencies

    @overrides
    def _encode(self, source_tokens: Dict[str, torch.Tensor], relations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode source input sentences.
        """
        # shape: (batch_size, source_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, source_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        if self._decomposed_dependencies:
            # relations: [batch_size, parts, seq_len, seq_len]
            # make sure there is at least one part that is not None (-1)
            relations_mask = torch.sum(relations != -1, dim=1) != 0
            # FieldsListTokenEmbedder uses -1 for masking
        else:
            relations_mask = (relations != -1)
            relations[~relations_mask] = 0  # make sure valid index
        # shape: (batch_size, source_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, relations.long(), relations_mask)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}
