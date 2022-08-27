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


@Model.register("custom_copynet_seq2seq_for_soft_rat")
class CustomCopyNetForRatSeq2Seq(CustomCopyNetSeq2Seq):
    @overrides
    def _encode(self, source_tokens: Dict[str, torch.Tensor], relations_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode source input sentences.
        """
        # shape: (batch_size, source_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, source_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, source_sequence_length, source_sequence_length)
        relations_mask = source_mask.unsqueeze(-1)*source_mask.unsqueeze(-2)
        # shape: (batch_size, source_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, relations_probs, relations_mask)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}
