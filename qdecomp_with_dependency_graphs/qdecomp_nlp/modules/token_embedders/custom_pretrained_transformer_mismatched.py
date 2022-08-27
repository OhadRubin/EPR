from typing import Optional

from enum import Enum
from overrides import overrides
import torch

from allennlp.modules.token_embedders import PretrainedTransformerEmbedder, PretrainedTransformerMismatchedEmbedder, TokenEmbedder
from allennlp.nn import util

import inspect

class Aggregation(str, Enum):
    AVG = 'average'
    FIRST = 'first'

@TokenEmbedder.register("custom_pretrained_transformer_mismatched")
class CustomPretrainedTransformerMismatchedEmbedder(PretrainedTransformerMismatchedEmbedder):
    """
    Allow adding parameters of `PretrainedTransformerEmbedder` to `PretrainedTransformerMismatchedEmbedder`
    """

    def __init__(
        self,
        aggregation: Aggregation = Aggregation.AVG,
        **kwargs
    ) -> None:
        sig = inspect.signature(super().__init__)
        super_kwargs ={k: v for k,v in kwargs.items() if k in sig.parameters}
        super().__init__(**super_kwargs)
        # The matched version v.s. mismatched
        self._matched_embedder = PretrainedTransformerEmbedder(
            **kwargs
        )
        self._aggregation = aggregation

    @overrides
    def forward(
            self,
            token_ids: torch.LongTensor,
            mask: torch.BoolTensor,
            offsets: torch.LongTensor,
            wordpiece_mask: torch.BoolTensor,
            type_ids: Optional[torch.LongTensor] = None,
            segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: [batch_size, num_wordpieces] (for exception see `PretrainedTransformerEmbedder`).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_orig_tokens].
        offsets: `torch.LongTensor`
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
        wordpiece_mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: [batch_size, num_wordpieces].
        segment_concat_mask: `Optional[torch.BoolTensor]`
            See `PretrainedTransformerEmbedder`.

        # Returns

        `torch.Tensor`
            Shape: [batch_size, num_orig_tokens, embedding_size].
        """
        if self._aggregation == Aggregation.AVG:
            return super().forward(token_ids=token_ids, mask=mask, offsets=offsets, wordpiece_mask=wordpiece_mask,
                                   type_ids=type_ids, segment_concat_mask=segment_concat_mask)

        # Shape: [batch_size, num_wordpieces, embedding_size].
        embeddings = self._matched_embedder(
            token_ids, wordpiece_mask, type_ids=type_ids, segment_concat_mask=segment_concat_mask
        )

        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask  # zero out paddings

        orig_embeddings = span_embeddings[:,:,0,:]
        span_embeddings_len = span_mask.sum(2)

        # All the places where the span length is zero, write in zeros.
        orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

        return orig_embeddings
