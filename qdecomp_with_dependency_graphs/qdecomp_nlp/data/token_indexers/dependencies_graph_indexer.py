"""
Based on allennlp/data/token_indexers/single_id_token_indexer.py
tag: v1.1.0
"""

from typing import Dict, List

import torch
from allennlp.common.util import pad_sequence_to_length
from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer

from qdecomp_nlp.data.token_indexers.multi_indexers_indexer import MultiIndexersTokenIndexer
from qdecomp_with_dependency_graphs.dependencies_graph.create_dependencies_graphs import get_extra_tokens


@TokenIndexer.register("dependencies_graph")
class DependenciesGraphTokenIndexer(MultiIndexersTokenIndexer):
    """
    A TokenIndexer for a pretrained GraphParser
    Should be used with `DependenciesGraphTokenEmbedder`
    If pos_tags_namespace is given, assumed to be used with `SpacyTokenizer` where pos_tags:true
    """

    def __init__(
            self,
            indexers: Dict[str, TokenIndexer] = None,
            pos_tags_namespace: str = None,
            add_extra_tokens: bool = True,
            original_sequence_mask_keyword: str = 'mask',
            **kwargs
    ) -> None:
        """
        Index tokens for pretrained GraphParser
        :param indexers:
            pretrained model tokens indexers
        :param pos_tags_namespace:
            namespace for pos tags. If None no pos_tags are indexed.
            If given, assumed to be used with `SpacyTokenizer` where pos_tags=true
        :param add_extra_tokens:
            Add graph special extra tokens
        :param original_sequence_mask_keyword:
            Keyword for original sequence mask. Set to different value than 'mask' to prevent conflicts with other
            indexers with 'mask' keyword. (see allennlp.nn.util.get_text_field_mask())
        :param kwargs:
        """
        if pos_tags_namespace and 'pos_tags' not in indexers:
            indexers['pos_tags'] = SingleIdTokenIndexer(namespace=pos_tags_namespace, feature_name='tag_')

        super().__init__(indexers=indexers, **kwargs)

        if add_extra_tokens:
            self._extra_tokens, self._extra_pos = get_extra_tokens()
        else:
            self._extra_tokens, self._extra_pos = [], []

        self._original_sequence_mask_keyword = original_sequence_mask_keyword

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> IndexedTokenList:
        tokens_ = tokens + [Token(text=x, tag_=y) for x, y in zip(self._extra_tokens, self._extra_pos)]

        output = super().tokens_to_indices(tokens=tokens_, vocabulary=vocabulary)

        # original sequence mask (excluding extra tokens)
        output[self._original_sequence_mask_keyword] = [True]*len(tokens)
        return output

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        output = super().get_empty_token_list()
        output[self._original_sequence_mask_keyword] = []
        return output

    @overrides
    def as_padded_tensor_dict(
            self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        output = super().as_padded_tensor_dict(tokens=tokens, padding_lengths=padding_lengths)

        mask_padding_lengths = padding_lengths.pop(self._original_sequence_mask_keyword)
        output[self._original_sequence_mask_keyword] = torch.BoolTensor(
            pad_sequence_to_length(
                tokens[self._original_sequence_mask_keyword], mask_padding_lengths, default_value=lambda: False
            )
        )
        return output
