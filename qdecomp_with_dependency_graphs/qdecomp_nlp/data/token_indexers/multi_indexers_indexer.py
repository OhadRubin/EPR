"""
Based on allennlp/data/token_indexers/single_id_token_indexer.py
tag: v1.1.0
"""
import dataclasses
from typing import Dict, List, Optional, Any
import itertools

import torch
from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList

from qdecomp_with_dependency_graphs.utils.data_structures import flatten_dict, nest_flatten_dict, merge_dataclasses


@TokenIndexer.register("multi")
class MultiIndexersTokenIndexer(TokenIndexer):
    """
    A TokenIndexer that wraps multiple indexers.
    Can be useful when you need multiple indexers output to be used in the same TokenEmbedder
    """

    def __init__(
        self,
        indexers: Dict[str, TokenIndexer],
        **kwargs
    ) -> None:
        super().__init__()
        self._indexers = indexers

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> IndexedTokenList:
        res = {k: v.tokens_to_indices(tokens=tokens, vocabulary=vocabulary)
               for k, v in self._indexers.items()}
        return flatten_dict(res)

    # @overrides
    # def indices_to_tokens(self, indexed_tokens: IndexedTokenList, vocabulary: Vocabulary) -> List[Token]:
    #     indices = nest_flatten_dict(indexed_tokens)
    #     tokens_map: Dict[str, List[Token]] = {}
    #     for k, v in self._indexers.items():
    #         try:
    #             tokens_map[k] = v.indices_to_tokens(indices[k])
    #         except:
    #             pass
    #     if not tokens_map:
    #         raise NotImplementedError
    #     if len(tokens_map) == 1:
    #         return list(tokens_map.values())[0]
    #     res = []
    #     for tokens in zip(*tokens_map.values()):
    #         t = merge_dataclasses(list(tokens), Token)
    #         res.append(t)
    #     return res


    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        res = {k: v.get_empty_token_list()
               for k, v in self._indexers.items()}
        return flatten_dict(res)

    @overrides
    def as_padded_tensor_dict(
            self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        res = {k: v.as_padded_tensor_dict(nest_flatten_dict(tokens)[k], nest_flatten_dict(padding_lengths)[k])
               for k, v in self._indexers.items()}
        return flatten_dict(res)
