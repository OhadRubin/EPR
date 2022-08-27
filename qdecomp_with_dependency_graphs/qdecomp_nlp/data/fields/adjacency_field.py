from itertools import product

import torch
from allennlp.data import Vocabulary
from overrides import overrides
from allennlp.data.fields import AdjacencyField
from typing import Dict


class CustomAdjacencyField(AdjacencyField):
    def __init__(self,
                 fill_with_none: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._none_label = fill_with_none and 'NONE'

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._indexed_labels is None and self.labels is not None:
            for label in self.labels:
                counter[self._label_namespace][label] += 1  # type: ignore

            if self._none_label:
                tokens_count = self.sequence_field.sequence_length()
                indices_ = set(self.indices)
                for i, j in product(range(tokens_count), range(tokens_count)):
                    if (i, j) not in indices_:
                        counter[self._label_namespace][self._none_label] += 1

    @overrides
    def index(self, vocab: Vocabulary):
        if self._none_label:
            self._padding_value = vocab.get_token_index(self._none_label, self._label_namespace) if self.labels else 1
        super().index(vocab=vocab)

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        if self._none_label and self.labels is None:
            desired_num_tokens = padding_lengths["num_tokens"]
            return torch.ones(desired_num_tokens, desired_num_tokens)
        return super().as_tensor(padding_lengths=padding_lengths)
