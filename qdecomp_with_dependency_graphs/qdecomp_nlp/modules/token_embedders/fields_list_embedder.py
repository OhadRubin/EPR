from typing import List

import torch
from allennlp.data import Vocabulary
from overrides import overrides
from allennlp.modules import TokenEmbedder, Embedding
from torch.nn import ModuleList


@TokenEmbedder.register("fields_list")
class FieldsListTokenEmbedder(TokenEmbedder):
    """
    Embedder for ListField like output tensor, where multiple semantically different tensors are stacked to a
    single input.
    Allocate a separated Embedding for each part, using the given namespaces.
    Final embeddings are either a sum of the parts embeddings (default), or a concatenation of them.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 embedding_dim: int,
                 namespaces: List[str],
                 concat_parts: bool = False,
                 **kwargs):
        super().__init__()
        self._parts_num = len(namespaces)
        assert self._parts_num > 0
        self._embedding_dim = embedding_dim
        self._concat_parts = concat_parts
        if concat_parts:
            # make sure the embeddings dim will sum up to embedding_dim
            emb_dim = [embedding_dim//self._parts_num]*(self._parts_num-1) + \
                      [embedding_dim//self._parts_num + embedding_dim % self._parts_num]
        else:
            emb_dim = [embedding_dim]*self._parts_num
        self._embeddings: ModuleList = ModuleList([
            Embedding(embedding_dim=emb_dim[i], vocab_namespace=namespaces[i], vocab=vocab, **kwargs)
            for i in range(self._parts_num)
        ])

    def get_output_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def forward(self, input: torch.Tensor):
        batch_size, parts, *_ = input.size()
        mask = input != -1
        input = input*mask
        embeddings = [emb(input[:, i].contiguous())*mask[:, i].unsqueeze(-1) for i, emb in enumerate(self._embeddings)]
        if self._concat_parts:
            joint_embeddings = torch.cat(embeddings, dim=-1)
        else:
            joint_embeddings = sum(embeddings)
        return joint_embeddings
