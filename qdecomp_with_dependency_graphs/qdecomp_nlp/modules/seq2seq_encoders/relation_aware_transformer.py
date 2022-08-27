"""
Based on https://github.com/microsoft/rat-sql/blob/master/ratsql/models/spider/spider_enc_modules.py
RelationalTransformerUpdate
"""
from allennlp.data import Vocabulary
from allennlp.modules import TokenEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from torch import nn
import torch
import qdecomp_nlp.modules.seq2seq_encoders.rat_sql_transformer_wrapper as transformer
from overrides import overrides


@Seq2SeqEncoder.register("relation_aware_transformer")
class RelationAwareTransformer(Seq2SeqEncoder):
    def __init__(
        self,
        vocab: Vocabulary,
        relations_namespace: str,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        ff_size: int,
        dropout: float = 0.1,
        tie_layers: bool = False,
        relation_k_embedder: TokenEmbedder = None,
        relation_v_embedder: TokenEmbedder = None,
    ):
        super().__init__()
        n_relations: int = vocab.get_vocab_size(relations_namespace)
        self.encoder = transformer.Encoder(
            lambda: transformer.EncoderLayer(
                hidden_size,
                transformer.MultiHeadedAttentionWithRelations(
                    num_heads, hidden_size, dropout
                ),
                transformer.PositionwiseFeedForward(hidden_size, ff_size, dropout),
                dropout,
                num_relation_kinds=n_relations,
                relation_k_embedder=relation_k_embedder,
                relation_v_embedder=relation_v_embedder
            ),
            hidden_size,
            num_layers,
            tie_layers,
        )
        self._input_dim = hidden_size

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._input_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self, enc: torch.Tensor, relation: torch.Tensor, mask: torch.BoolTensor):
        return self.encoder(enc, relation, mask)
