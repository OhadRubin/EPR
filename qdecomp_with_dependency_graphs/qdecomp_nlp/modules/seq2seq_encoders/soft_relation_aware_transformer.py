from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward, TokenEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, FeedForwardEncoder
import torch
from allennlp.nn import Activation

import qdecomp_nlp.modules.seq2seq_encoders.rat_sql_transformer_wrapper as transformer
from overrides import overrides


@Seq2SeqEncoder.register("soft_relation_aware_transformer")
class SoftRelationAwareTransformer(Seq2SeqEncoder):
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
            lambda: EncoderLayer(
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
    def forward(self, enc: torch.Tensor, relation_probs: torch.Tensor, mask: torch.BoolTensor):
        return self.encoder(enc, relation_probs, mask)


#############

class EncoderLayer(transformer.EncoderLayer):
    @overrides
    def forward(self, x, relation: torch.FloatTensor, mask):
        relation_k = torch.matmul(relation, self.relation_k_emb.weight)
        relation_v = torch.matmul(relation, self.relation_v_emb.weight)

        x = self.sublayer[0](
            x, lambda x: self.self_attn(x, x, x, relation_k, relation_v, mask)
        )
        return self.sublayer[1](x, self.feed_forward)
