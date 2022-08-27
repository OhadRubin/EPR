from allennlp.modules import FeedForward
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, FeedForwardEncoder
import torch
from allennlp.nn import Activation

import qdecomp_nlp.modules.seq2seq_encoders.rat_sql_transformer_wrapper as transformer
from overrides import overrides


@Seq2SeqEncoder.register("latent_relation_aware_transformer")
class LatentRelationAwareTransformer(Seq2SeqEncoder):
    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            hidden_size: int,
            ff_size: int,
            dropout: float = 0.1,
            tie_layers: bool = False,
            relation_k_encoder: Seq2SeqEncoder = None,
            relation_v_encoder: Seq2SeqEncoder = None,
    ):
        super().__init__()
        self.encoder = Encoder(
            lambda: EncoderLayer(
                size=hidden_size,
                self_attn=transformer.MultiHeadedAttentionWithRelations(
                    num_heads, hidden_size, dropout
                ),
                feed_forward=transformer.PositionwiseFeedForward(hidden_size, ff_size, dropout),
                dropout=dropout,
                relation_encoders_outdim=hidden_size//num_heads,
                relation_k_encoder=relation_k_encoder,
                relation_v_encoder=relation_v_encoder
            ),
            hidden_size,
            num_layers,
            tie_layers,
        )
        self._input_dim = hidden_size
        self.tie_layers = tie_layers

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
    def forward(self, enc: torch.Tensor, mask: torch.BoolTensor, return_relations=False):
        return self.encoder(enc, mask, return_relations=return_relations)


#############

class Encoder(transformer.Encoder):
    @overrides
    def forward(self, x, mask, return_relations=False):
        batch_size, n, d = x.size()
        x_pairs = x.unsqueeze(-2).expand(-1, -1, n, -1) - x.unsqueeze(-3).expand(-1, n, -1, -1)
        relation_k, relation_v = [], []
        for layer in self.layers:
            x, r_k, r_v = layer(x, x_pairs, mask)
            relation_k.append(r_k)
            relation_v.append(r_v)
        res = self.norm(x)
        if return_relations:
            if self.tie_layers:
                # encodings should be the same for all layers (shared x_pairs)
                return res, [relation_k[0]], [relation_v[0]]
            return res, relation_k, relation_v
        return res


class EncoderLayer(transformer.EncoderLayer):
    def __init__(self,
                 size: int,
                 relation_encoders_outdim: int = None,
                 relation_k_encoder: Seq2SeqEncoder = None,
                 relation_v_encoder: Seq2SeqEncoder = None,
                 **kwargs
                 ):
        relation_k_encoder = relation_k_encoder or FeedForwardEncoder(
            FeedForward(input_dim=size, num_layers=1, hidden_dims=relation_encoders_outdim, activations=Activation.by_name("relu")())
        )
        relation_v_encoder = relation_v_encoder or FeedForwardEncoder(
            FeedForward(input_dim=size, num_layers=1, hidden_dims=relation_encoders_outdim, activations=Activation.by_name("relu")())
        )
        super().__init__(**kwargs,
                         size=size,
                         relation_k_embedder=relation_k_encoder,
                         relation_v_embedder=relation_v_encoder)

    @overrides
    def forward(self, x, x_pairs, mask):
        batch_size, n, d = x.size()
        mask = mask.unsqueeze(-1)*mask.unsqueeze(-2)
        # x_pairs = torch.cat([x.unsqueeze(-2).expand(-1, -1, n, -1),
        #                      x.unsqueeze(-3).expand(-1, n, -1, -1)], dim=-1)
        # x_pairs = original_x.unsqueeze(-2).expand(-1, -1, n, -1) - original_x.unsqueeze(-3).expand(-1, n, -1, -1)
        relation_k = self.relation_k_emb(x_pairs.view(batch_size, -1, d)).view(batch_size, n, n, -1)
        relation_v = self.relation_v_emb(x_pairs.view(batch_size, -1, d)).view(batch_size, n, n, -1)

        x = self.sublayer[0](
            x, lambda x: self.self_attn(x, x, x, relation_k, relation_v, mask)
        )
        return self.sublayer[1](x, self.feed_forward), relation_k, relation_v
