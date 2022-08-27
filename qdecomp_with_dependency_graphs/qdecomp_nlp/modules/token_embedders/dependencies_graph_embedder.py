import torch
from allennlp.common.checks import ConfigurationError
from allennlp.models import Model
from allennlp.modules import Embedding

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

from qdecomp_nlp.models.dependencies_graph.biaffine_graph_parser import GraphParser
from qdecomp_with_dependency_graphs.utils.data_structures import nest_flatten_dict
from qdecomp_with_dependency_graphs.utils.modules import capture_model_internals


@TokenEmbedder.register("dependencies_graph_embedder")
class DependenciesGraphTokenEmbedder(TokenEmbedder):
    """
    A TokenEmbedder that uses a pretrained GraphParser to encode graphical information.
    Should be used with `DependenciesGraphTokenIndexer`
    """

    def __init__(
        self,
        model: Model,
        arc_tags_embedding_dim: int = None,
        arc_tags_namespace: str = 'labels',
        original_sequence_mask_keyword: str = 'mask',
        append_weighted_neighbors: bool = True,
        use_ff_encodings: bool = False,
        separate_head_and_child: bool = False,
    ) -> None:
        """
        Embed tokens using pretrained GraphParser
        :param model: pretrained GraphParser model
        :param arc_tags_embedding_dim: tags embedding dim
        :param arc_tags_namespace: tags namespace (should be correlated with model)
        :param original_sequence_mask_keyword: index key for original tokens (exclude extra special tokens) mask
            Should be correlated with DependenciesGraphTokenIndexer.original_sequence_mask_keyword
        """
        super().__init__()
        self._model: GraphParser = model
        self._original_sequence_mask_keyword = original_sequence_mask_keyword

        self._append_weighted_neighbors = append_weighted_neighbors
        self._use_ff_encodings = use_ff_encodings
        self._separate_head_and_child = separate_head_and_child
        if not self._append_weighted_neighbors and arc_tags_embedding_dim:
            raise ConfigurationError('arc_tags_embedding_dim is allowed only when append_weighted_neighbors is True')
        if self._separate_head_and_child and not (self._use_ff_encodings and self._append_weighted_neighbors):
            raise ConfigurationError('separate_head_and_child is allowed only for use_ff_encodings=True and append_weighted_neighbors=True')

        # set dependencies embeddings
        if self._append_weighted_neighbors:
            self._arc_tags_embedder = Embedding(embedding_dim=arc_tags_embedding_dim,
                                                vocab_namespace=arc_tags_namespace,
                                                vocab=self._model.vocab)
            self._arc_tags_number = self._model.vocab.get_vocab_size(arc_tags_namespace)

        # calculate output dim
        if self._use_ff_encodings:
            # encoding[i] = [FF_arc_h[i]+FF_arc_c[i] ; FF_tag_h[i]+FF_tag_c[i]
            encoding_dim = sum([
                self._model.head_arc_feedforward.get_output_dim(),
                # self._model.child_arc_feedforward.get_output_dim(),
                self._model.head_tag_feedforward.get_output_dim(),
                # self._model.child_tag_feedforward.get_output_dim(),
            ])
        else:
            encoding_dim = self._model.encoder.get_output_dim()

        if self._append_weighted_neighbors:
            # [encoding[i]; in_embedding[i]; out_embedding[i]]
            # in_embedding[i] = sum_j [encoding[j]; arc_tags_embeddings[j,i]]
            self._output_dim = encoding_dim + 2*(encoding_dim + arc_tags_embedding_dim)
        else:
            self._output_dim = encoding_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, *input, **kwargs) -> torch.Tensor:
        original_sequence_mask: torch.BoolTensor = kwargs.pop(self._original_sequence_mask_keyword)

        tokens = nest_flatten_dict(kwargs)
        pos_tags = tokens.pop('pos_tags').get('tokens', None) if 'pos_tags' in tokens else None

        # Run model and get encodings, arcs probabilities and tags probabilities
        if self._use_ff_encodings:
            modules = ['head_arc_feedforward', 'child_arc_feedforward', 'head_tag_feedforward', 'child_tag_feedforward']
        else:
            modules = ['encoder']
        with capture_model_internals(self._model, '|'.join(modules)) as internals:
            output_dict = self._model.forward(*input, tokens=tokens, pos_tags=pos_tags)

        if self._use_ff_encodings:
            head_encodings = torch.cat([internals['head_arc_feedforward'], internals['head_tag_feedforward']], dim=-1)
            child_encodings = torch.cat([internals['child_arc_feedforward'], internals['child_tag_feedforward']], dim=-1)
            encodings = head_encodings + child_encodings
        else:
            head_encodings, child_encodings = None, None
            encodings = internals['encoder']

        # construct embeddings
        if self._append_weighted_neighbors:
            arc_probs = output_dict["arc_probs"]
            arc_tags_probs = output_dict['arc_tag_probs']

            # nan to zero (avoid nan-loss)
            arc_probs[arc_probs.isnan()] = 0
            arc_tags_probs[arc_tags_probs.isnan()] = 0
            # avoid in-place modifications for backprop
            # arc_tags_probs = torch.where(torch.isnan(arc_tags_probs), torch.zeros_like(arc_tags_probs), arc_tags_probs)

            if not self._separate_head_and_child:
                head_encodings, child_encodings = None, None
            embeddings = self._weighted_neighbors_embeddings(encodings, arc_probs, arc_tags_probs,
                                                             head_encodings, child_encodings)
        else:
            # use just the basic encodings
            embeddings = encodings

        # trim to original size
        max_length = original_sequence_mask.sum(-1).max()
        embeddings = embeddings[:, :max_length, :] * original_sequence_mask.unsqueeze(-1)

        return embeddings

    def _weighted_neighbors_embeddings(self, encodings: torch.Tensor,
                                       arc_probs: torch.Tensor, arc_tags_probs: torch.Tensor,
                                       head_encodings: torch.Tensor = None, child_encodings: torch.Tensor =None,
                                       ) -> torch.Tensor:
        batch_size, nodes_num, *_ = arc_probs.size()
        # arc tags embeddings
        # (tags number, embedding dim)
        tags_embeddings = self._arc_tags_embedder(torch.arange(0, self._arc_tags_number, dtype=torch.long,
                                                               device=arc_probs.device))
        # (batch size, tokens, tokens, tag_embedding_dim)
        arc_tags_embeddings = arc_tags_probs.matmul(tags_embeddings)

        # exp_head_enc[i,j]: head_encoding[i]
        # exp_child_enc[i,j]: child_encoding[j]
        # (batch size, tokens, tokens, encoding_dim)
        head_encodings = head_encodings if head_encodings is not None else encodings
        child_encodings = child_encodings if child_encodings is not None else encodings
        expanded_head_encodings = head_encodings.unsqueeze(dim=2).expand(-1, -1, nodes_num, -1)
        expanded_child_encodings = child_encodings.unsqueeze(dim=1).expand(-1, nodes_num, -1, -1)

        # in-arcs embeddings
        # (i,j): [head_encoding[i]; arc_tags_embeddings[i,j]]
        # (batch size, tokens, tokens, encoding_dim + tag_embedding_dim)
        in_arc_tag_embeddings = torch.cat([expanded_head_encodings, arc_tags_embeddings], dim=-1)
        # (i): sum_j (arc_probs[j,i] * in_arc_tag_embeddings[j,i])
        # (batch size, tokens, encoding_dim + tag_embedding_dim)
        in_embeddings = (arc_probs.unsqueeze(-1) * in_arc_tag_embeddings).transpose(1, 2).sum(dim=-2)

        # out-arcs embeddings
        # (i,j): [child_encoding[j]; arc_tags_embeddings[i,j]]
        # (batch size, tokens, tokens, encoding_dim + tag_embedding_dim)
        out_arc_tag_embeddings = torch.cat([expanded_child_encodings, arc_tags_embeddings], dim=-1)
        # (i): sum_j (arc_probs[i,j] * in_arc_tag_embeddings[i,j])
        # (batch size, tokens, encoding_dim + tag_embedding_dim)
        out_embeddings = (arc_probs.unsqueeze(-1) * out_arc_tag_embeddings).sum(dim=-2)

        # (i): [encoding[i]; in_embeddings[i]; out_embeddings[i]]
        # (batch size, tokens, encoding_dim + 2(encoding_dim + arc_tags_embedding_dim)
        embeddings = torch.cat([encodings, in_embeddings, out_embeddings], dim=-1)

        return embeddings
