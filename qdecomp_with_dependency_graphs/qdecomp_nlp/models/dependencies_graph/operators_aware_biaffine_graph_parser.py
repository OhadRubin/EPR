"""
Based on allennlp_models/structured_prediction/models/graph_parser.py
tag: v1.1.0
"""
from typing import Dict, Tuple, Any, List
import logging
import copy
from enum import Enum

import os
from overrides import overrides
import torch
from torch.nn.modules import Dropout
import numpy as np

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, Activation
from allennlp.nn.util import min_value_of_dtype, masked_softmax
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import Metric, F1Measure, FBetaMeasure, BooleanAccuracy, CategoricalAccuracy

from qdecomp_with_dependency_graphs.dependencies_graph.check_frequency__structural_constraints import get_operator

logger = logging.getLogger(__name__)


class DecodeStrategy(str, Enum):
    OPERATORS_MASK = 'operators_mask'
    PROBS_MULTIPLY = 'probs_multiplication'
    NONE = 'none'


@Model.register("operators_aware_biaffine_graph_parser")
class OperatorsAwareGraphParser(Model):
    """
    A Parser for arbitrary graph structures.
    Registered as a `Model` with name "graph_parser".
    # Parameters
    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the `tokens` `TextField` we get as input to the model.
    encoder : `Seq2SeqEncoder`
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    tag_representation_dim : `int`, required.
        The dimension of the MLPs used for arc tag prediction.
    arc_representation_dim : `int`, required.
        The dimension of the MLPs used for arc prediction.
    tag_feedforward : `FeedForward`, optional, (default = `None`).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : `FeedForward`, optional, (default = `None`).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    pos_tag_embedding : `Embedding`, optional.
        Used to embed the `pos_tags` `SequenceLabelField` we get as input to the model.
    dropout : `float`, optional, (default = `0.0`)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : `float`, optional, (default = `0.0`)
        The dropout applied to the embedded text input.
    edge_prediction_threshold : `int`, optional (default = `0.5`)
        The probability at which to consider a scored edge to be 'present'
        in the decoded graph. Must be between 0 and 1.
    tag_prediction_threshold : `int`, optional (default = 0.5)
        The probability at which to consider a scored tag to be 'present'
        in the decoded graph on multi_label=True. Must be between 0 and 1.
    arc_tags_only : `bool`, optional (default = False)
        Produce tags representations only (considers arcs with no tags as no-arc).
    multi_label : `bool`, optional (default = False)
        If True, allows multiple tags per arc, where no tag means no arc. Currently, enabled in arc_tags_only=True mode only.
        Otherwise, concatenates arc tags for a single tag.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        operator_representation_dim: int,
        tag_representation_dim: int,
        operator_feedforward: FeedForward = None,
        tag_feedforward: FeedForward = None,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        tags_namespace: str = "labels",
        operators_namespace: str = "operators_labels",
        decode_strategy: DecodeStrategy = DecodeStrategy.OPERATORS_MASK,
        operator_embeddings_dim: int = None,
        graph_based_metric: Metric = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._tags_namespace = tags_namespace
        self._operators_namespace = operators_namespace
        self._tag_none_index = self.vocab.get_token_to_index_vocabulary(self._tags_namespace).get('NONE', None)
        if self._tag_none_index is None:
            raise ConfigurationError(
                "expecting NONE tag"
            )
        self._operator_none_index = self.vocab.get_token_to_index_vocabulary(self._operators_namespace).get('NONE', None)
        if self._operator_none_index is None:
            raise ConfigurationError(
                "expecting NONE operator"
            )

        # operators mapping
        operator_to_tag = torch.tensor(
            [[(all(any(get_operator(t) == o
                       for o in operator.split('&'))
                   for t in tag.split('&')) or tag in ['NONE', 'duplicate'])
              for tag in vocab.get_token_to_index_vocabulary(self._tags_namespace)]
             for operator in vocab.get_token_to_index_vocabulary(self._operators_namespace)]
        )
        # self._operator_to_tag =
        self.register_buffer('_operator_to_tag', operator_to_tag)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        encoder_dim = encoder.get_output_dim()

        num_operators = self.vocab.get_vocab_size(self._operators_namespace)
        self._operator_feedforward = operator_feedforward or FeedForward(
            encoder_dim, 1, operator_representation_dim, Activation.by_name("elu")()
        )
        self._operator_classification_layer = torch.nn.Linear(self._operator_feedforward.get_output_dim(), num_operators)

        num_labels = self.vocab.get_vocab_size(self._tags_namespace)
        self._head_tag_feedforward = tag_feedforward or FeedForward(
            encoder_dim, 1, tag_representation_dim, Activation.by_name("elu")()
        )
        self._child_tag_feedforward = copy.deepcopy(self._head_tag_feedforward)

        self._tag_bilinear = BilinearMatrixAttention(
            tag_representation_dim, tag_representation_dim, label_dim=num_labels
        )

        self._operators_embedder = operator_embeddings_dim and Embedding(embedding_dim=operator_embeddings_dim,
                                                                         vocab=vocab, vocab_namespace=self._operators_namespace)
        if self._operators_embedder:
            mask = torch.ones(num_operators)
            mask[self._operator_none_index] = 0
            self.register_buffer('_operators_embedder_mask', mask)

        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)

        representation_dim = text_field_embedder.get_output_dim()
        check_dimensions_match(
            representation_dim,
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )
        check_dimensions_match(
            tag_representation_dim,
            self._head_tag_feedforward.get_output_dim(),
            "tag representation dim",
            "tag feedforward output dim",
        )
        check_dimensions_match(
            operator_representation_dim,
            self._operator_feedforward.get_output_dim(),
            "operator representation dim",
            "operator feedforward output dim",
        )
        check_dimensions_match(
            encoder.get_output_dim() + (self._operators_embedder.get_output_dim() if self._operators_embedder else 0),
            self._head_tag_feedforward.get_input_dim(),
            "encoder output dim + operators_embedder output dim",
            "tag feedforward input dim",
        )

        self._operators_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self._tag_loss = torch.nn.CrossEntropyLoss(reduction="none")

        self._decode_strategy = decode_strategy

        self._graph_based_metric = graph_based_metric
        # arc tag metrics
        labels_indices = [v for k, v in self.vocab.get_token_to_index_vocabulary(self._tags_namespace).items()
                          if v != self._tag_none_index]
        self._tags_f1_macro = FBetaMeasure(average='macro', labels=labels_indices)
        self._tags_f1_micro = FBetaMeasure(average='micro', labels=labels_indices)

        # operators
        labels_indices = [v for k, v in self.vocab.get_token_to_index_vocabulary(self._operators_namespace).items()
                          if v != self._operator_none_index]
        self._operators_f1_macro = FBetaMeasure(average='macro', labels=labels_indices)
        self._operators_f1_micro = FBetaMeasure(average='micro', labels=labels_indices)

        self._detailed_prediction: bool = os.environ.get('detailed') is not None
        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        metadata: List[Dict[str, Any]] = None,
        operators: torch.LongTensor = None,
        arc_tags: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters
        tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()`.
        pos_tags : `torch.LongTensor`, optional (default = `None`)
            The output of a `SequenceLabelField` containing POS tags.
        metadata : `List[Dict[str, Any]]`, optional (default = `None`)
            A dictionary of metadata for each batch element which has keys:
                tokens : `List[str]`, required.
                    The original string tokens in the sentence.
        arc_tags : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape `(batch_size, sequence_length, sequence_length)`.
        # Returns
        An output dictionary.
        """
        embedded_text_input = self._text_field_embedder(tokens)

        mask = get_text_field_mask(tokens)
        embedded_text_input = self._input_dropout(embedded_text_input)
        encoded_text = self._encoder(embedded_text_input, mask)

        encoded_text = self._dropout(encoded_text)

        # shape (batch_size, sequence_length, operator_representation_dim)
        operator_representation = self._dropout(self._operator_feedforward(encoded_text))
        operator_logits = self._operator_classification_layer(operator_representation)
        operator_probs = masked_softmax(operator_logits, mask.unsqueeze(-1))

        # concat operators embeddings for tag FF
        if self._operators_embedder:
            operators_emb = torch.matmul(operator_probs, self._operators_embedder_mask.unsqueeze(-1) * self._operators_embedder.weight)
            encoded_text = torch.cat([encoded_text, operators_emb], dim=-1)

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self._head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self._child_tag_feedforward(encoded_text))
        # shape (batch_size, num_tags, sequence_length, sequence_length)
        arc_tag_logits = self._tag_bilinear(head_tag_representation, child_tag_representation)
        # Switch to (batch_size, sequence_length, sequence_length, num_tags)
        arc_tag_logits = arc_tag_logits.permute(0, 2, 3, 1).contiguous()

        arc_tag_probs = self._greedy_decode(operator_probs, arc_tag_logits, mask)
        output_dict = {"operator_probs": operator_probs, "arc_tag_probs": arc_tag_probs, "mask": mask}

        if metadata:
            output_dict["metadata"] = metadata

        if arc_tags is not None:
            operator_nll, tag_nll = self._construct_loss(
                operator_logits=operator_logits, arc_tag_logits=arc_tag_logits,
                operators=operators, arc_tags=arc_tags, mask=mask
            )
            output_dict["loss"] = operator_nll + tag_nll
            output_dict["operator_loss"] = operator_nll
            output_dict["tag_loss"] = tag_nll

            # compute metrics on validation only
            if not self.training:
                self._operators_f1_macro(operator_probs, operators, mask)
                self._operators_f1_micro(operator_probs, operators, mask)
                tags_mask = mask.unsqueeze(-1)*mask.unsqueeze(-2)
                self._tags_f1_macro(arc_tag_probs, arc_tags, tags_mask)
                self._tags_f1_micro(arc_tag_probs, arc_tags, tags_mask)

                # graph based metric on validation only
                if self._graph_based_metric:
                    readable_dict = self.make_output_human_readable(output_dict=output_dict.copy())
                    self._graph_based_metric(
                        arcs=readable_dict['arcs'],
                        arc_tags=readable_dict['arc_tags'],
                        metadata=metadata
                    )

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        _prop_getter = output_dict.get if self._detailed_prediction else output_dict.pop
        operator_probs = _prop_getter("operator_probs").cpu().detach()
        arc_tag_probs = _prop_getter("arc_tag_probs").cpu().detach()
        mask = _prop_getter("mask")
        lengths = get_lengths_from_binary_sequence_mask(mask)
        batch_operators = []
        batch_arcs = []
        batch_arc_tags = []

        for instance_operator_probs, instance_arc_tag_probs, length in zip(
                operator_probs, arc_tag_probs, lengths
        ):
            operators = []
            arcs = []
            arc_tags = []
            for i in range(length):
                max_prob, operator_index = instance_operator_probs[i].max(-1)
                operator_index = operator_index.item()
                if max_prob == 0:
                    logger.warning("unexpected probability distribution - no operator label (max_prob=0)")
                    operator_index = self._operator_none_index
                operators.append(self.vocab.get_token_from_index(operator_index, self._operators_namespace))

                for j in range(length):
                    # Ignore the diagonal, because we don't self edges
                    if i != j:
                        max_prob, tag = instance_arc_tag_probs[i, j].max(-1)
                        tag = tag.item()
                        if max_prob == 0:
                            logger.warning("unexpected probability distribution - no tag label (max_prob=0)")
                            tag = self._tag_none_index
                        if tag == self._tag_none_index:
                            continue
                        tags = self.vocab.get_token_from_index(tag, self._tags_namespace).split('&')
                        for tag in tags:
                            arcs.append((i, j))
                            arc_tags.append(tag)
            batch_operators.append(operators)
            batch_arcs.append(arcs)
            batch_arc_tags.append(arc_tags)

        output_dict["operators"] = batch_operators
        output_dict["arcs"] = batch_arcs
        output_dict["arc_tags"] = batch_arc_tags
        return output_dict

    def _construct_loss(
        self,
        operator_logits: torch.Tensor,
        arc_tag_logits: torch.Tensor,
        operators: torch.Tensor,
        arc_tags: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for an adjacency matrix.
        # Parameters
        arc_scores : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate a
            binary classification decision for whether an edge is present between two words.
        arc_tag_logits : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length, num_tags) used to generate
            a distribution over edge tags for a given edge.
        arc_tags : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length).
            The labels for every arc.
        mask : `torch.BoolTensor`, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.
        # Returns
        arc_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc loss.
        tag_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc tag loss.
        """
        batch_size, sequence_length, num_operators = operator_logits.size()
        original_shape = [batch_size, sequence_length]
        reshaped_logits = operator_logits.view(-1, num_operators)
        reshaped_tags = operators.view(-1)
        operator_nll = self._operators_loss(reshaped_logits, reshaped_tags).view(original_shape) * mask

        # Make the arc tags not have negative values anywhere
        # (by default, no edge is indicated with -1)
        arc_tags = arc_tags * (arc_tags != -1).float()
        # We want the mask for the tags to only include the unmasked words
        # and we only care about the loss with respect to the gold arcs.
        tag_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        batch_size, sequence_length, _, num_tags = arc_tag_logits.size()
        original_shape = [batch_size, sequence_length, sequence_length]
        reshaped_logits = arc_tag_logits.view(-1, num_tags)
        reshaped_tags = arc_tags.view(-1)
        tag_nll = self._tag_loss(reshaped_logits, reshaped_tags.long()).view(original_shape) * tag_mask

        operator_nll = operator_nll.sum() / mask.sum().float()
        tag_nll = tag_nll.sum() / tag_mask.sum().float()
        return operator_nll, tag_nll

    def _greedy_decode(
        self, operator_probs: torch.Tensor, arc_tag_logits: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently.
        # Parameters
        arc_scores : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        arc_tag_logits : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length, num_tags) used to
            generate a distribution over tags for each arc.
        mask : `torch.BoolTensor`, required.
            A mask of shape (batch_size, sequence_length).
        # Returns
        arc_probs : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, sequence_length) representing the
            probability of an arc being present for this edge.
        arc_tag_probs : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, sequence_length, sequence_length)
            representing the distribution over edge tags for a given edge.
        """
        # Mask the diagonal, because we don't self edges.
        diagonal_mask = torch.eye(mask.size(1)).to(arc_tag_logits)
        arcs_mask = mask.unsqueeze(1)*mask.unsqueeze(2)

        arc_tags_mask = torch.ones_like(arc_tag_logits)
        arc_tags_mask = arc_tags_mask * (1-diagonal_mask).unsqueeze(0).unsqueeze(-1)
        arc_tags_mask[:, :, :, self._tag_none_index] = 1
        arc_tags_mask = arc_tags_mask * arcs_mask.unsqueeze(-1)

        # apply decode strategy
        batch_size, sequence_length, operators_num = operator_probs.size()
        _, _, _, tags_num = arc_tag_logits.size()
        if self._decode_strategy == DecodeStrategy.OPERATORS_MASK:
            operators = torch.argmax(operator_probs, dim=-1)
            operators_mask = self._operator_to_tag[operators.view(-1), :].view([batch_size, sequence_length, tags_num])
            arc_tags_mask = arc_tags_mask * (operators_mask.unsqueeze(-2))
        elif self._decode_strategy == DecodeStrategy.PROBS_MULTIPLY:
            # calculate initial probs
            arc_tag_probs_ = masked_softmax(arc_tag_logits, mask=arc_tags_mask, dim=-1)
            # update logits by prob multiplication
            arc_tag_logits = arc_tag_probs_ * torch.matmul(operator_probs, self._operator_to_tag.float()).unsqueeze(-2)

        # shape (batch_size, sequence_length, sequence_length, num_tags)
        arc_tag_probs = masked_softmax(arc_tag_logits, mask=arc_tags_mask, dim=-1)
        return arc_tag_probs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        if not self.training:
            operators_f1_macro = {f'operators_macro_{k}': v for k, v in self._operators_f1_macro.get_metric(reset).items()}
            operators_f1_micro = {f'operators_micro_{k}': v for k, v in self._operators_f1_micro.get_metric(reset).items()}
            tags_f1_macro = {f'tags_macro_{k}': v for k, v in self._tags_f1_macro.get_metric(reset).items()}
            tags_f1_micro = {f'tags_micro_{k}': v for k, v in self._tags_f1_micro.get_metric(reset).items()}
            metrics.update({**operators_f1_macro, **operators_f1_micro, **tags_f1_macro, **tags_f1_micro})

            if self._graph_based_metric:
                metrics.update(self._graph_based_metric.get_metric(reset=reset))

        return metrics
