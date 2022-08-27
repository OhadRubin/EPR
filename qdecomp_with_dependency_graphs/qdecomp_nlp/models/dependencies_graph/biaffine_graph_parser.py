"""
Based on allennlp_models/structured_prediction/models/graph_parser.py
tag: v1.1.0
"""
from typing import Dict, Tuple, Any, List
import logging
import copy

import os
from overrides import overrides
import torch
from torch.nn.modules import Dropout
import numpy

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

logger = logging.getLogger(__name__)


@Model.register("biaffine_graph_parser")
class GraphParser(Model):
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
        tag_representation_dim: int,
        arc_representation_dim: int,
        tag_feedforward: FeedForward = None,
        arc_feedforward: FeedForward = None,
        pos_tag_embedding: Embedding = None,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        edge_prediction_threshold: float = 0.5,
        tag_prediction_threshold: float = 0.5,
        arc_tags_only: bool = False,
        multi_label: bool = False,
        graph_based_metric: Metric = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.edge_prediction_threshold: float = edge_prediction_threshold
        if not 0 < edge_prediction_threshold < 1:
            raise ConfigurationError(
                f"edge_prediction_threshold must be between "
                f"0 and 1 (exclusive) but found {edge_prediction_threshold}."
            )
        self.tag_prediction_threshold: float = tag_prediction_threshold
        if not 0 < tag_prediction_threshold < 1:
            raise ConfigurationError(
                f"tag_prediction_threshold must be between "
                f"0 and 1 (exclusive) but found {tag_prediction_threshold}."
            )

        if (not arc_tags_only) and multi_label:
            raise ConfigurationError(
                f"multi_label are allowed only when arc_tags_only is True."
            )
        self.multi_label = multi_label
        self.arc_tags_only = arc_tags_only
        if arc_tags_only:
            self.edge_prediction_threshold = -1

        encoder_dim = encoder.get_output_dim()

        self.head_arc_feedforward = arc_feedforward or FeedForward(
            encoder_dim, 1, arc_representation_dim, Activation.by_name("elu")()
        )
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(
            arc_representation_dim, arc_representation_dim, use_input_biases=True
        )

        num_labels = self.vocab.get_vocab_size("labels")
        self.head_tag_feedforward = tag_feedforward or FeedForward(
            encoder_dim, 1, tag_representation_dim, Activation.by_name("elu")()
        )
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = BilinearMatrixAttention(
            tag_representation_dim, tag_representation_dim, label_dim=num_labels
        )

        self._pos_tag_embedding = pos_tag_embedding or None
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)

        representation_dim = text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += pos_tag_embedding.get_output_dim()

        check_dimensions_match(
            representation_dim,
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )
        check_dimensions_match(
            tag_representation_dim,
            self.head_tag_feedforward.get_output_dim(),
            "tag representation dim",
            "tag feedforward output dim",
        )
        check_dimensions_match(
            arc_representation_dim,
            self.head_arc_feedforward.get_output_dim(),
            "arc representation dim",
            "arc feedforward output dim",
        )

        self._none_index = self.vocab.get_token_to_index_vocabulary('labels').get('NONE', -1)
        if (self.arc_tags_only and not self.multi_label) and self._none_index == -1:
            raise ConfigurationError(
                "arc_tags_only=True and multi_label=False, but no NONE label in arcs labels vocabulary"
            )
        if (not (self.arc_tags_only and not self.multi_label)) and self._none_index != -1:
            raise ConfigurationError(
                "NONE label is allowed just on arc_tags_only=True, multi_label=False"
            )

        self._arc_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self._tag_loss = torch.nn.BCEWithLogitsLoss(reduction="none") if self.multi_label \
            else torch.nn.CrossEntropyLoss(reduction="none")

        self._arcs_f1 = F1Measure(positive_label=1)
        self._arcs_accuracy = CategoricalAccuracy()
        self._arcs_exact_match = BooleanAccuracy()

        if self.multi_label:
            self._arcs_and_tags_f1 = F1Measure(positive_label=1)
        else:
            labels_indices = [v for k, v in self.vocab.get_token_to_index_vocabulary('labels').items()
                              if v != self._none_index]
            self._arcs_and_tags_f1_macro = FBetaMeasure(average='macro', labels=labels_indices)
            self._arcs_and_tags_f1_micro = FBetaMeasure(average='micro', labels=labels_indices)

        self._graph_based_metric = graph_based_metric

        self._detailed_prediction: bool = os.environ.get('detailed') is not None
        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        pos_tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        arc_indices: torch.LongTensor = None,
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
        embedded_text_input = self.text_field_embedder(tokens)
        if pos_tags is not None and self._pos_tag_embedding is not None:
            embedded_pos_tags = self._pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        elif self._pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        mask = get_text_field_mask(tokens)
        embedded_text_input = self._input_dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)

        encoded_text = self._dropout(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        child_arc_representation = self._dropout(self.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self.child_tag_feedforward(encoded_text))
        # shape (batch_size, sequence_length, sequence_length)
        arc_scores = self.arc_attention(head_arc_representation, child_arc_representation)
        # shape (batch_size, num_tags, sequence_length, sequence_length)
        arc_tag_logits = self.tag_bilinear(head_tag_representation, child_tag_representation)
        # Switch to (batch_size, sequence_length, sequence_length, num_tags)
        arc_tag_logits = arc_tag_logits.permute(0, 2, 3, 1).contiguous()

        # Since we'll be doing some additions, using the min value will cause underflow
        minus_mask = ~mask * min_value_of_dtype(arc_scores.dtype) / 10
        arc_scores = arc_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        arc_probs, arc_tag_probs = self._greedy_decode(arc_scores, arc_tag_logits, mask)

        output_dict = {"arc_probs": arc_probs, "arc_tag_probs": arc_tag_probs, "mask": mask}

        if metadata:
            output_dict["metadata"] = metadata

        if arc_tags is not None:
            arc_indices = (arc_indices == 1).float()  # padding_value => 0
            arc_nll, tag_nll = self._construct_loss(
                arc_scores=arc_scores, arc_tag_logits=arc_tag_logits,
                arc_indices=arc_indices, arc_tags=arc_tags, mask=mask
            )
            output_dict["loss"] = arc_nll + tag_nll
            output_dict["arc_loss"] = arc_nll
            output_dict["tag_loss"] = tag_nll

            # compute metrics on validation only
            if not self.training:
                arc_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
                # Ignore the diagonal, because we don't self edges
                arc_mask = (arc_mask * (1-torch.diag(arc_mask.new_ones(arc_mask.size(-1)).int()))).bool()
                tag_mask = arc_mask

                batch_size, sequence_length, _, num_tags = arc_tag_probs.size()
                if self.multi_label:
                    # predict an arc if any tag exists
                    arc_tags_count = torch.sum((arc_tag_probs > self.tag_prediction_threshold), dim=-1)
                    arc_indices_pred = arc_tags_count > 0
                    arc_probs = arc_indices_pred.float()
                    # replace -1 of padding with 0
                    arc_tags = arc_tags * (arc_tags != -1).float()
                elif self.arc_tags_only:
                    arc_indices_pred = (arc_tag_probs.max(dim=-1)[1] != self._none_index)
                    arc_probs = arc_indices_pred.float()
                else:
                    arc_indices_pred = arc_probs > self.edge_prediction_threshold
                    pred_arcs = arc_indices_pred.float().unsqueeze(-1)
                    # replace -1 of none arcs with num_tags
                    arc_tags = arc_tags * arc_indices + num_tags * (~arc_indices.bool())
                    # extend classes probabilities - last class is none
                    arc_tag_probs = torch.cat([arc_tag_probs * pred_arcs, 1 - pred_arcs], -1)

                # We stack scores here because the f1 measure expects a
                # distribution, rather than a single value.
                one_minus_arc_probs = 1 - arc_probs
                self._arcs_f1(torch.stack([one_minus_arc_probs, arc_probs], -1), arc_indices, arc_mask)
                self._arcs_accuracy(torch.stack([one_minus_arc_probs, arc_probs], -1), arc_indices, arc_mask)
                self._arcs_exact_match(arc_indices_pred, arc_indices, arc_mask)

                if self.multi_label:
                    # We stack scores here because the f1 measure expects distribution
                    arc_tag_probs = torch.stack([1 - arc_tag_probs, arc_tag_probs], -1)
                    tag_mask = tag_mask.unsqueeze(-1).expand(-1, -1, -1, num_tags)
                    self._arcs_and_tags_f1(arc_tag_probs, arc_tags, tag_mask)
                else:
                    self._arcs_and_tags_f1_macro(arc_tag_probs, arc_tags, tag_mask)
                    self._arcs_and_tags_f1_micro(arc_tag_probs, arc_tags, tag_mask)

        # graph based metric on validation only
        if not self.training and self._graph_based_metric:
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
        arc_tag_probs = _prop_getter("arc_tag_probs").cpu().detach().numpy()
        arc_probs = _prop_getter("arc_probs").cpu().detach().numpy()
        mask = _prop_getter("mask")
        lengths = get_lengths_from_binary_sequence_mask(mask)
        arcs = []
        arc_tags = []
        for instance_arc_probs, instance_arc_tag_probs, length in zip(
            arc_probs, arc_tag_probs, lengths
        ):

            arc_matrix = instance_arc_probs > self.edge_prediction_threshold
            edges = []
            edge_tags = []
            for i in range(length):
                for j in range(length):
                    # Ignore the diagonal, because we don't self edges
                    if arc_matrix[i, j] == 1 and i != j:
                        if self.multi_label:
                            tags_indices, = numpy.where(instance_arc_tag_probs[i, j] > self.tag_prediction_threshold)
                        else:
                            tags_indices = [instance_arc_tag_probs[i, j].argmax(-1)]
                        tags = [x for tag_index in tags_indices
                                for x in self.vocab.get_token_from_index(tag_index, "labels").split('&')
                                if tag_index != self._none_index]
                        for tag in tags:
                            edges.append((i, j))
                            edge_tags.append(tag)
            arcs.append(edges)
            arc_tags.append(edge_tags)

        output_dict["arcs"] = arcs
        output_dict["arc_tags"] = arc_tags
        return output_dict

    def _construct_loss(
        self,
        arc_scores: torch.Tensor,
        arc_tag_logits: torch.Tensor,
        arc_indices: torch.Tensor,
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
        # ignore arcs part on tags_only mode
        arc_nll = arc_indices.new_zeros(1) if self.arc_tags_only \
            else (self._arc_loss(arc_scores, arc_indices) * mask.unsqueeze(1) * mask.unsqueeze(2))

        # Make the arc tags not have negative values anywhere
        # (by default, no edge is indicated with -1).
        arc_tags = arc_tags * (arc_tags != -1).float()
        # We want the mask for the tags to only include the unmasked words
        # and we only care about the loss with respect to the gold arcs.
        tag_mask = mask.unsqueeze(1) * mask.unsqueeze(2) if self.arc_tags_only \
            else mask.unsqueeze(1) * mask.unsqueeze(2) * arc_indices
        if self.multi_label:
            tag_nll = self._tag_loss(arc_tag_logits, arc_tags) * tag_mask.unsqueeze(-1)
        else:
            batch_size, sequence_length, _, num_tags = arc_tag_logits.size()
            original_shape = [batch_size, sequence_length, sequence_length]
            reshaped_logits = arc_tag_logits.view(-1, num_tags)
            reshaped_tags = arc_tags.view(-1)
            tag_nll = (
                self._tag_loss(reshaped_logits, reshaped_tags.long()).view(original_shape) * tag_mask
            )

        valid_positions = tag_mask.sum()

        arc_nll = arc_nll.sum() / valid_positions.float()
        tag_nll = tag_nll.sum() / valid_positions.float()
        return arc_nll, tag_nll

    def _greedy_decode(
        self, arc_scores: torch.Tensor, arc_tag_logits: torch.Tensor, mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        diagonal_mask = torch.eye(mask.size(1)).to(arc_scores)
        arcs_mask = mask.unsqueeze(1)*mask.unsqueeze(2)
        if self.arc_tags_only:
            arc_tags_mask = torch.ones_like(arc_tag_logits)
            arc_tags_mask = arc_tags_mask * (1-diagonal_mask).unsqueeze(0).unsqueeze(-1)
            arc_tags_mask[:, :, :, self._none_index] = 1
            arc_tags_mask = arc_tags_mask * arcs_mask.unsqueeze(-1)
        else:
            arcs_mask = arcs_mask * (1-diagonal_mask)
            arc_tags_mask = arcs_mask.unsqueeze(-1)

        # shape (batch_size, sequence_length, sequence_length)
        arc_probs = arc_scores.sigmoid() * arcs_mask
        # shape (batch_size, sequence_length, sequence_length, num_tags)
        arc_tag_probs = arc_tag_logits.sigmoid() * arc_tags_mask if self.multi_label \
            else masked_softmax(arc_tag_logits, mask=arc_tags_mask, dim=-1)

        return arc_probs, arc_tag_probs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        if not self.training:
            try:
                metrics.update({f'arcs_{k}': v for k, v in self._arcs_f1.get_metric(reset).items()})

                metrics['arcs_exact_match'] = self._arcs_exact_match.get_metric(reset)
                metrics['arcs_accuracy'] = self._arcs_accuracy.get_metric(reset)

                if self.multi_label:
                    metrics.update({f'arcs_and_tags_{k}': v for k, v in self._arcs_and_tags_f1.get_metric(reset).items()})
                else:
                    arcs_and_tags_f1_macro = {f'arcs_and_tags_macro_{k}': v for k, v in self._arcs_and_tags_f1_macro.get_metric(reset).items()}
                    arcs_and_tags_f1_micro = {f'arcs_and_tags_micro_{k}': v for k, v in self._arcs_and_tags_f1_micro.get_metric(reset).items()}

                    metrics = {**metrics, **arcs_and_tags_f1_macro, **arcs_and_tags_f1_micro}
            except RuntimeError as ex:
                logger.warning(f'got an exception on get_metrics(): {str(ex)}')
            if self._graph_based_metric:
                metrics.update(self._graph_based_metric.get_metric(reset=reset))

        return metrics
