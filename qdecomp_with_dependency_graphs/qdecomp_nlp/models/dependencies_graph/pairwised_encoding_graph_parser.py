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
import numpy

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, Activation
from allennlp.nn.util import min_value_of_dtype
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import Metric, F1Measure, FBetaMeasure, BooleanAccuracy, CategoricalAccuracy

logger = logging.getLogger(__name__)


class PairCombination(str, Enum):
    CONCAT = 'concat'
    SUBTRACT = 'subtract'


@Model.register("pairwise_graph_parser")
class PairwiseGraphParser(Model):
    """
    A Parser for arbitrary graph structures.
    # Parameters
    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the `tokens` `TextField` we get as input to the model.
    encoder : `Seq2SeqEncoder`
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    pairs_encoder : `Seq2SeqEncoder`
        The encoder that will used to represent a pair of tokens encodings
    pair_combination: `PairCombination`
        Strategy for constructing a pair embeddings based on the tokens encoder output.
        These pairs will be fed to the pairs_encoder
    labels_namespace: `str`
        Namespace of the arc tags labels
    dropout : `float`, optional, (default = `0.0`)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : `float`, optional, (default = `0.0`)
        The dropout applied to the embedded text input.
    graph_based_metric: `Metric`
        A metric to calculate on validation phases, that receives the predicted graphs
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            encoder: Seq2SeqEncoder,
            pairs_encoder: Seq2SeqEncoder,
            pair_combination: PairCombination = PairCombination.SUBTRACT,
            labels_namespace: str = 'labels',
            dropout: float = 0.0,
            input_dropout: float = 0.0,
            graph_based_metric: Metric = None,
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._num_labels = vocab.get_vocab_size(namespace=labels_namespace)
        self._label_namespace = labels_namespace
        self._pair_combination = pair_combination

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self._pairs_encoder = pairs_encoder
        self._classification_layer = torch.nn.Linear(self._pairs_encoder.get_output_dim(), self._num_labels)

        check_dimensions_match(
            self._text_field_embedder.get_output_dim(),
            self._encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )
        check_dimensions_match(
            {
                PairCombination.CONCAT: 2*self._encoder.get_output_dim(),
                PairCombination.SUBTRACT: self._encoder.get_output_dim()
            }[self._pair_combination],
            self._pairs_encoder.get_input_dim(),
            "encoder output dim",
            "pairs encoder input dim",
        )

        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)

        self._loss = torch.nn.CrossEntropyLoss(reduction="none")

        # metrics
        self._arc_tags_accuracy = CategoricalAccuracy()

        self._none_index = self.vocab.get_token_to_index_vocabulary(labels_namespace).get('NONE', -1)
        labels_indices = [v for k, v in self.vocab.get_token_to_index_vocabulary(labels_namespace).items()
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
            metadata: List[Dict[str, Any]] = None,
            arc_indices: torch.LongTensor = None,
            arc_tags: torch.LongTensor = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters
        tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()`.
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

        batch_size, n, d = encoded_text.size()
        encoded_text_head = encoded_text.unsqueeze(-2).expand(-1, -1, n, -1)
        encoded_text_child = encoded_text.unsqueeze(-3).expand(-1, n, -1, -1)
        if self._pair_combination == PairCombination.SUBTRACT:
            pairs_encodings = encoded_text_head - encoded_text_child
        else:
            pairs_encodings = torch.cat([encoded_text_head, encoded_text_child], dim=-1)
        pairs_encodings = self._pairs_encoder(pairs_encodings.view(batch_size, -1, d)).view(batch_size, n, n, -1)

        tag_mask = mask.unsqueeze(-1)*mask.unsqueeze(-2)
        arc_tag_logits = self._classification_layer(pairs_encodings)
        arc_tag_probs = torch.nn.functional.softmax(arc_tag_logits, dim=-1)

        output_dict = {"arc_tag_probs": arc_tag_probs, "mask": mask}

        if metadata:
            output_dict["metadata"] = metadata

        if arc_tags is not None:
            arc_tags = arc_tags * tag_mask  # padding_value => 0
            batch_size, sequence_length, _, num_tags = arc_tag_logits.size()
            original_shape = [batch_size, sequence_length, sequence_length]
            reshaped_logits = arc_tag_logits.view(-1, num_tags)
            reshaped_tags = arc_tags.view(-1)
            loss = self._loss(reshaped_logits, reshaped_tags.long()).view(original_shape) * tag_mask
            output_dict["loss"] = loss.sum() / tag_mask.sum().clamp(min=1).float()

            # compute metrics on validation only
            if not self.training:
                self._arc_tags_accuracy(arc_tag_logits, arc_tags)
                self._arcs_and_tags_f1_macro(arc_tag_probs, arc_tags, tag_mask)
                self._arcs_and_tags_f1_micro(arc_tag_probs, arc_tags, tag_mask)

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
        arc_tag_probs = _prop_getter("arc_tag_probs").cpu().detach().numpy()
        mask = _prop_getter("mask")
        lengths = get_lengths_from_binary_sequence_mask(mask)
        arcs = []
        arc_tags = []
        for instance_arc_tag_probs, length in zip(arc_tag_probs, lengths):
            edges = []
            edge_tags = []
            for i in range(length):
                for j in range(length):
                    # Ignore the diagonal, because we don't self edges
                    if i != j:
                        tag = instance_arc_tag_probs[i, j].argmax(-1)
                        if tag == self._none_index:
                            continue
                        tags = self.vocab.get_token_from_index(tag, self._label_namespace).split('&')
                        for tag in tags:
                            edges.append((i, j))
                            edge_tags.append(tag)
            arcs.append(edges)
            arc_tags.append(edge_tags)

        output_dict["arcs"] = arcs
        output_dict["arc_tags"] = arc_tags
        return output_dict


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        if not self.training:
            arcs_and_tags_f1_macro = {f'arcs_and_tags_macro_{k}': v for k, v in self._arcs_and_tags_f1_macro.get_metric(reset).items()}
            arcs_and_tags_f1_micro = {f'arcs_and_tags_micro_{k}': v for k, v in self._arcs_and_tags_f1_micro.get_metric(reset).items()}

            metrics = {**metrics, **arcs_and_tags_f1_macro, **arcs_and_tags_f1_micro}

            if self._graph_based_metric:
                metrics.update(self._graph_based_metric.get_metric(reset=reset))

        return metrics
