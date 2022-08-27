from typing import Dict, Tuple, List, Any
import logging
from enum import Enum

import os
from allennlp.nn import util, InitializerApplicator
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from overrides import overrides
import torch

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models.model import Model
from allennlp.training.metrics import Average, Metric, CategoricalAccuracy

from qdecomp_nlp.modules.seq2seq_encoders.latent_relation_aware_transformer import LatentRelationAwareTransformer


logger = logging.getLogger(__name__)


class CombinationStrategy(str, Enum):
    MAX = 'max'
    MULTIPLY = 'multiply'


@Model.register("multitask_rat")
class MultitaskRatBasedModel(Model):
    """
    Multitask model
    """
    def __init__(self,
                 vocab: Vocabulary,
                 tags_namespace: str,
                 relations_encoding_dim: int,
                 seq2seq_model: Model,
                 separate_kv_classification: bool = False,
                 combination_strategy: CombinationStrategy = CombinationStrategy.MULTIPLY,
                 graph_based_metric: Metric = None,
                 graph_loss_weight: float = 1.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super().__init__(vocab)
        self._seq2seq_model = seq2seq_model
        self._label_namespace = tags_namespace
        self._none_index = self.vocab.get_token_to_index_vocabulary(self._label_namespace).get('NONE', -1)
        self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._graph_loss_weight = graph_loss_weight

        self._embedder = self._seq2seq_model._source_embedder
        self._encoder: LatentRelationAwareTransformer = self._seq2seq_model._encoder
        assert isinstance(self._encoder, LatentRelationAwareTransformer)

        self._combination_strategy = combination_strategy
        self._separate_kv_classification = separate_kv_classification
        assert self._encoder.tie_layers or self._separate_kv_classification, \
            "tie_layers=false is supported on separate mode only"
        if self._separate_kv_classification:
            N = len(self._encoder.encoder.layers) if not self._encoder.tie_layers else 1
            self._classification_layer_k = torch.nn.ModuleList([torch.nn.Linear(relations_encoding_dim, self._num_labels) for _ in range(N)])
            self._classification_layer_v = torch.nn.ModuleList([torch.nn.Linear(relations_encoding_dim, self._num_labels) for _ in range(N)])
        else:
            self._classification_layer = torch.nn.Linear(2*relations_encoding_dim, self._num_labels)

        # self._none_index = vocab.get_token_index('NONE', self._label_namespace)
        self._arc_tags_loss = torch.nn.CrossEntropyLoss(reduction="none") #, ignore_index=self._none_index)

        # tasks loss as metrics
        self._losses: Dict[str, Average] = {k: Average() for k in ['seq2seq', 'graph_parser']}
        self._losses.update({"total": Average()})

        self._graph_based_metric = graph_based_metric
        self._arc_tags_accuracy = CategoricalAccuracy() if not self._separate_kv_classification else None

        self._detailed_prediction: bool = os.environ.get('detailed') is not None
        initializer(self)


    @overrides
    def forward(self, task: List[str], *args, **kwargs) -> Dict[str, torch.Tensor]:
        task_set = set(task)
        if len(task_set) != 1:
            raise ValueError(f"Unexpected batch - got instances from different tasks {task_set}")
        task = task_set.pop()
        if task == 'seq2seq':
            output = self._seq2seq_model(*args, **kwargs)
        else:
            output = self.forward_graph(*args, **kwargs)

        if "loss" in output:
            if task != "seq2seq":
                output["loss"] *= self._graph_loss_weight
            loss = output['loss'].item()  # float (need to be serialized to json)
            self._losses[task](loss)
            self._losses["total"](loss)
        output['is_seq2seq'] = torch.tensor(task == 'seq2seq')
        return output

    def forward_graph(self,
                      tokens: TextFieldTensors,
                      pos_tags: torch.LongTensor = None,
                      metadata: List[Dict[str, Any]] = None,
                      arc_indices: torch.LongTensor = None,
                      arc_tags: torch.LongTensor = None,) -> Dict[str, torch.Tensor]:
        # enc_output = self.seq2seq_model._encode(tokens)
        # encoded_text = enc_output["encoder_outputs"]

        # shape: (batch_size, source_sequence_length, encoder_input_dim)
        embedded_input = self._embedder(tokens)
        # shape: (batch_size, source_sequence_length)
        source_mask = util.get_text_field_mask(tokens)
        # shape: (batch_size, source_sequence_length, encoder_output_dim)
        _, relation_k, relation_v = self._encoder(embedded_input, source_mask, return_relations=True)

        tag_mask = source_mask.unsqueeze(-1)*source_mask.unsqueeze(-2)
        if self._separate_kv_classification:
            arc_tag_logits_k = [cls(r_k) for cls, r_k in zip(self._classification_layer_k, relation_k)]
            arc_tag_logits_v = [cls(r_v) for cls, r_v in zip(self._classification_layer_v, relation_v)]
            arc_tag_probs_k = [torch.nn.functional.softmax(x, dim=-1) for x in arc_tag_logits_k]
            arc_tag_probs_v = [torch.nn.functional.softmax(x, dim=-1) for x in arc_tag_logits_v]
            agg_func = {
                CombinationStrategy.MAX: torch.max,
                CombinationStrategy.MULTIPLY: torch.prod
            }[self._combination_strategy]
            arc_tag_probs = agg_func(torch.stack([*arc_tag_probs_k, *arc_tag_probs_v], dim=-1), dim=-1)
            arc_tag_probs = torch.softmax(arc_tag_probs, dim=-1)
            output_dict = {"arc_tag_probs": arc_tag_probs, "mask": source_mask}
        else:
            assert len(relation_k) == len(relation_v) == 1
            relations_enc = torch.cat([relation_k[0], relation_v[0]], dim=-1)
            arc_tag_logits = self._classification_layer(relations_enc)
            arc_tag_probs = torch.nn.functional.softmax(arc_tag_logits, dim=-1)
            output_dict = {"arc_tag_probs": arc_tag_probs, "mask": source_mask}

        if metadata:
            output_dict["metadata"] = metadata

        if arc_tags is not None:
            arc_tags = arc_tags * tag_mask  # padding_value => 0
            def get_loss(logits):
                batch_size, sequence_length, _, num_tags = logits.size()
                original_shape = [batch_size, sequence_length, sequence_length]
                reshaped_logits = logits.view(-1, num_tags)
                reshaped_tags = arc_tags.view(-1)
                loss = self._arc_tags_loss(reshaped_logits, reshaped_tags.long()).view(original_shape) * tag_mask
                return (loss.sum() / tag_mask.sum().clamp(min=1).float())

            if self._separate_kv_classification:
                for i, (logits_k, logits_v) in enumerate(zip(arc_tag_logits_k, arc_tag_logits_v)):
                    output_dict[f"loss_k.{i}"] = get_loss(logits_k)
                    output_dict[f"loss_v.{i}"] = get_loss(logits_v)
                loss_terms = [output_dict[f"loss_{x}.{i}"]
                              for x in ["k", "v"]
                              for i in range(len(arc_tag_logits_k))]
                output_dict["loss"] = sum(loss_terms) / max(1, len(loss_terms))
            else:
                output_dict["loss"] = get_loss(arc_tag_logits)
                self._arc_tags_accuracy(arc_tag_logits, arc_tags)

            # compute metrics on validation only
            if not self.training:
                # graph based metric on validation only
                if self._graph_based_metric:
                    readable_dict = self._graph_make_output_human_readable(output_dict=output_dict.copy())
                    self._graph_based_metric(
                        arcs=readable_dict['arcs'],
                        arc_tags=readable_dict['arc_tags'],
                        metadata=metadata
                    )

        return output_dict

    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        is_seq2seq = output_dict["is_seq2seq"].item()
        if is_seq2seq:
            return self._seq2seq_model.make_output_human_readable(output_dict=output_dict)

        return self._graph_make_output_human_readable(output_dict=output_dict)

    def _graph_make_output_human_readable(
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

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        models_metrics = {}

        metrics_map = {
            'seq2seq': lambda: [self._seq2seq_model.get_metrics(reset=reset)],
            'graph_parser': lambda: [x for x in [
                self._graph_based_metric and self._graph_based_metric.get_metric(reset=reset),
                self._arc_tags_accuracy and {"arc_tags_accuracy":  self._arc_tags_accuracy.get_metric(reset=reset)}
                ] if x]
        }
        for model_name, get_metrics in metrics_map.items():
            # workaround: calling get_metrics() before any call to this model forward
            try:
                models_metrics.update({
                    f'{model_name}-{k}': v for m in get_metrics() for k, v in m.items()
                })
            except Exception as ex:
                logger.exception(f"Failed to get metrics from model {model_name}")
        models_losses = {f'{k}-loss': v.get_metric(reset=reset) for k, v in self._losses.items()}

        logical_form_ems = [v for k, v in models_metrics.items() if 'logical_form_em' in k]
        logical_form = {'maximal_logical_form_em': max(logical_form_ems)} if logical_form_ems else {}
        return {**models_metrics, **models_losses, **logical_form}

