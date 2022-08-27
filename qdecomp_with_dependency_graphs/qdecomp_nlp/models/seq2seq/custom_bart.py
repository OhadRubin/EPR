from typing import Dict, Tuple, Any, cast, List

from allennlp.modules import TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator
from allennlp_models.generation import Bart
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import ROUGE, BLEU, Metric
from transformers import BartTokenizer

from transformers.models.bart.modeling_bart import BartModel, BartForConditionalGeneration

import torch
from torch import nn
import torch.nn.functional as F


@Model.register("custom_bart")
class CustomBart(Bart):
    def __init__(self,
                 vocab: Vocabulary,
                 model_name: str,
                 indexer: PretrainedTransformerIndexer = None,
                 max_decoding_steps: int = 140,
                 beam_size: int = 4,
                 encoder: Seq2SeqEncoder = None,

                 embedder: TextFieldEmbedder = None,
                 token_based_metric: Metric = None,
                 model_config: Dict[str, Any] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        Model.__init__(self, vocab)
        model_config = model_config or {}
        self.bart = BartForConditionalGeneration.from_pretrained(model_name, **model_config)
        self._indexer = indexer or PretrainedTransformerIndexer(model_name, namespace="tokens")

        self._start_id = self.bart.config.bos_token_id  # CLS
        self._decoder_start_id = self.bart.config.decoder_start_token_id or self._start_id
        self._end_id = self.bart.config.eos_token_id  # SEP
        self._pad_id = self.bart.config.pad_token_id  # PAD

        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(
            self._end_id, max_steps=max_decoding_steps, beam_size=beam_size or 1
        )

        self._rouge = ROUGE(exclude_indices={self._start_id, self._pad_id, self._end_id})
        self._bleu = BLEU(exclude_indices={self._start_id, self._pad_id, self._end_id})

        self._source_embedder = embedder
        self._encoder = encoder

        # super().__init__(vocab=vocab, model_name=model_name, **kwargs)
        self._token_based_metric = token_based_metric
        self._tokenizer: BartTokenizer = self._indexer._tokenizer if self._indexer else BartTokenizer.from_pretrained(model_name)
        self._special_tokens_to_ignore = [x for x in self._tokenizer.all_special_tokens if x not in self._tokenizer.additional_special_tokens]
        self.bart.resize_token_embeddings(len(self._tokenizer))  # we added special tokens (@@SEP@@, @@1@@, ...)

        initializer(self)

    @overrides
    def forward(self, metadata: List[Dict[str, Any]],
                source_tokens: TextFieldTensors, target_tokens: TextFieldTensors = None) -> Dict[str, torch.Tensor]:
        inputs = source_tokens
        targets = target_tokens
        input_ids, input_mask = inputs["tokens"]["token_ids"], inputs["tokens"]["mask"]

        outputs = {}

        # update encoder outputs
        encoder_outputs = None
        if self._encoder:
            if self._source_embedder:
                embeddings = self._source_embedder(source_tokens)
                source_mask = util.get_text_field_mask(source_tokens)
            else:
                embeddings = self.bart.model.encoder.embed_tokens(input_ids) + self.bart.model.encoder.embed_positions(input_ids)
                source_mask = input_mask
            encoder_outputs = (self._encoder(embeddings, source_mask),)

        # If no targets are provided, then shift input to right by 1. Bart already does this internally
        # but it does not use them for loss calculation.
        if targets is not None:
            target_ids, target_mask = targets["tokens"]["token_ids"], targets["tokens"]["mask"]
        else:
            target_ids = input_ids[:, 1:]
            target_mask = input_mask[:, 1:]

        if self.training:
            bart_outputs = self.bart(
                input_ids=input_ids,
                attention_mask=input_mask,
                decoder_input_ids=target_ids[:, :-1].contiguous(),
                decoder_attention_mask=target_mask[:, :-1].contiguous(),
                use_cache=False,
                return_dict=True,
                encoder_outputs=encoder_outputs,
            )

            outputs["decoder_logits"] = bart_outputs.logits

            # The BART paper mentions label smoothing of 0.1 for sequence generation tasks
            outputs["loss"] = sequence_cross_entropy_with_logits(
                bart_outputs.logits,
                cast(torch.LongTensor, target_ids[:, 1:].contiguous()),
                cast(torch.BoolTensor, target_mask[:, 1:].contiguous()),
                label_smoothing=0.1,
                average="token",
            )
        else:
            # Use decoder start id and start of sentence to start decoder
            initial_decoder_ids = torch.tensor(
                [[self._decoder_start_id]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).repeat(input_ids.shape[0], 1)

            inital_state = {
                "input_ids": input_ids,
                "input_mask": input_mask,
            }
            if encoder_outputs:
                inital_state["encoder_states"] = encoder_outputs[0]
            beam_result = self._beam_search.search(
                initial_decoder_ids, inital_state, self.take_step
            )

            predictions = beam_result[0]
            max_pred_indices = (
                beam_result[1].argmax(dim=-1).view(-1, 1, 1).expand(-1, -1, predictions.shape[-1])
            )
            predictions = predictions.gather(dim=1, index=max_pred_indices).squeeze(dim=1)

            self._rouge(predictions, target_ids)
            self._bleu(predictions, target_ids)

            outputs["predictions"] = predictions
            outputs["log_probabilities"] = (
                beam_result[1].gather(dim=-1, index=max_pred_indices[..., 0]).squeeze(dim=-1)
            )

            self.make_output_human_readable(outputs)


        # add metadata
        if metadata:
            outputs['metadata'] = metadata

        # token-based metric
        if not self.training:
            if self._token_based_metric is not None:
                self._token_based_metric(
                    outputs["predicted_tokens"],
                    # [self._decode(self._indexer.tokens_to_indices([Token(y) for y in x["target_tokens"]], self.vocab)) for x in metadata],
                    self._wordpieces_to_tokens([x["target_tokens"] for x in metadata]),
                    metadata
                )
        return outputs

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        # Parameters
        output_dict : `Dict[str, torch.Tensor]`
            A dictionary containing a batch of predictions with key `predictions`. The tensor should have
            shape `(batch_size, max_sequence_length)`
        # Returns
        `Dict[str, Any]`
            Original `output_dict` with an additional `predicted_tokens` key that maps to a list of lists of
            tokens.
        """
        # predictions = output_dict["predictions"]
        # predicted_tokens = [None] * predictions.shape[0]
        # for i in range(predictions.shape[0]):
        #     predicted_tokens[i] = self._decode(predictions[i].tolist()).split()
        # output_dict["predicted_tokens"] = predicted_tokens

        output_dict = super().make_output_human_readable(output_dict)
        output_dict["predicted_tokens"] = self._wordpieces_to_tokens(
            [[t.text for t in x] for x in output_dict["predicted_tokens"]]
        )
        return output_dict

    # def _decode(self, token_ids: List[int]) -> str:
    #     return self._tokenizer.decode(token_ids=token_ids, skip_special_tokens=True)

    def _wordpieces_to_tokens(self, wordpieces_text: List[List[str]]) -> List[List[str]]:
        return [
            self._tokenizer.convert_tokens_to_string([y for y in x if y not in self._special_tokens_to_ignore]).split()
            for x in wordpieces_text
        ]

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = super().get_metrics(reset=reset)
        if not self.training and self._token_based_metric:
            metrics.update(self._token_based_metric.get_metric(reset=reset))
        return metrics