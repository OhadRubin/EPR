from typing import Dict, Optional, List, Any

import numpy
import torch
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from allennlp_models.generation.modules.seq_decoders import SeqDecoder

from transformers import EncoderDecoderModel, AutoTokenizer

@Model.register("pretrained-transformers-seq2seq")
class PretrainedTransformersSeq2Seq(Model):
    """
    This class is a `Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.
    # Parameters
    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    encoder_model_name : `str`
        The name of the `transformers` model to use.
    decoder_model_name : `str`, optional (default: encoder_model_name)
        The name of the `transformers` model to use.
    beam_size : ``int``, optional (default = None)
        Width of the beam for beam search. If not specified, greedy decoding is used.
    trainable : ``bool``, optional (default : True)
        If True, the weights of the pretrained transformer model will be updated during training.
        Otherwise, they will be frozen and only the final linear layer will be trained.
    regularizer : `RegularizerApplicator`, optional (default=`None`)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
            self,
            vocab: Vocabulary,
            encoder_model_name: str,
            decoder_model_name: str = None,
            beam_size: int = None,
            trainable: bool = True,
            regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)
        self._tokenizer = AutoTokenizer.from_pretrained(decoder_model_name or encoder_model_name)
        self.beam_size = beam_size

        if not decoder_model_name:
            decoder_model_name = encoder_model_name
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model_name, decoder_model_name)

        for param in self.model.parameters():
            param.requires_grad = trainable


    @overrides
    def forward(self,
                source_tokens: TextFieldTensors,
                target_tokens: TextFieldTensors = None,
                metadata: List[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:

        source_mask = source_tokens["tokens"]["mask"]

        if target_tokens:
            target_mask = target_tokens["tokens"]["mask"]
            outputs = self.model(input_ids=source_tokens['tokens']['token_ids'], decoder_input_ids=target_tokens["tokens"]['token_ids'],
                                 labels=target_tokens["tokens"]['token_ids'],
                                 encoder_attention_mask=source_mask, decoder_attention_mask=target_mask)
            output_dict = {"loss": outputs[0]}
        else:
            output_dict = {}

        if not self.training:
            if target_tokens:
                outputs = outputs[1:]
            else:
                # todo: not just "[CLS]" (generic...)
                # todo: to(cuda) - check if in "cuda-mode"
                dummy = self._tokenizer.batch_encode_plus(["[CLS]"]*source_tokens['tokens']['token_ids'].shape[0], add_special_tokens=False)
                target_tokens = {"tokens": torch.tensor(dummy["input_ids"], dtype=torch.long).to('cuda')}
                outputs = self.model(source_tokens['tokens']['token_ids'], target_tokens["tokens"]['token_ids'],
                                     encoder_attention_mask=source_mask) #, decoder_beam_size=self.beam_size)
            predictions = outputs[0]  # todo: beam search
            predicted_indices = torch.argmax(predictions, dim=2)
            output_dict.update({"predictions": predicted_indices})

        if metadata:
            output_dict['metadata'] = metadata
        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.
        This method overrides `Model.decode`, which gets called after `Model.forward`, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the `forward` method.
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called `predicted_tokens` to the `output_dict`.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)

            predicted_tokens = self._tokenizer.convert_ids_to_tokens(indices, skip_special_tokens=True)
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    # @overrides
    # def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    #     # todo: implement
    #     return {}