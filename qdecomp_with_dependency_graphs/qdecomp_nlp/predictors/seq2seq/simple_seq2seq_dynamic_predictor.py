"""
based on allennlp_models/generation/predictors/seq2seq.py
tag: v1.1.0
"""

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('seq2seq_dynamic')
class Seq2SeqDynamicPredictor(Predictor):
    """
    Predictor for sequence to sequence models, including
    [`SimpleSeq2SeqDynamic`](../models/seq2seq/simple_seq2seq_dynamic.md) and
    [`CopyNetSeq2SeqDynamic`](../models/seq2seq/copynet_seq2seq_dynamic.md).
    """

    def predict(self, source: str, allowed_tokens: str) -> JsonDict:
        return self.predict_json({"source": source, "allowed_tokens": allowed_tokens})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"source": "..."}`.
        """
        source = json_dict["source"]
        allowed_tokens = json_dict["allowed_tokens"]
        return self._dataset_reader.text_to_instance(source, allowed_tokens)
