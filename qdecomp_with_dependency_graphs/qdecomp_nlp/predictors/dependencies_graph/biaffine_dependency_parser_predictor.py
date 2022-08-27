from typing import Dict, Any, List, Tuple

from allennlp.data.dataset_readers import InterleavingDatasetReader
from allennlp.data.fields import MetadataField
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

from qdecomp_with_dependency_graphs.dependencies_graph.create_dependencies_graphs import get_extra_tokens
from qdecomp_nlp.data.dataset_readers.multitask import CustomMultiTaskDatasetReader
from qdecomp_with_dependency_graphs.utils.graph import render_dependencies_graph_svg


@Predictor.register("biaffine-dependency-parser", exist_ok=True)
class BiaffineDependencyParserPredictor(Predictor):
    """
    Predictor for the [`BiaffineDependencyParser`](../models/biaffine_dependency_parser.md) model.
    """

    def __init__(
        self, model: Model, dataset_reader: DatasetReader, language: str = "en_core_web_sm"
    ) -> None:
        super().__init__(model, dataset_reader)
        # TODO(Mark) Make the language configurable and based on a model attribute.
        self._tokenizer = SpacyTokenizer(language=language, pos_tags=True)

    def predict(self, sentence: str, metadata: dict=None) -> JsonDict:
        """
        Predict a dependency parse for the given sentence.
        # Parameters
        sentence The sentence to parse.
        # Returns
        A dictionary representation of the dependency tree.
        """
        return self.predict_json({"sentence": sentence, "metadata": metadata})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        """
        spacy_tokens = self._tokenizer.tokenize(json_dict["sentence"])
        sentence_text = [token.text for token in spacy_tokens]

        dataset_reader = self._dataset_reader
        if isinstance(self._dataset_reader, InterleavingDatasetReader):
            potential_dataset_readers = [(x, y) for x, y in self._dataset_reader._readers.items() if 'graph' in x]
            assert len(potential_dataset_readers) == 1
            dataset_reader_name, dataset_reader = potential_dataset_readers[0]
        elif isinstance(self._dataset_reader, CustomMultiTaskDatasetReader):
            potential_dataset_readers = [(x, y) for x, y in self._dataset_reader.readers.items() if 'graph' in x]
            assert len(potential_dataset_readers) == 1
            dataset_reader_name, dataset_reader = potential_dataset_readers[0]

        if dataset_reader.pos_field == "tag":  # type: ignore
            # fine-grained part of speech
            pos_tags = [token.tag_ for token in spacy_tokens]
        else:
            # coarse-grained part of speech (Universal Depdendencies format)
            pos_tags = [token.pos_ for token in spacy_tokens]

        extra_tokens, extra_pos = get_extra_tokens()
        sentence_text.extend(extra_tokens)
        pos_tags.extend(extra_pos)
        instance = dataset_reader.text_to_instance(sentence_text, pos_tags, metadata=json_dict["metadata"])

        if isinstance(self._dataset_reader, InterleavingDatasetReader):
            instance.fields[self._dataset_reader._dataset_field_name] = MetadataField(dataset_reader_name)
        if isinstance(self._dataset_reader, CustomMultiTaskDatasetReader):
            instance.fields['task'] = MetadataField(dataset_reader_name)

        return instance
