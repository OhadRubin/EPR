import json
from typing import Dict, Tuple, List

from overrides import overrides
from spacy.tokens.doc import Doc

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import QDMROperation, StepsSpans
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.steps_spans_extractors.base_steps_spans_extractor import BaseSpansExtractor


class FromFileSpansExtractor(BaseSpansExtractor):
    def __init__(self, *file_path: str):
        super().__init__()
        self._cache: Dict[str, Tuple[StepsSpans, dict]] = {}
        for file in file_path:
            with open(file, 'r') as f:
                for line in f.readlines():
                    content = json.loads(line.strip())
                    steps_spans: StepsSpans = StepsSpans.from_dict(content['steps_spans'])
                    metadata = content['metadata']
                    self._cache[metadata['question_id']] = steps_spans, metadata

    @overrides
    def extract(self, question_id: str, question: str, decomposition:str, operators: List[str] = None,
                debug: dict = None) -> StepsSpans:
        steps_spans, metadata = self._cache[question_id]
        if debug is not None: debug.update(**metadata)
        return steps_spans

    @overrides
    def _extract(self, question_id: str, question_tokens: Doc, steps_tokens: List[Doc], steps_operators: List[QDMROperation] = None,
                 debug: dict = None) -> StepsSpans:
        raise NotImplementedError('use extract()')