import re
from abc import ABC, abstractmethod
from typing import List, Tuple

import spacy
from spacy.tokens.doc import Doc

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import QDMROperation, StepsSpans


class BaseSpansExtractor(ABC):
    def __init__(self, tokens_parser=None):
        self._parser = tokens_parser or spacy.load("en_core_web_sm")

    def extract(self, question_id: str, question: str, decomposition:str, operators: List[str] = None,
                debug: dict = None) -> StepsSpans:
        def format(text: str):
            return re.sub(r'\s+', ' ', text)

        question_tokens = self._parser(format(question))
        decomposition = re.sub(r'#(\d+)', '@@\g<1>@@', decomposition)
        steps_tokens = [self._parser(' '.join(format(x).split(' ')[1:])) for x in decomposition.split(';')]
        steps_operators = operators and [QDMROperation(x) for x in operators]
        return self._extract(question_id=question_id, question_tokens=question_tokens,
                             steps_tokens=steps_tokens, steps_operators=steps_operators)

    @abstractmethod
    def _extract(self, question_id: str, question_tokens: Doc, steps_tokens: List[Doc], steps_operators: List[QDMROperation] = None,
                 debug: dict = None) -> StepsSpans:
        raise NotImplementedError()