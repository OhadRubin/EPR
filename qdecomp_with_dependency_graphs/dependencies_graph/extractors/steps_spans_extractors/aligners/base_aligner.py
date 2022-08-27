import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Set

import spacy
from spacy.tokens.doc import Doc

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import QDMROperation, StepsSpans


class BaseAligner(ABC):
    def align(self, question: Doc, steps: List[Doc], steps_operators: List[QDMROperation],
              index_to_steps: List[Set[Tuple[int, int]]]) -> List[Set[Tuple[int, int]]]:
        raise NotImplementedError()
