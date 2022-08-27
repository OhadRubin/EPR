"""
Extract spans for a QDMR decomposition steps
"""
from typing import Dict, List, Iterable, Set
from overrides import overrides
from collections import defaultdict

from spacy.tokens import Token, Doc

from qdecomp_with_dependency_graphs.dependencies_graph.extractors.steps_spans_extractors.aligners.base_aligner import BaseAligner
from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition
from qdecomp_with_dependency_graphs.dependencies_graph.data_types import QDMROperation, Span, StepsSpans
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.steps_spans_extractors.base_steps_spans_extractor import BaseSpansExtractor

from qdecomp_with_dependency_graphs.scripts.data_processing.break_app_store_generation import get_lexicon_tokens


class VariationsBasedSpansExtractor(BaseSpansExtractor):
    def __init__(self, allow_conflicts: bool = False, include_gaps: bool = False, aligner: BaseAligner = None):
        super().__init__()
        self.allow_conflicts = allow_conflicts
        self.aligner = aligner
        self.include_gaps = include_gaps

    @overrides
    def _extract(self, question_id: str, question_tokens: Doc, steps_tokens: List[Doc], steps_operators: List[QDMROperation] = None,
                 debug: dict = None) -> StepsSpans:
        # map question tokens variations to indexes
        mapping_lemma_to_index = _get_variations_to_index_dict(question_tokens)

        # map steps to question tokens
        index_to_steps = [set([]) for i in range(len(question_tokens))]
        for step_id, step in enumerate(steps_tokens):
            for j, token in enumerate(step):
                question_tokens_indexes = mapping_lemma_to_index.get(token.text.lower(), [])
                for i in question_tokens_indexes:
                    index_to_steps[i].add((step_id, j))

        # break multiple steps to single token mapping
        if self.aligner:
            index_to_steps = self.aligner.align(question=question_tokens, steps=steps_tokens, index_to_steps=index_to_steps,
                               steps_operators=steps_operators)

        index_to_steps = [set([step_id for step_id, ind in x]) for x in index_to_steps]

        # operational keywords
        _map_operational_keywords(question=question_tokens, steps=steps_tokens, index_to_steps=index_to_steps,
                                  steps_operators=steps_operators)

        # extract spans
        spans: List[List[Span]] = [[] for _ in range(len(steps_tokens))]
        last_step_id: int = None
        last_span: Span = None
        conflicts_amount = 0
        for i, steps_ids in enumerate(index_to_steps):
            if len(steps_ids) > 1:
                conflicts_amount += 1
                if last_span:
                    spans[last_step_id].append(last_span)
                last_span = None
                last_step_id = None
            elif len(steps_ids) == 1:
                curr_step_id = steps_ids.pop()
                if last_step_id == curr_step_id:
                    last_span.end = i
                else:
                    if last_span:
                        spans[last_step_id].append(last_span)
                    last_span = Span(i, i)
                last_step_id = curr_step_id
            elif not self.include_gaps:
                if last_span:
                    spans[last_step_id].append(last_span)
                last_span = None
                last_step_id = None
        if last_span:
            spans[last_step_id].append(last_span)

        # add conflicts
        if self.allow_conflicts:
            for i, steps_ids in enumerate(index_to_steps):
                if len(steps_ids) > 1:
                    for step_id in steps_ids:
                        spans[step_id].append(Span(i,i))
            # merge if needed
            for step_id, step_spans in enumerate(spans):
                if not step_spans: continue
                sorted_ = sorted(step_spans, key=lambda x: x.start)
                new_list: List[Span] = [sorted_[0]]
                for x in sorted_[1:]:
                    if x.start == new_list[-1].end+1:
                        new_list[-1].end = x.end
                    else:
                        new_list.append(x)
                spans[step_id] = new_list

        # remove unnecessary spans
        spans = [_clean_spans(question_tokens, step_spans) for step_spans in spans]

        if debug is not None: debug['conflicts'] = conflicts_amount
        return StepsSpans(question_tokens=[x.text for x in question_tokens],
                          steps=[str(x) for x in steps_tokens],
                          steps_spans=spans,
                          steps_tags=[x.value for x in steps_operators])


def _get_variations_to_index_dict(tokens: Iterable[Token]) -> Dict[str, List[int]]:
    """
    Map each variation of a token to the same indexes list
    :param tokens: tokens list
    :return: dictionary of <token-variation, list(indexes)>
    """
    variations_dict = defaultdict(list)
    for i, token in enumerate(tokens):
        curr_value = variations_dict[token.text.lower()]
        curr_value.append(i)
        variations = [x.lower() for x in get_lexicon_tokens(token)]
        variations_dict.update(dict.fromkeys(variations, curr_value))
    return variations_dict


def _clean_spans(question:Doc, spans: Iterable[Span]):
    def is_valid(span: Span):
        if all(_ignore(t) for t in question[span.start:span.end+1]):
            return False
        return True
    return [x for x in spans if is_valid(x)]


def _ignore(token: Token):
    return token.pos_ in ['DET', 'AUX'] or token.text in ['of', 'for']


def _map_operational_keywords(question: Doc, steps: List[Doc], steps_operators: List[QDMROperation],
                              index_to_steps: List[Set[int]]):
    if not steps_operators:
        return

    assert len(question) == len(index_to_steps) and len(steps) == len(steps_operators), \
        f"mismatch sizes({len(question)}={len(index_to_steps)}, {len(steps)}={len(steps_operators)})"

    if steps_operators[-1] in [QDMROperation.AGGREGATE, QDMROperation.ARITHMETIC]:
        if str(question[:2]).lower() in ['how many', 'how much']:
            for i in [0, 1]:
                index_to_steps[i].add(len(steps) - 1)