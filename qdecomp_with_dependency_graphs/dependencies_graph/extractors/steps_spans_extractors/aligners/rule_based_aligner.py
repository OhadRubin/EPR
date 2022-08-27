from collections import defaultdict
from typing import List, Set, Tuple, Callable

from spacy.tokens.doc import Doc

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import QDMROperation
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.steps_spans_extractors.aligners.base_aligner import BaseAligner
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.steps_spans_extractors.variations_based_steps_spans_extractor import _ignore
from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition


class RuleBasedAligner(BaseAligner):
    def align(self, question: Doc, steps: List[Doc], steps_operators: List[QDMROperation],
              index_to_steps: List[Set[Tuple[int, int]]]) -> List[Set[Tuple[int, int]]]:
        index_to_steps_: List[Set[int]] = [set(s for s, _ in x) for x in index_to_steps]

        # break multiple steps to single token mapping
        _break_multi_mapping(question=question, steps=steps, index_to_steps=index_to_steps_,
                             steps_operators=steps_operators)

        return [set((s, i) for s, i in x if s in y) for x, y in zip(index_to_steps, index_to_steps_)]


##############################################
#       Break multi mapping rules            #
##############################################

def _break_multi_mapping(question: Doc, steps: List[Doc], steps_operators: List[QDMROperation],
                         index_to_steps: List[Set[int]]):
    assert len(question) == len(index_to_steps), f"mismatch sizes({len(question)}={len(index_to_steps)})"
    assert not steps_operators or len(steps) == len(
        steps_operators), f"mismatch sizes({len(steps)}={len(steps_operators)})"
    local_rules = [
        _break_rule_prefer_sequences,
        _break_rule_use_references,
    ]
    # holistic_rules = [
    #     _break_rule_prefer_not_aligned,
    #     _break_rule_prefer_exact_match,
    # ]

    # _break_rule_prefer_not_aligned(question, steps, steps_operators, index_to_steps)
    for lr in local_rules:
        for i in range(len(index_to_steps)):
            if len(index_to_steps[i]) > 1:
                lr(i, question, steps, steps_operators, index_to_steps)
        # for hr in holistic_rules:
        #     hr(question, steps, steps_operators, index_to_steps)
        _break_rule_prefer_not_aligned(question, steps, steps_operators, index_to_steps)
    _break_rule_prefer_exact_match(question, steps, steps_operators, index_to_steps)


def _util_maximal_condition(index:int, question:Doc, steps: List[Doc], steps_operators: List[QDMROperation], index_to_steps: List[Set[int]],
                            condition: Callable[[int, int],bool]):
    if len(index_to_steps[index]) < 2:
        return

    candidates_counts = {x:1 for x in index_to_steps[index]}

    def count_side(index_generator):
        current_candidates = set(candidates_counts.keys())
        for i in index_generator:
            for x in list(current_candidates):
                if condition(i,x):
                    candidates_counts[x] += 1
                else:
                    current_candidates.remove(x)

    def left_index_generator():
        i = index-1
        while i >= 0:
            yield i
            i -= 1

    def right_index_generator():
        i = index+1
        while i < len(index_to_steps):
            yield i
            i += 1

    # count to the left
    count_side(left_index_generator())

    # count to the right
    count_side(right_index_generator())

    sorted_list = sorted(candidates_counts.items(), key=lambda x: x[1], reverse=True)
    if sorted_list[0][1] > sorted_list[1][1]:
        index_to_steps[index] = set([sorted_list[0][0]])


def _break_rule_prefer_sequences(index:int, question:Doc, steps: List[Doc], steps_operators: List[QDMROperation], index_to_steps: List[Set[int]]):
    def condition(index: int, step_id_candidate: int):
        return step_id_candidate in index_to_steps[index]
    _util_maximal_condition(index=index, question=question, steps=steps, steps_operators=steps_operators, index_to_steps=index_to_steps,
                            condition=condition)


def _break_rule_use_references(index:int, question:Doc, steps: List[Doc], steps_operators: List[QDMROperation], index_to_steps: List[Set[int]]):
    def condition(index: int, step_id_candidate: int):
        if step_id_candidate in index_to_steps[index]:
            return True
        references = Decomposition._get_references_ids(str(steps[step_id_candidate]))
        return any([x for x in references if (x-1) in index_to_steps[index]])

    _util_maximal_condition(index=index, question=question, steps=steps, steps_operators=steps_operators, index_to_steps=index_to_steps,
                            condition=condition)


def _break_rule_prefer_not_aligned(question:Doc, steps: List[Doc], steps_operators: List[QDMROperation], index_to_steps: List[Set[int]]):
    aligned = {}
    for i, token in enumerate(question):
        if _ignore(token):
            continue
        if len(index_to_steps[i]) == 1:
            aligned[next(iter(index_to_steps[i]))] = None

    for i in range(len(index_to_steps)):
        if len(index_to_steps[i])>1:
            new = [x for x in index_to_steps[i] if x not in aligned]
            if len(new) == 0:
                continue
            if len(new) == 1:
                aligned[new[0]] = None
            index_to_steps[i] = set(new)


def _break_rule_prefer_exact_match(question:Doc, steps: List[Doc], steps_operators: List[QDMROperation], index_to_steps: List[Set[int]]):
    steps_texts = {i: [x.text.lower() for x in step] for i, step in enumerate(steps)}

    for i in range(len(index_to_steps)):
        if len(index_to_steps[i])>1:
            new = [x for x in index_to_steps[i] if question[i].text.lower() in steps_texts[x]]
            if len(new) == 0:
                continue
            index_to_steps[i] = set(new)