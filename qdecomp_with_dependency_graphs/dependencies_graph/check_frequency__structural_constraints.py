from dataclasses import dataclass
from typing import List, Dict, Iterable, Generator, Tuple, Set, Callable
from overrides import overrides
import pandas as pd
import re
import os

from qdecomp_with_dependency_graphs.dependencies_graph.check_frequency import BaseMatcher, BaseMetric, DataStructuresParam, run_metrics, \
    unify_matchers_output
from qdecomp_with_dependency_graphs.dependencies_graph.data_types import QDMROperation, StepsDependencies, StepsSpans, TokensDependencies, \
    TokenDependencyData, TokenDependencyType
from qdecomp_with_dependency_graphs.dependencies_graph.examine_predictions import json_file_to_dependencies_graphs, prediction_to_dependencies_graph, \
    gold_to_dependencies_graph

from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition

import qdecomp_with_dependency_graphs.utils.data_structures as utils

import logging
_logger = logging.getLogger(__name__)


##################################
#           Utils                #
##################################

def get_operator(tag: str):
    return get_parts(tag, [0, 2])


def get_parts(tag: str, part_ind: List[int]):
    res = []
    for tag_ in tag.split('&'):
        parts = re.match(r'(\w+)(?:-(\w+))?(?:(\[.*\]))?', tag_).groups()
        res.extend([x for i, x in enumerate(parts) if i in part_ind])
    return ''.join([x for x in res if x])

########################################
#           Matchers                   #
########################################

class SingleTagViolationMatcher(BaseMatcher):
    @overrides
    def is_match(self, sample: DataStructuresParam) ->bool:
        dep_tags: Dict[int, Iterable[Tuple[int, int, TokenDependencyData]]] = utils.list_to_multivalue_dict(sample.tok_dep.dependencies(), key=lambda x: (x[0], x[1]))
        return any(len(x) > 1 for _, x in dep_tags.items())


class SpanLeftToRightViolationMatcher(BaseMatcher):
    @overrides
    def is_match(self, sample: DataStructuresParam) ->bool:
        deps = sample.tok_dep.dependencies()
        return any(i >= j for i, j, d in deps if d.dep_type == TokenDependencyType.SPAN)


class SpanSingleInOutViolationMatcher(BaseMatcher):
    @overrides
    def is_match(self, sample: DataStructuresParam) ->bool:
        out_deps = sample.tok_dep.tokens_dependencies()
        in_deps = sample.tok_dep.tokens_dependencies(in_dependencies=True)
        for x, deps in [*list(out_deps), *list(in_deps)]:
            if len([d for _, _, d in deps if d.dep_type == TokenDependencyType.SPAN]) > 1:
                return True
        return False


class DuplicateFromDUPToProperViolationMatcher(BaseMatcher):
    @overrides
    def is_match(self, sample: DataStructuresParam) ->bool:
        deps = sample.tok_dep.dependencies()
        tokens = sample.tok_dep.tokens()
        return any((tokens[i].text.upper() != '[DUP]' or tokens[j].text.upper() in ['[RSC]', '[DUP]', '[DUM]'])
                   for i, j, d in deps if d.dep_type == 'duplicate')


class DuplicateFromUsedDUPViolationMatcher(BaseMatcher):
    @overrides
    def is_match(self, sample: DataStructuresParam) ->bool:
        out_deps = sample.tok_dep.tokens_dependencies()
        in_deps = sample.tok_dep.tokens_dependencies(in_dependencies=True)
        tokens = sample.tok_dep.tokens()
        for i, token in enumerate(tokens):
            if token.text.upper() == '[DUP]':
                if any(out_deps[i]) or any(in_deps[i]):
                    if all(d.dep_type!='duplicate' for _,_,d in out_deps[i]):
                        return True
        return False


class DuplicateSingleOutViolationMatcher(BaseMatcher):
    @overrides
    def is_match(self, sample: DataStructuresParam) ->bool:
        out_deps = sample.tok_dep.tokens_dependencies()
        for x, deps in out_deps:
            if len([d for _, _, d in deps if d.dep_type == 'duplicate']) > 1:
                return True
        return False


class InconsistencyDependenciesMatcher(BaseMatcher):
    @overrides
    def is_match(self, sample: DataStructuresParam) ->bool:
        out_deps = sample.tok_dep.tokens_dependencies()
        for _, deps in out_deps:
            if len(set(get_operator(d.dep_type) for _, _, d in deps if d.dep_type not in ['duplicate'])) > 1:
                return True
        return False


class SpanSingleRepresentativeViolationMatcher(BaseMatcher):
    @overrides
    def is_match(self, sample: DataStructuresParam) -> bool:
        out_deps = sample.tok_dep.tokens_dependencies()
        in_deps = sample.tok_dep.tokens_dependencies(in_dependencies=True)
        for u, deps in out_deps:
            # if u is a middle-node of a span chain
            if any(d.dep_type == TokenDependencyType.SPAN for _, _, d in deps):
                # single out dependency (='span')
                if len(deps) > 1:
                    return True
                # no incoming dependencies but a single 'span'
                if any(d.dep_type != TokenDependencyType.SPAN for _, _, d in in_deps[u]) or len(in_deps[u]) > 1:
                    return True
        return False


class ConnectivityViolationMatcher(BaseMatcher):
    @overrides
    def is_match(self, sample: DataStructuresParam) -> bool:
        tokens = sample.tok_dep.tokens()
        out_deps = sample.tok_dep.tokens_dependencies()
        in_deps = sample.tok_dep.tokens_dependencies(in_dependencies=True)
        roots = []
        for u, _ in enumerate(tokens):
            # if u is not a middle-node of a span chain
            in_deps_but_span = [d for _, _, d in in_deps[u] if d.dep_type != TokenDependencyType.SPAN]
            if all(d.dep_type != TokenDependencyType.SPAN for _, _, d in out_deps[u]) \
                    and (len(in_deps_but_span) == 0) \
                    and (any(in_deps[u]) or any(out_deps[u])):
                roots.append(u)
        return len(roots) > 1


class BoundAllowedCombinationsViolationMatcher(BaseMatcher):
    @dataclass
    class Param:
        patten: str
        count_condition: Callable[[int], bool]

    combinations = [
        Param(r'^(?!union|boolean).*-sub.*', lambda x: x <= 1),
        Param(r'union-sub', lambda x: x != 1),
        Param(r'aggregate-arg(.*)', lambda x: x <= 1),
        Param(r'comparison-arg(.*)', lambda x: x != 1),
        Param(r'intersection-intersection(.*)', lambda x: x != 1),
        Param(r'superlative-sub(.*)', lambda x: x <= 1),
        Param(r'superlative-attribute(.*)', lambda x: x <= 1),
    ]

    @overrides
    def is_match(self, sample: DataStructuresParam) -> bool:
        out_deps = sample.tok_dep.tokens_dependencies()
        for _, deps in out_deps:
            for c in self.combinations:
                if not c.count_condition(len([d.dep_type for _,_,d in deps if re.match(c.patten, d.dep_type)])):
                    return True
        return False


class AllowedCombinationsViolationMatcher(BaseMatcher):
    """
    If a certain dependency/ies exists, ensure their complete form holds.
    Examples:
        * arithmetic-right => arithmetic-left, arithmetic-right
        * filter-condition => filter-sub, filter-condition
        * group-key, group-value
    """
    @dataclass
    class Param:
        # pattern to holds for some dependency. Might contain groups to inject in combination_patterns.
        trigger_patten: str
        # combination patterns. might contain groups.
        combination_patterns: List[str]
        # allow one of the dependency be missing (=implicit, embedded in step)
        allow_implicit: bool = False

    combinations = [
        Param(r'(.*)-(?:left|right)(.*)', [r'\1-left\2', r'\1-right\2']),  # add allow_implicit?
        Param(r'(boolean|filter)-condition(.*)', [r'\1-sub\2', r'\1-condition\2']),
        Param(r'(comparative)-(?:sub|attribute)(.*)', [r'\1-sub\2', r'\1-attribute\2']),
        Param(r'(comparative)-condition(.*)', [r'\1-sub\2', r'\1-attribute\2', r'\1-condition\2']),
        Param(r'(discard)-(?:sub|exclude)(.*)', [r'\1-sub\2', r'\1-exclude\2'], allow_implicit=True),
        Param(r'(group)-(?:key|value)(.*)', [r'\1-key\2', r'\1-value\2'], allow_implicit=True),
        Param(r'(intersection)-projection(.*)', [r'\1-intersection\2', r'\1-projection\2']),
        Param(r'(sort)-order(.*)', [r'\1-sub\2', r'\1-order\2']),
        Param(r'(superlative)-(?:attribute|sub)(.*)', [r'\1-sub\2', r'\1-attribute\2']),
    ]

    @staticmethod
    def get_variations(dependency: str, pattern: Param):
        if re.match(pattern.trigger_patten, dependency):
            # fill combinations with appropriate groups
            return [re.sub(pattern.trigger_patten, x, dependency) for x in pattern.combination_patterns]
        return None

    @overrides
    def is_match(self, sample: DataStructuresParam) -> bool:
        out_deps = sample.tok_dep.tokens_dependencies()
        for _, deps in out_deps:
            deps_str = [d.dep_type for _, _, d in deps]
            for c in self.combinations:
                for x in deps_str:
                    required_deps = self.get_variations(x, c) or []
                    if not required_deps: continue
                    if c.allow_implicit:
                        # all but one (maybe) should be exist
                        sat = [y for y in required_deps if y in deps_str]
                        if len(sat) < len(required_deps)-1:
                            return True
                    elif any(y not in deps_str for y in required_deps):
                        return True
        return False

################################################
#                 Metrics                      #
################################################
class DependenciesCombinationCounter(BaseMetric):
    def __init__(self):
        super().__init__()
        self._combinations_types: Dict[str, Set[str]] = None

    @overrides
    def init(self):
        self._combinations_types = {}

    @overrides
    def dump_metrics(self, file_base_path, *args, **kwargs):
        data = [
            {'combination': k, 'combination_len': len(k.split(',')), 'samples': len(v)} #, 'question_ids': ','.join(set(v))}
        for k,v in self._combinations_types.items()]
        df = pd.DataFrame.from_records(data)
        df.to_csv(file_base_path+'.csv', index=False)

    @staticmethod
    def compare(files: Dict[str, str], dest_file: str):
        assert len(files) == 2
        names, paths = zip(*files.items())
        df1 = pd.read_csv(paths[0])
        df1[names[0]] = True
        df2 = pd.read_csv(paths[1])
        df2[names[1]] = True
        df = df1.merge(df2, on='combination', how='outer', suffixes=(f'_{names[0]}', f'_{names[1]}'))
        df.to_csv(dest_file, index=False)

    @overrides
    def append(self, sample: DataStructuresParam) -> bool:
        out_deps = sample.tok_dep.tokens_dependencies()
        for u, deps in out_deps:
            combination = ','.join(sorted([str(d.dep_type) for _, _, d in deps]))
            self._combinations_types[combination] = self._combinations_types.get(combination, []) + [sample.question_id]


def gold_tokens_dependencies_generator(gold_file: str)-> Generator[DataStructuresParam, None, None]:
    for question_id, tok_dep in json_file_to_dependencies_graphs(gold_file, gold_to_dependencies_graph):
        yield DataStructuresParam(question_id=question_id, tok_dep=tok_dep)


def predicted_tokens_dependencies_generator(preds_file: str)-> Generator[DataStructuresParam, None, None]:
    for question_id, tok_dep in json_file_to_dependencies_graphs(preds_file, prediction_to_dependencies_graph):
        yield DataStructuresParam(question_id=question_id, tok_dep=tok_dep)


##################

def get_metrices():
    metrics = [
        # matchers
        SingleTagViolationMatcher(),
        SpanLeftToRightViolationMatcher(),
        SpanSingleInOutViolationMatcher(),
        DuplicateFromDUPToProperViolationMatcher(),
        DuplicateFromUsedDUPViolationMatcher(),
        DuplicateSingleOutViolationMatcher(),
        InconsistencyDependenciesMatcher(),
        SpanSingleRepresentativeViolationMatcher(),
        ConnectivityViolationMatcher(),
        BoundAllowedCombinationsViolationMatcher(),
        AllowedCombinationsViolationMatcher(),

        DependenciesCombinationCounter(),
    ]
    return metrics


def run_check_freq(preds_path:str):
    datasets_df = {
        'dev': pd.read_csv(f'datasets/Break/QDMR/dev.csv'),
    }
    metrics = get_metrices()
    data_structure_getters = {
        # 'train_gold': (
        #     pd.read_csv(f'datasets/Break/QDMR/train.csv'),
        #     gold_tokens_dependencies_generator(
        #         'datasets/Break/QDMR/train_dependencies_graph.json'),
        # ),
        'dev_gold': (
            datasets_df['dev'],
            gold_tokens_dependencies_generator('datasets/Break/QDMR/dev_dependencies_graph.json'),
        ),
        'dev_pred': (
            datasets_df['dev'],
            predicted_tokens_dependencies_generator(preds_path)
        )
    }
    for name, (df, preds_getter) in data_structure_getters.items():
        run_metrics(dest_dir=os.path.join('_debug/check_frequency_structural', name),
                    metrics=metrics,
                    gold=df,
                    preds_getter=preds_getter
                    )


def run_check_freq_on_preds_file(preds_file: str, dest_dir: str = None):
    if dest_dir is None:
        dest_dir = os.path.dirname(preds_file)
    dest_dir = os.path.join(dest_dir, 'check_structural')

    run_metrics(dest_dir=dest_dir,
                metrics=get_metrices(),
                gold=pd.read_csv(f'datasets/Break/QDMR/dev.csv'),
                preds_getter=predicted_tokens_dependencies_generator(preds_file),
                )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    preds_path = 'tmp/datasets_Break_QDMR/_best/Break/QDMR/dependencies_graph/biaffine-graph-parser--transformer-encoder/biaffine-graph-parser--transformer-encoder/eval/dv_preds.json'
    run_check_freq(preds_path=preds_path)

    # run_check_freq_on_preds_file(preds_path)