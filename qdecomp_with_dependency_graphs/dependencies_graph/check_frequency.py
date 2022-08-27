import json
import os
from dataclasses import dataclass, asdict
from typing import List, Generator, Dict, Tuple, Iterable
from overrides import overrides
import pandas as pd
from ast import literal_eval
import re
import inflect
import networkx as nx
from pathlib import Path

from qdecomp_with_dependency_graphs.utils.dependencies_graph.config.configuration_loader import config, save
from qdecomp_with_dependency_graphs.dependencies_graph.data_types import QDMROperation, StepsDependencies, StepsSpans, TokensDependencies

from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition

import qdecomp_with_dependency_graphs.utils.data_structures as utils
from qdecomp_with_dependency_graphs.utils.graph import get_graph_levels

import logging
_logger = logging.getLogger(__name__)


@dataclass
class DataStructuresParam:
    question_id: str
    gold_steps: List[str] = None
    operators: List[QDMROperation] = None
    steps_spans: StepsSpans = None
    steps_dep: StepsDependencies = None
    tok_dep: TokensDependencies = None
    decomp: Decomposition = None


class BaseMetric:
    def __init__(self):
        self.name = re.sub(r'(?<!^)(?=[A-Z])', '_', self.__class__.__name__).lower()

    def init(self):
        raise NotImplementedError

    def dump_metrics(self, file_base_path, *args, **kwargs):
        raise NotImplementedError()

    def append(self, sample: DataStructuresParam):
        raise NotImplementedError()


########################################
#           Matchers                   #
########################################

class BaseMatcher(BaseMetric):
    def __init__(self):
        super().__init__()
        self.examples = {'dev':[], 'train':[]}
        self.matches = None

    @overrides
    def init(self):
        self.matches = []

    @overrides
    def dump_metrics(self, file_base_path, metadata: pd.DataFrame):
        summary = {}
        metadata['test'] = self.matches
        matches = metadata['test'].sum()
        total = len(metadata)
        summary['matches'] = f'{matches}/{total} ({matches / total * 100:.2f}%)'
        _logger.info(f"{summary['matches']} matches")

        # write matches
        match_df = metadata[metadata['test']]
        match_df.to_csv(f'{file_base_path}.csv', index=False)

        # validation
        sets = metadata['question_id'].apply(lambda x: x.split('_')[1]).unique()
        val = [x for set_ in sets for x in self.examples.get(set_,[])]
        val_df = metadata[metadata['test'] & metadata['question_id'].isin(val)]
        if len(val) != len(val_df):
            missing = [x for x in val if x not in val_df['question_id'].tolist()]
            summary['missing_validation_samples'] = {
                'rate': f'{len(missing)}/{len(val)}',
                'samples': missing
            }
            _logger.info(f'ERROR: missing {summary["missing_validation_samples"]["rate"]} samples: {summary["missing_validation_samples"]["samples"]}')

        # write summary
        with open(f'{file_base_path}__summary.json', 'wt') as f:
            json.dump(summary, f, indent=2)

    @overrides
    def append(self, sample: DataStructuresParam):
        try:
            self.matches.append(self.is_match(sample))
        except Exception as ex:
            _logger.exception(f'{self.name} error while check: {sample.question_id}, {str(ex)}')
            self.matches.append(False)

    def is_match(self, sample: DataStructuresParam) ->bool:
        raise NotImplementedError()


class SingleAlignedTokenMatcher(BaseMatcher):
    @overrides
    def is_match(self, sample: DataStructuresParam) ->bool:
        aligned = [x for _, tok_list in sample.steps_spans.step_tokens_indexes() for x in tok_list]
        return len(aligned) == 1


class SingleTokenSelectMatcher(SingleAlignedTokenMatcher):
    def __init__(self):
        super().__init__()
        self.examples['dev'] = ['ATIS_dev_121', 'CLEVR_dev_993', 'GEO_dev_2', 'GEO_dev_9']

    @overrides
    def is_match(self, sample: DataStructuresParam) -> bool:
        return len(sample.operators) == 1 and sample.operators[0] == QDMROperation.SELECT and \
               super().is_match(sample)


class ImplicitSelectMatcher(BaseMatcher):
    def __init__(self):
        super().__init__()
        self.examples['dev'] = ['CLEVR_dev_3551', 'ATIS_dev_83']

    @overrides
    def is_match(self, sample: DataStructuresParam) ->bool:
        # todo: would be easier without cleaning the steps spans
        tokens = [x.text for x in sample.steps_spans.tokens()]
        for operator, step, (_, texts) in zip(sample.operators, sample.gold_steps, sample.steps_spans.step_spans_text()):
            if operator == QDMROperation.SELECT:
                if not texts and all((x not in tokens) for x in step.split()):
                    return True
        return False


class ImplicitProjectMatcher(BaseMatcher):
    def __init__(self):
        super().__init__()
        self.examples['dev'] = ['COMQA_dev_cluster-856-1']
        self._p = inflect.engine()

    @overrides
    def is_match(self, sample: DataStructuresParam) ->bool:
        # todo: would be easier without cleaning the steps spans
        tokens = [x.text for x in sample.steps_spans.tokens()]
        for operator, step, (_, texts) in zip(sample.operators, sample.gold_steps, sample.steps_spans.step_spans_text()):
            if operator == QDMROperation.PROJECT:
                steps_tok = [x for tok in step.split() for x in [tok, self._p.plural_noun(tok), self._p.singular_noun(tok)]
                             if tok not in ['of', 'return'] if x]
                if not texts and all((x not in tokens) for x in steps_tok):
                    return True
        return False


class MultiCollapseMatcher(BaseMatcher):
    def __init__(self):
        super().__init__()
        self.examples['dev'] = ['NLVR2_dev_dev-900-1-0']

    @overrides
    def is_match(self, sample: DataStructuresParam) ->bool:
        not_aligned_steps = {i for i, tok in sample.steps_spans.step_tokens_indexes() if not tok}
        for step_id in not_aligned_steps:
            refs = [int(x) for x in re.findall(r'#(\d+)', sample.gold_steps[step_id-1])]
            if any(x in not_aligned_steps for x in refs):
                return True
        return False


class SingleTokenToMultipleProjectOrFilter(BaseMatcher):
    def __init__(self):
        super().__init__()
        self.examples['dev'] = ['DROP_dev_history_2088_806017ea-fe47-4515-a485-319ca782f07b', 'ATIS_dev_166']

    @overrides
    def is_match(self, sample: DataStructuresParam) ->bool:
        not_aligned_steps = {i for i, tok in sample.steps_spans.step_tokens_indexes() if not tok}
        steps_to_cont = {i+1: re.sub(r'#\d+', '', x) for i, x in enumerate(sample.gold_steps)}
        cont_to_steps = utils.swap_keys_and_values_dict(steps_to_cont)
        question = ' '.join(x.text for x in sample.steps_spans.tokens())
        for cont, steps in cont_to_steps.items():
            if len(steps)>=2 and sample.operators[steps[0]-1] in [QDMROperation.PROJECT, QDMROperation.FILTER]:
                if any(question.count(x)==1 for x in cont.split()):
                    return True
        return False


class NotAlignedLastStep(BaseMatcher):
    def __init__(self):
        super().__init__()
        self.examples['dev'] = ['CLEVR_dev_2090']

    @overrides
    def is_match(self, sample: DataStructuresParam) ->bool:
        not_aligned_steps = {i for i, tok in sample.steps_spans.step_tokens_indexes() if not tok}
        return len(sample.gold_steps) in not_aligned_steps


# Regex #################################################

class RegexMatcher(BaseMatcher):
    def __init__(self, operator:QDMROperation, patterns: List[str]):
        super().__init__()
        self.operator = operator
        self.patterns = patterns

    @overrides
    def is_match(self, sample: DataStructuresParam) ->bool:
        for step_id, spans_texts in sample.steps_spans.step_spans_text():
            step = sample.gold_steps[step_id-1]
            if sample.operators[step_id-1] == self.operator:
                for p in self.patterns:
                    match = re.search(p, step)
                    if match:
                        if not spans_texts:
                            return True
                        if any(all(g not in s for s in spans_texts) for g in match.groups()):
                            return True
        return False


class BooleanExistenceAndComparisonMatcher(RegexMatcher):
    def __init__(self):
        super().__init__(operator=QDMROperation.BOOLEAN, patterns=[
            r'(?:at least|equal to)\s([^#]+)',
            r'(?:is|are)\s(true|false)'
        ])
        self.examples['dev'] = ['NLVR2_dev_dev-640-1-1', 'NLVR2_dev_dev-900-1-0']


################################################
#                 Metrics                      #
################################################
class MultiCollapseSequencesCounter(BaseMetric):
    def __init__(self):
        super().__init__()
        self._sequences_types = None

    @overrides
    def init(self):
        self._sequences_types = {}

    @overrides
    def dump_metrics(self, file_base_path, *args, **kwargs):
        data = [
            {'sequence': k, 'sequence_len': len(k), 'samples': len(v), 'question_ids': v}
        for k,v in self._sequences_types.items()]
        df = pd.DataFrame.from_records(data)
        df.to_csv(file_base_path+'.csv', index=False)

        deps_order, total, satisfied = self._get_order(data)
        with open(f'{file_base_path}__deps_order.json', 'wt') as f:
            json.dump({
                'deps_order': deps_order,
                'total': total,
                'satisfied': satisfied}, f, indent=2)

    def _get_order(self, data):
        # find dependencies order greedy
        data_sorted = sorted([x for x in data if x['sequence_len']>1], key=lambda x: x['sequence_len'], reverse=True)
        pred_graph = nx.DiGraph()
        total_samples = sum(x['samples'] for x in data_sorted)
        successfully = 0
        for sequence_data in data_sorted:
            full_success = True
            sequence = [x for x in sequence_data['sequence'] if x not in ['None', 'select']]
            for x, y in zip(sequence, sequence[1:]):
                if x == y:
                    continue
                pred_graph.add_edge(x,y)
                if not nx.is_directed_acyclic_graph(pred_graph):
                    full_success = False
                    pred_graph.remove_edge(x, y)
            if full_success:
                successfully += sequence_data['samples']
        _, levels = get_graph_levels(pred_graph)
        deps = [list(sorted(x)) for _, x in sorted(levels.items(), key=lambda itm: itm[0])]
        return deps, total_samples, successfully

    @overrides
    def append(self, sample: DataStructuresParam) ->bool:
        not_aligned_steps = {i for i, tok in sample.steps_spans.step_tokens_indexes() if not tok}
        sequences = {}
        for step_id in not_aligned_steps:
            refs = [int(x) for x in re.findall(r'#(\d+)', sample.gold_steps[step_id-1])]
            sequences[step_id] = [(sample.operators[step_id-1],)+x for r in refs for x in sequences.get(r, []) if r in sequences] or \
                                 [(sample.operators[step_id-1],)]

        for x in set(y for v in sequences.values() for y in v):
            if x not in self._sequences_types:
                self._sequences_types[x] = []
            self._sequences_types[x].append(sample.question_id)


class CountDependencies(BaseMetric):
    def __init__(self):
        super().__init__()
        self.dependencies = None

    @overrides
    def init(self):
        self.dependencies = []

    @overrides
    def dump_metrics(self, file_base_path, *args, **kwargs):
        labels = sorted(set(self.dependencies))
        with open(file_base_path+'.txt', 'wt') as fp:
            fp.write('\n'.join(labels))
        with open(file_base_path + '_summary.txt', 'wt') as fp:
            fp.write(f"different labels: {len(labels)}")

    def append(self, sample: DataStructuresParam):
        self.dependencies.extend(data.dep_type for _,_, data in sample.tok_dep.dependencies())


class CountSpecialTokensUsage(BaseMetric):
    def __init__(self, special_tokens: List[str] = ['[DUP]', '[DUM]']):
        super().__init__()
        self.counts = {x: None for x in special_tokens}

    @overrides
    def init(self):
        self.counts = {k: [] for k in self.counts}

    @overrides
    def dump_metrics(self, file_base_path, *args, **kwargs):
        counts = {}
        for k, v in self.counts.items():
            v = utils.list_to_index_dict(v)
            counts[k] = {x: len(y) for x, y in v.items()}

        with open(f'{file_base_path}_{"_".join(sorted(self.counts.keys()))}.txt', 'wt') as fp:
            json.dump(counts, fp, indent=2, sort_keys=True)

    def append(self, sample: DataStructuresParam):
        indices = {k: [i for i, t in enumerate(sample.tok_dep.tokens()) if t.text == k] for k in self.counts}
        used_indices = {k: set([]) for k in self.counts}
        for u,v,*_ in sample.tok_dep.dependencies():
            for k, ind in indices.items():
                if u in ind: used_indices[k].add(u)
                if v in ind: used_indices[k].add(v)
        for k, c in self.counts.items():
            c.append(len(used_indices[k]))


def run_metrics(dest_dir: str,
                gold: pd.DataFrame,
                preds_getter: Generator[DataStructuresParam, None, None],
                metrics: List[BaseMetric]):
    _logger.info(f'{dest_dir}:')
    metrics: Iterable[BaseMetric] = sorted(metrics, key=lambda x: x.name)

    os.makedirs(dest_dir, exist_ok=True)
    save(dest_dir)

    # initialize metrices
    for x in metrics:
        x.init()

    # apply on samples
    q_ids = []
    tokens_dependencies = []
    for sample in preds_getter:
        q_ids.append(sample.question_id)
        tokens_dependencies.append(sample.tok_dep)
        for x in metrics:
            try:
                x.append(sample)
            except Exception as ex:
                _logger.exception(f'error while check: {sample.question_id} ({x.name}), {str(ex)}')

    matadata_df = gold[gold['question_id'].isin(q_ids)].copy()
    assert len(matadata_df) == len(q_ids)

    # add tokens dependencies representation
    matadata_df['graph_tokens'] = [(str([(i, x.text) for i, x in enumerate(g.tokens())]) if g else '') for g in tokens_dependencies]
    matadata_df['graph_dependencies'] = [(str([(u,v,str(d.dep_type)) for u,v,d in g.dependencies()]) if g else '') for g in tokens_dependencies]

    for x in metrics:
        _logger.info(f'run {x.name}')
        os.makedirs(dest_dir, exist_ok=True)
        x.dump_metrics(os.path.join(dest_dir, x.name), metadata=matadata_df)
        _logger.info('---------')

    unify_matchers_output(metrics, dest_dir)


def gold_tokens_dependencies_generator(df: pd.DataFrame) -> Generator[DataStructuresParam, None, None]:
    spans_extractor = config.spans_extractor
    token_dep_extractor = config.tokens_dependencies_extractor
    token_dep_to_qdmr_extractor = config.tokens_dependencies_to_qdmr_extractor

    for index, row in df.iterrows():
        question_id, question_text, decomposition, operators, *_ = row
        operators = [x.strip() for x in literal_eval(operators)]
        gold_steps = [x.strip() for x in re.sub(r'\s+', ' ', decomposition).split(';')]
        param = DataStructuresParam(question_id=question_id, gold_steps=gold_steps, operators=operators)
        try:
            debug = {}
            tokens_dependencies = token_dep_extractor and token_dep_extractor.extract(question_id=question_id,
                                                                                      question=question_text,
                                                                                      decomposition=decomposition,
                                                                                      operators=operators, debug=debug)
            param.steps_spans = debug.get('steps_spans', None)
            param.steps_dep = debug.get('steps_dependencies', None)
            param.tok_dep = tokens_dependencies

            redecomposition = token_dep_to_qdmr_extractor and token_dep_to_qdmr_extractor.extract(tokens_dependencies)
            param.decomp = redecomposition
        except Exception:
            pass
        yield param


def unify_matchers_output(metrics: Iterable[BaseMetric], dir_path: str):
    dest_file_base = os.path.join(dir_path, 'unify_matchers_output')

    # unify matchers csv
    matchers = [x for x in metrics if isinstance(x, BaseMatcher)]
    files = [(x.name, os.path.join(dir_path, f'{x.name}.csv')) for x in matchers]
    files = [(name, path) for name, path in files if os.path.exists(path)]
    if not files:
        return
    name, path = files[0]
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c!='test']
    df[name] = df['test']
    df = df.drop(['test'], axis=1)
    for name, path in files[1:]:
        df_ = pd.read_csv(path)
        df_[name] = df_['test']
        df = pd.merge(df, df_[cols+[name]], on=cols, how='outer', suffixes=('', ''))
    df[cols+list(sorted(x for x,_ in files))].to_csv(dest_file_base+'.csv', index=False)

    # unify summaries
    pathlist = Path(dir_path).glob('*_summary.json')
    summary = {}
    for path in pathlist:
        matcher_name = re.sub(r'_*summary\.json', '', path.name)
        with open(path, 'rt') as fp:
            summary[matcher_name] = json.load(fp)
    with open(dest_file_base+'.json', 'wt') as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    datasets_df = {
        'dev': pd.read_csv(f'datasets/Break/QDMR/dev.csv'),
        'train': pd.read_csv(f'datasets/Break/QDMR/train.csv'),
    }

    for k in ['dev']:  #, 'train']:
        _logger.info(f'{k}:')
        run_metrics(dest_dir=os.path.join('_debug/check_frequency', k),
                    gold=datasets_df[k],
                    preds_getter=gold_tokens_dependencies_generator(df=datasets_df[k]),
                    metrics=[
                        # matchers
                        SingleAlignedTokenMatcher(),
                        SingleTokenSelectMatcher(),
                        ImplicitSelectMatcher(),
                        ImplicitProjectMatcher(),
                        MultiCollapseMatcher(),
                        SingleTokenToMultipleProjectOrFilter(),
                        NotAlignedLastStep(),
                        BooleanExistenceAndComparisonMatcher(),

                        MultiCollapseSequencesCounter(),
                        CountDependencies(),
                        CountSpecialTokensUsage(),
                    ]
                    )



    # m = MultiCollapseSequencesCounter()
    # for set in ['dev', 'train']:
    #     file_base_path= f'_debug/check_frequency/multi_collapse_sequences_counter_{set}'
    #     df = pd.read_csv(file_base_path+'.csv')
    #     df['sequence'] = df['sequence'].apply(lambda x:literal_eval(x))  # str to tuple
    #     data = df.to_records(index=False)
    #
    #     deps_order, total, satisfied = m._get_order(data)
    #     with open(f'{file_base_path}__deps_order.json', 'wt') as f:
    #         json.dump({
    #             'deps_order': deps_order,
    #             'total': int(total),
    #             'satisfied': int(satisfied)}, f, indent=2)