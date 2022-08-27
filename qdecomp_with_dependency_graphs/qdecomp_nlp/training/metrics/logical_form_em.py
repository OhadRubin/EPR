from typing import Dict, Any, Optional, Callable, List, Tuple

import pandas as pd

from allennlp.training.metrics import Average
from overrides import overrides

import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric

from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.evaluate_dep_graph import get_logical_form_tokens_formatters
from qdecomp_with_dependency_graphs.dependencies_graph.examine_predictions import prediction_to_dependencies_graph
from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition
from qdecomp_with_dependency_graphs.scripts.data_processing.create_decomp_logical_form import SpecialTokens, SectorsRep, Processor, \
    get_logical_form_tokens_formatter
from qdecomp_with_dependency_graphs.utils.helpers import silent_logs


@Metric.register("logical_form_em")
class LogicalFormEM(Average):
    def __init__(self,
                 dataset_path: str = 'datasets/Break/QDMR/dev.csv',
                 silent: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self._qdmr_formatter, self._dep_graph_formatter = get_logical_form_tokens_formatters()
        if silent:
            self._qdmr_formatter, self._dep_graph_formatter = silent_logs(self._qdmr_formatter), silent_logs(self._dep_graph_formatter)
        self._dataset = pd.read_csv(dataset_path)

        self._cashed_gold_logical_form = {}
        self._cached_question_and_gold = {}

    @overrides
    def __call__(self, question_id: List[str], predicted_logical_form: List[str]):
        assert len(question_id) == len(predicted_logical_form), \
            f'mismatched lists {len(question_id)}, {len(predicted_logical_form)}'

        for q_id, lf in zip(question_id, predicted_logical_form):
            question_text, decomposition = self._get_question_and_gold(q_id)

            if q_id not in self._cashed_gold_logical_form:
                self._cashed_gold_logical_form[q_id] = self._qdmr_formatter(q_id, question_text, decomposition)
            gold_logical_form = self._cashed_gold_logical_form[q_id]

            super().__call__(int(lf == gold_logical_form and lf != 'ERROR'))

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        """
        # Returns

        The average of all values that were passed to `__call__`.
        """

        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return {'logical_form_em': average_value}

    def _get_question_and_gold(self, question_id: str):
        if question_id not in self._cached_question_and_gold:
            row = self._dataset[self._dataset['question_id'] == question_id].iloc[0]
            question_text = row['question_text']
            decomposition = row['decomposition']
            self._cached_question_and_gold[question_id] = (question_text, decomposition)
        return self._cached_question_and_gold[question_id]



@Metric.register("logical_form_em_for_graph")
class LogicalFormEMForDependenciesGraph(LogicalFormEM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @overrides
    def __call__(self,
                 arcs: List[Tuple[int, int]], arc_tags: List[str],
                 metadata: List[Dict[str, Any]]):
        assert len(arcs)==len(arc_tags)==len(metadata), \
            f'mismatched lists {[len(x) for x in [arcs, arc_tags, metadata]]}'

        question_ids = [x['question_id'] for x in metadata]
        predicted_logical_forms = []
        for q_id, a, at, m in zip(question_ids, arcs, arc_tags, metadata):
            prediction_dict = {
                'arcs': a,
                'arc_tags': at,
                'metadata': {
                    'tokens': [x.text for x in m['tokens']],
                    'pos': [str(x) for x in m['pos']]
                }
            }

            question_text, _ = self._get_question_and_gold(q_id)

            predicted_dependencies_graph = prediction_to_dependencies_graph(prediction_dict=prediction_dict)
            predicted_logical_forms.append(self._dep_graph_formatter(q_id, question_text, predicted_dependencies_graph))

        super().__call__(question_id=question_ids, predicted_logical_form=predicted_logical_forms)


@Metric.register("logical_form_em_for_seq2seq")
class LogicalFormEMForQDMR(LogicalFormEM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @overrides
    def __call__(self,
                 predicted_tokens: List[List[str]],
                 target_tokens: List[List[str]],
                 metadata: List[Dict[str, Any]]
                 ):
        assert len(predicted_tokens) == len(target_tokens), \
            f'mismatched lists {len(predicted_tokens)}, {len(target_tokens)}'

        question_ids = [x['question_id'] for x in metadata]

        predicted_logical_form = []
        for q_id, dec in zip(question_ids, predicted_tokens):
            try:
                dec = ' '.join(dec)
                dec = Decomposition.from_str(dec).to_break_standard_string()
                question_text, _ = self._get_question_and_gold(q_id)
                predicted_logical_form.append(self._qdmr_formatter(q_id, question_text, dec))
            except:
                predicted_logical_form.append('ERROR')

        super().__call__(question_id=question_ids, predicted_logical_form=predicted_logical_form)


@Metric.register("logical_form_em_for_LF_seq2seq")
class LogicalFormEMForLF(LogicalFormEM):
    def __init__(self, formatter: str, **kwargs):
        super().__init__(**kwargs)
        self.processor = Processor.get_by_name(formatter)
        self.lf_formatter = get_logical_form_tokens_formatter(self.processor)

    @overrides
    def __call__(self,
                 predicted_tokens: List[List[str]],
                 target_tokens: List[List[str]],
                 metadata: List[Dict[str, Any]]
                 ):
        assert len(predicted_tokens) == len(target_tokens), \
            f'mismatched lists {len(predicted_tokens)}, {len(target_tokens)}'

        question_ids = [x['question_id'] for x in metadata]

        predicted_logical_form = []
        for q_id, dec in zip(question_ids, predicted_tokens):
            try:
                dec = ' '.join(dec)
                question_text, _ = self._get_question_and_gold(q_id)
                predicted_logical_form.append(self.lf_formatter(q_id, question_text, dec))
            except:
                predicted_logical_form.append('ERROR')

        super().__call__(question_id=question_ids, predicted_logical_form=predicted_logical_form)
