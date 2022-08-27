import logging
import networkx as nx
from spacy.tokens.token import Token

from qdecomp_with_dependency_graphs.evaluation.normal_form.normalization_rules import DecomposeRule, ReferenceToken, run_tests
from qdecomp_with_dependency_graphs.evaluation.normal_form.operations_normalization_rules import OperationDecomposeRule, WrapperDecomposeRule

from qdecomp_with_dependency_graphs.scripts.qdmr_to_program import QDMROperation
import qdecomp_with_dependency_graphs.scripts.qdmr_to_program as qdmr
from qdecomp_with_dependency_graphs.scripts.qdmr_to_logical_form.qdmr_identifier import StepIdentifier

_logger = logging.getLogger(__name__)


class NewWrapperDecomposeRule(OperationDecomposeRule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_identifier = StepIdentifier()
        self._preserved_tokens_map = WrapperDecomposeRule._preserved_tokens_map.copy()
        self.preserved_tokens = [i for v in self._preserved_tokens_map.values() for i in v]

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        try:
            qdmr_step = ' '.join([(f"#{t.get_id()}" if isinstance(t, ReferenceToken) else t.text) for t in doc])
            step = self._step_identifier.identify(qdmr_step)
            self.operation = QDMROperation[(step.operator or 'NONE').upper()]  # todo: deal with QDMR-high-level?
            if self.operation in [QDMROperation.NONE, QDMROperation.SELECT, QDMROperation.PROJECT]:
                return False, None

            if not step.arguments:
                return False, None

            meta = step.meta

            # align args to doc
            arguments_spans = []

            for arg in step.arguments:
                # todo: #REF of project and boolean
                arg = arg.replace('#REF', '').strip()
                arg = qdmr.qdmr_to_prediction(arg)
                arg_tokens = arg.split(' ')
                start_index = 0
                span = None
                while start_index + len(arg_tokens) <= len(doc):
                    if all([at == dt.text for at,dt in zip(arg_tokens, doc[start_index:start_index+len(arg_tokens)])]):
                        span = (start_index, start_index+len(arg_tokens)-1)
                        break
                    start_index += 1
                if span:
                    arguments_spans.append(span)
                else:
                    raise ValueError(f"unexpected args parse: could not find {arg} in {str(doc)}")

            # fix preserved
            self._preserved_tokens = WrapperDecomposeRule.fix_preserved_tokens(preserved_tokens=self._preserved_tokens,
                                                               operation=self.operation)
            return self.update_node(node_id=node_id, graph=graph, params_span=arguments_spans, meta=meta)
        except Exception as ex:
            _logger.debug(self._get_doc_str(doc=doc), exc_info=True)
            return False, None

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            # ("largest @@1@@", ["AGGREGATE[MAX](@@1@@)"]),  # OK
            # ("@@1@@ that is largest", ["AGGREGATE[MAX](@@1@@)"]),  # not ok

            # ("if any @@1@@", ["BOOLEAN[EXIST](@@1@@)"]),  # not ok
            # ("if there is a @@1@@", ["BOOLEAN[EXIST](@@1@@)"]), not ok
        ]


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    NewWrapperDecomposeRule()._test()