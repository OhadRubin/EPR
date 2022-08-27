from typing import List
from overrides import overrides
import re
import networkx as nx

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import QDMROperation, DependencyType, \
    StepData, StepDependencyData, StepsDependencies
from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.steps_dependencies_extractors.base_steps_dependencies_extractor import \
    BaseStepsDependenciesExtractor


class PatternBasedStepsDependenciesExtractor(BaseStepsDependenciesExtractor):
    @overrides
    def extract(self, question_id: str, question: str, decomposition:str, operators: List[str] = None,
                debug: dict = None) -> StepsDependencies:
        decomposition = re.sub(r'#(\d+)', '@@\g<1>@@', decomposition)
        decomposition = decomposition.split(';')
        operators = [QDMROperation(x) for x in operators]

        dependencies_patterns = {
            QDMROperation.AGGREGATE: [
                [DependencyType.AGGREGATE_ARG]
            ],
            QDMROperation.ARITHMETIC: [
                [DependencyType.ARITHMETIC_LEFT, DependencyType.ARITHMETIC_RIGHT]
            ],
            QDMROperation.BOOLEAN: [
                [DependencyType.BOOLEAN_SUB],
                [DependencyType.BOOLEAN_SUB, DependencyType.BOOLEAN_CONDITION]
            ],
            QDMROperation.COMPARATIVE: [
                [DependencyType.COMPARATIVE_SUB, DependencyType.COMPARATIVE_ATTRIBUTE],
                [DependencyType.COMPARATIVE_SUB, DependencyType.COMPARATIVE_ATTRIBUTE, DependencyType.COMPARATIVE_CONDITION]
            ],
            QDMROperation.COMPARISON: [
                [DependencyType.COMPARISON_ARG, DependencyType.COMPARISON_ARG]
            ],
            QDMROperation.DISCARD: [
                [DependencyType.DISCARD_SUB, DependencyType.DISCARD_EXCLUDE]
            ],
            QDMROperation.FILTER: [
                [DependencyType.FILTER_SUB],
                [DependencyType.FILTER_SUB, DependencyType.FILTER_CONDITION],
            ],
            QDMROperation.GROUP: [
                [DependencyType.GROUP_VALUE, DependencyType.GROUP_KEY],
            ],
            QDMROperation.INTERSECTION: [
                [DependencyType.INTERSECTION_INTERSECTION, DependencyType.INTERSECTION_INTERSECTION],
                [DependencyType.INTERSECTION_PROJECTION, DependencyType.INTERSECTION_INTERSECTION, DependencyType.INTERSECTION_INTERSECTION],
            ],
            QDMROperation.NONE: [
                []
            ],
            QDMROperation.PROJECT: [
                [DependencyType.PROJECT_SUB],
                [DependencyType.PROJECT_PROJECTION, DependencyType.PROJECT_SUB],
            ],
            QDMROperation.SELECT: [
                []
            ],
            QDMROperation.SORT: [
                [DependencyType.SORT_SUB, DependencyType.SORT_ORDER]
            ],
            QDMROperation.SUPERLATIVE: [
                [DependencyType.SUPERLATIVE_SUB, DependencyType.SUPERLATIVE_ATTRIBUTE]
            ],
            QDMROperation.UNION: [
                [DependencyType.UNION_SUB, DependencyType.UNION_SUB],
                [DependencyType.UNION_SUB, DependencyType.UNION_SUB, DependencyType.UNION_SUB],
            ]
        }

        dependencies_graph = nx.MultiDiGraph()
        for i, (step, operator) in enumerate(zip(decomposition, operators)):
            step_id = i+1
            dependencies_graph.add_node(step_id, data=StepData(step))
            refs = Decomposition._get_references_ids(step)
            dep = None
            patterns = dependencies_patterns[operator]
            for pattern in patterns:
                if len(pattern) == len(refs):
                    dep = [(step_id, r, {'data': StepDependencyData(d)}) for r, d in zip(refs, pattern)]
                    break
            if dep is not None:
                dependencies_graph.add_edges_from(dep)
            else:
                raise ValueError(f'could not extract dependencies for {operator.value} step: {step}')
        return StepsDependencies(dependencies_graph)


