from typing import List
from overrides import overrides
import networkx as nx

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import DependencyType, \
    StepData, StepDependencyData, StepsDependencies
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.steps_dependencies_extractors.base_steps_dependencies_extractor import \
    BaseStepsDependenciesExtractor

from qdecomp_with_dependency_graphs.scripts.qdmr_to_logical_form.qdmr_identifier import QDMRProgramBuilder


class LogicalFormBasedStepsDependenciesExtractor(BaseStepsDependenciesExtractor):
    def __init__(self): #, include_properties: bool = False):
        super().__init__()
        # self.include_properties = include_properties

    @overrides
    def extract(self, question_id: str, question: str, decomposition:str, operators: List[str] = None,
                debug: dict = None) -> StepsDependencies:
        builder = QDMRProgramBuilder(decomposition)
        builder.build()

        dependencies_graph = nx.MultiDiGraph()
        for i, step in enumerate(builder.steps):
            #todo: deal with None
            step_id = i+1
            dependencies_graph.add_node(step_id, data=StepData(step.step))
            # operator = step.operator
            # if self.include_properties:
            #     properties = f'[{",".join(step.properties)}]' if step.properties else ""
            #     operator = f'{step.operator}{properties}'

            for arg_type, refs in step.get_references().items():
                dependencies_graph.add_edges_from([
                    (step_id, r, {'data': StepDependencyData(dep_type=DependencyType(arg_type), properties=step.properties)}) for r in refs
                ])

        return StepsDependencies(dependencies_graph)
