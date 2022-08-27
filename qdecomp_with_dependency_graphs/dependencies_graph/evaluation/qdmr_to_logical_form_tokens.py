from typing import List, Dict, Any
import networkx as nx
import re

from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.logical_form_matcher import QDMRStepTokens, QDMRStepTokensDependencies
from qdecomp_with_dependency_graphs.scripts.qdmr_to_logical_form.qdmr_identifier import QDMRStep, QDMROperator, get_identifier, QDMRProgramBuilder
from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpanDependencyData, DependencyType


class QDMRToQDMRStepTokensConverter:
    def __init__(self):
        self._program_converter = QDMRProgramToQDMRStepTokensConverter()

    def convert(self, question_id: str, question_text: str, decomposition: str) -> QDMRStepTokensDependencies:
        builder = QDMRProgramBuilder(decomposition)
        builder.build()
        return self._program_converter.convert(question_id, question_text, builder.steps)


class QDMRProgramToQDMRStepTokensConverter:
    def convert(self, question_id: str, question_text: str, program: List[QDMRStep]) -> QDMRStepTokensDependencies:
        graph = nx.MultiDiGraph()
        for i, step in enumerate(program):
            step_id = i+1
            for arg_name, args in step.arguments.items():
                refs = [x for arg in args for x in re.findall(r"#(\d+)", arg)]
                edges = [(step_id, int(x),
                          dict(data=SpanDependencyData(dep_type=DependencyType(arg_name))))
                         for x in refs]
                graph.add_edges_from(edges)

            graph.add_node(step_id,
                           data=QDMRStepTokens(
                               operator=step.operator,
                               properties=list(step.properties) if step.properties else [],
                               arguments_tokens={k:re.sub(r"#(\d+)","", x).replace('  ', ' ').split()
                                            for k,v in step.arguments.items()
                                            for x in v}
                           ))

        return QDMRStepTokensDependencies(dependencies_graph=graph)


if __name__ == '__main__':
    def graph_str(graph: nx.MultiDiGraph):
        for n_id in graph.nodes:
            print(f"{n_id}. {graph.nodes[n_id]}{graph.out_edges(n_id, data=True)}")
    converter = QDMRToQDMRStepTokensConverter()
    res = converter.convert(None, None,
                      "return flights ;return #1 from  denver ;return #2 to philadelphia ;return #3 if  available")
    graph_str(res)
