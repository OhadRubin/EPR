from overrides import overrides
import re

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_extractors.collapsers.base_collapser import BaseCollapser


class AddOperatorsPropertiesCollapser(BaseCollapser):
    @overrides
    def collapse(self, spans_dependencies: SpansDependencies, decomposition: str= None) -> None:
        for _, _, data in spans_dependencies.dependencies():
            if data.properties:
                data.dep_type = f'{data.dep_type}[{",".join(data.properties)}]'

    @overrides
    def unwind(self, spans_dependencies: SpansDependencies) -> None:
        for _, _, data in spans_dependencies.dependencies():
            regx = re.search(r'(.*)\[(.+)\]', data.dep_type)
            if regx:
                dep, prop = regx.groups()
                data.dep_type = dep
                data.properties = prop.split(",")
