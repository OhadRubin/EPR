from collections import defaultdict
from typing import Tuple

from overrides import overrides

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies, SpanDependencyData, SpansData
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_extractors.collapsers.base_collapser import BaseCollapser


class ConcatCollapser(BaseCollapser):
    @overrides
    def collapse(self, spans_dependencies: SpansDependencies, decomposition: str= None) -> None:
        empty_steps_ids = [i for i,data in spans_dependencies.steps() if not (data and data.spans_list)]

        dependencies_graph = spans_dependencies._dependencies_graph
        for i in empty_steps_ids:
            outgoing_edges_data = dependencies_graph.out_edges(i, data=True)

            # outgoing edges - copy to predecessors
            assert dependencies_graph.in_degree(i) > 0, f"could not collapse step {i}: no predecessors"
            for pred_id in dependencies_graph.predecessors(i):
                dependencies_graph.add_edges_from((pred_id, v, {'data': SpanDependencyData(dep_type='&'.join([data['data'].dep_type, pred_data['data'].dep_type]))})
                                                  for u, v, data in outgoing_edges_data
                                                  for pred_data in dependencies_graph.get_edge_data(pred_id, i).values())

            # remove node
            dependencies_graph.remove_node(i)

    @overrides
    def unwind(self, spans_dependencies: SpansDependencies) -> None:
        dependencies_graph = spans_dependencies._dependencies_graph

        def add_node(current_node, dependencies_seq):
            new_node_id = dependencies_graph.number_of_nodes() + 1
            dependencies_graph.add_node(new_node_id, data=SpansData([]))
            new_k = dependencies_graph.add_edge(current_node, new_node_id, data=SpanDependencyData(dep_type=dependencies_seq[0]))
            suffix = dependencies_seq[1:]
            operator = self._get_operator(dependencies_seq[0])
            suffix_to_arcs[len(suffix)][suffix][operator].append(
                (dependencies_seq, (n, new_node_id, new_k)))
            return new_node_id

        for n in list(dependencies_graph.nodes):
            outgoing_edges = dependencies_graph.out_edges(n, keys=True, data=True)

            # map of {dependencies sequence length: {dependencies sequence suffix : {first operator: (sequence, edge)}}}
            suffix_to_arcs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            for u, v, k, data in outgoing_edges:
                dependencies: Tuple[str] = tuple([x for x in data['data'].dep_type.split('&')])
                if len(dependencies) > 1:  # collapsed dependencies
                    suffix = dependencies[1:]
                    operator = self._get_operator(dependencies[0])
                    suffix_to_arcs[len(suffix)][suffix][operator]\
                        .append((dependencies, (u, v, k)))

            max_length = max(suffix_to_arcs.keys(), default=0)
            for i in range(max_length, 0, -1):
                if i not in suffix_to_arcs: continue
                for suffix, operator_map in suffix_to_arcs[i].items():
                    for operator, deps in operator_map.items():
                        common_new = (len(deps)>1 and len(set(x[0] for x in deps))>1)
                        if common_new:
                            new_node_id = add_node(n, suffix)

                        # update dependencies
                        for dep, (u, v, k) in deps:
                            if not common_new:
                                new_node_id = add_node(n, suffix)

                            dependencies_graph.add_edge(new_node_id, v, data=SpanDependencyData(dep_type=dep[0]))
                            dependencies_graph.remove_edge(u, v, k)
