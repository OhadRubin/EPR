from overrides import overrides

import qdecomp_with_dependency_graphs.utils.data_structures as util

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import DependencyType, SpansDependencies, SpansData
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_extractors.collapsers.base_collapser import BaseCollapser


class JoinCollapser(BaseCollapser):
    @overrides
    def collapse(self, spans_dependencies: SpansDependencies, decomposition: str= None) -> None:
        empty_steps_ids = [i for i,data in spans_dependencies.steps() if not (data and data.spans_list)]

        dependencies_graph = spans_dependencies._dependencies_graph
        for i in empty_steps_ids:
            outgoing_edges_data = dependencies_graph.out_edges(i, data=True)

            # outgoing edges - copy to predecessors
            assert dependencies_graph.in_degree(i) > 0, f"could not collapse step {i}: no predecessors"
            for pred_id in dependencies_graph.predecessors(i):
                dependencies_graph.add_edges_from((pred_id, v, data) for u, v, data in outgoing_edges_data)

            # ingoing edges - replace with subject
            cur_step_main_successor_candidates = [(v, data) for u, v, data in outgoing_edges_data
                                                  if _is_main_dependency(data['data'].dep_type)]
            assert len(set(x for x, _ in cur_step_main_successor_candidates)) == 1, \
                        f'could not find a single main successor of step {i} among {cur_step_main_successor_candidates}'
            cur_step_main_successor, _ = cur_step_main_successor_candidates[0]
            for u, v, data in dependencies_graph.in_edges(i, data=True):
                dependencies_graph.add_edge(u, cur_step_main_successor, **data)

            # remove node
            dependencies_graph.remove_node(i)

    @overrides
    def unwind(self, spans_dependencies: SpansDependencies) -> None:
        dependencies_graph = spans_dependencies._dependencies_graph

        for n in list(dependencies_graph.nodes):
            outgoing_edges = dependencies_graph.out_edges(n, keys=True, data=True)
            dependencies_to_edges = util.list_to_multivalue_dict(outgoing_edges, key=lambda x: x[3]['data'].dep_type)

            def has_more(*dependencies_types: DependencyType) -> bool:
                match_case = all(dp in dependencies_to_edges for dp in dependencies_types)
                extra_dep = len(dependencies_to_edges) > len(dependencies_types)
                different_operator = all(self._get_operator(x) != self._get_operator(dependencies_types[0])
                               for x in dependencies_to_edges.keys() if x not in dependencies_types)
                return match_case and extra_dep and different_operator

            def try_unwind(*dependencies_types: DependencyType, allow_multiple=True):
                """
                Try to unwind a group of dependencies by extract them to a new separated node (i.e change their parent
                to be that new node). We point on point on the new node with all the outgoing dependencies that
                currently point on the node that the "main dependency" points on.
                If allow_multiple=True, treat multiple arcs with the same dependencies as multiple references of a single
                argument group - i.e a single operator. Otherwise, create a separated step for each of the arcs.
                e.g: allow_multiple=False: x-[aggregate-sub]->y-[aggregate-sub]->z  (x,y are new)
                     allow_multiple=True: x-[arithmetic-arg]->y, x-[arithmetic-arg]->z  (x is new)
                """
                new_node_id = dependencies_graph.number_of_nodes() + 1
                if not has_more(*dependencies_types):
                    return
                # todo: multiple dependencies from the same type - distinguish between multiple steps and multiple
                #  arg references
                assert allow_multiple or (
                        len(dependencies_types) == 1 or
                        all(len(dependencies_to_edges[x]) == 1 for x in dependencies_types)
                ),  f'ambiguous dependencies to unwind (multiple options) {dependencies_to_edges}'
                dependencies_graph.add_node(new_node_id, data=SpansData([]))

                # point from new node
                # if allow_multiple, copy all the dependencies of a given type, else just the first one
                last_dep_ind = None if allow_multiple else 1
                for dep in dependencies_types:
                    to_delete = []
                    for i, (u, v, k, data) in enumerate(list(dependencies_to_edges[dep][0:last_dep_ind])):
                        dependencies_graph.add_edge(new_node_id, v, **data)
                        dependencies_graph.remove_edge(u, v, k)
                        to_delete.append(i)

                    for i in reversed(to_delete):
                        del dependencies_to_edges[dep][i]

                    if not dependencies_to_edges[dep]:
                        del dependencies_to_edges[dep]

                # point to new node
                main_type = [x for x in dependencies_types if _is_main_dependency(x)]
                assert len(main_type) == 1, f"there is no main dependency in types {dependencies_types}"
                main_type = main_type[0]
                main_node_candidates = set(v for _, v, data in dependencies_graph.out_edges(new_node_id, data=True)
                                           if data['data'].dep_type == main_type)
                assert len(main_node_candidates) == 1, f'ambiguous candidate for main node {main_node_candidates}'
                main_node = main_node_candidates.pop()
                for data in dependencies_graph.get_edge_data(n, main_node).values():
                    new_k = dependencies_graph.add_edge(n, new_node_id, **data)
                    # update dependencies_to_edges
                    dep = data['data'].dep_type
                    dependencies_to_edges[dep].append((n, new_node_id, new_k, data))
                    dependencies_to_edges[dep] = [x for x in dependencies_to_edges[dep]
                                                  if (x[0], x[1]) != (n, main_node)]  # we are going to remove edge (n, main_node)
                    if not dependencies_to_edges[dep]: del dependencies_to_edges[dep]

                dependencies_graph.remove_edge(n, main_node)

            # unwind by constant order
            unwind_order = [
                [DependencyType.DISCARD_SUB, DependencyType.DISCARD_EXCLUDE],
                [DependencyType.PROJECT_SUB, DependencyType.PROJECT_PROJECTION],
                [DependencyType.PROJECT_SUB],
                [DependencyType.FILTER_SUB, DependencyType.FILTER_CONDITION],
                [DependencyType.SORT_SUB, DependencyType.SORT_ORDER],
                [DependencyType.GROUP_VALUE, DependencyType.GROUP_KEY],
                [DependencyType.SUPERLATIVE_SUB, DependencyType.SUPERLATIVE_ATTRIBUTE],
                [DependencyType.COMPARATIVE_SUB, DependencyType.COMPARATIVE_ATTRIBUTE],
                [DependencyType.COMPARATIVE_CONDITION],
                [DependencyType.INTERSECTION_PROJECTION, DependencyType.INTERSECTION_INTERSECTION],
                [DependencyType.AGGREGATE_ARG],
                # # [DependencyType.ARITHMETIC_ARG],
                [DependencyType.BOOLEAN_SUB, DependencyType.BOOLEAN_CONDITION],
                [DependencyType.BOOLEAN_SUB],
                [DependencyType.COMPARISON_ARG],
                # # [DependencyType.UNION_SUB],
            ]
            for x in unwind_order:
                prev_nodes_count = 0
                while dependencies_graph.number_of_nodes() > prev_nodes_count:
                    prev_nodes_count = dependencies_graph.number_of_nodes()
                    allow_multiple = (DependencyType.AGGREGATE_ARG not in x)
                    try_unwind(*x, allow_multiple=allow_multiple)


def _is_main_dependency(dependency: DependencyType):
    return dependency in \
           [
               DependencyType.AGGREGATE_ARG,
               # DependencyType.ARITHMETIC_ARG,
               DependencyType.BOOLEAN_SUB,
               DependencyType.COMPARATIVE_SUB,
               DependencyType.DISCARD_SUB,
               DependencyType.FILTER_SUB,
               DependencyType.GROUP_VALUE,
               DependencyType.INTERSECTION_PROJECTION,
               DependencyType.PROJECT_SUB,
               DependencyType.SORT_SUB,
               DependencyType.SUPERLATIVE_SUB,
               # DependencyType.UNION_SUB
            ]
    # or dependency.value.endswith('-sub')