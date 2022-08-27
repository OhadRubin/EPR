from collections import defaultdict

from overrides import overrides

import qdecomp_with_dependency_graphs.utils.data_structures as util

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies, SpansData, SpanDependencyData
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_extractors.collapsers.base_collapser import BaseCollapser


class PreSingleToMultipleStepsCollapser(BaseCollapser):
    @overrides
    def collapse(self, spans_dependencies: SpansDependencies, decomposition: str= None) -> None:
        # find conflicts
        index_to_step = defaultdict(set)
        for step_id, data in spans_dependencies.steps():
            for i in [x for s in data.spans_list for x in range(s.start, s.end+1)]:
                index_to_step[i].add(step_id)
        conflicts = {k: v for k, v in index_to_step.items() if len(v) > 1}
        conflicts_groups = conflicts.values()
        # conflicts_groups = set(index_to_step.values())
        # todo: clean if not identical (by decomposition)?

        dependencies_graph = spans_dependencies._dependencies_graph
        next_node_id = max(dependencies_graph.nodes())+1
        for steps_ids in conflicts_groups:
            steps_ids = [x for x in steps_ids if x in dependencies_graph.nodes()]
            if not steps_ids or any(dependencies_graph.out_degree(i)!=1 for i in steps_ids): continue
            out_dep = set(e['data'].dep_type for _,_, e in dependencies_graph.out_edges(steps_ids, data=True))
            if len(out_dep) != 1: continue
            out_dep = out_dep.pop()

            # add pivot
            pivot_id = next_node_id
            next_node_id += 1
            dependencies_graph.add_node(pivot_id, **dependencies_graph.nodes(data=True)[steps_ids[0]])

            # point to pivot
            for step_id in steps_ids:
                # if step_id not in dependencies_graph.nodes(): continue
                step_successor = list(dependencies_graph.successors(step_id))
                assert len(step_successor) == 1, f"expected only one successor for step {step_id}, got {len(step_successor)}"
                step_successor = step_successor[0]
                for u, _, data in dependencies_graph.in_edges(step_id, data=True):
                    dependencies_graph.add_edge(u, step_successor, **data)
                    dependencies_graph.add_edge(u, pivot_id, data=SpanDependencyData(dep_type=f"pre_{out_dep}_{data['data'].dep_type}"))
                dependencies_graph.remove_node(step_id)

            # clean pre edges
            for i, data in dependencies_graph.nodes(data=True):
                out_edges = dependencies_graph.out_edges(i, data=True, keys=True)
                pre_edges = [(u,v,k,data) for (u,v,k,data) in out_edges if data['data'].dep_type.startswith('pre_')]
                if not pre_edges: continue

                # omit unnecessary arg part

                # remove duplicate pre
                identical_dep = util.list_to_multivalue_dict(pre_edges, lambda x: x[3]['data'].dep_type)
                for identical_group in identical_dep.values():
                    for u,v,k,_ in identical_group[1:]:
                        dependencies_graph.remove_edge(u,v,k)

        # remove remain conflicts
        for i, steps in index_to_step.items():
            remain_steps = [x for x in steps if x in dependencies_graph.nodes()]
            if len(remain_steps)>1:
                for s_id in remain_steps:
                    data: SpansData = dependencies_graph.nodes(data=True)[s_id]['data']
                    data.remove_index(i)


    @overrides
    def unwind(self, spans_dependencies: SpansDependencies) -> None:
        dependencies_graph = spans_dependencies._dependencies_graph
        next_node_id = max(dependencies_graph.nodes())+1

        for n_id in list(dependencies_graph.nodes()):
            if n_id not in dependencies_graph.nodes(): continue
            out_edges = dependencies_graph.out_edges(n_id, data=True, keys=True)
            pre_edges = [(u, v, k, data) for (u, v, k, data) in out_edges if data['data'].dep_type.startswith('pre_')]

            out_edges_by_arg = util.list_to_multivalue_dict(out_edges, lambda x: x[3]['data'].dep_type)
            # unwind pre dependencies
            for _, v, vk, vdata in pre_edges:
                pre_dep: SpanDependencyData = vdata['data']
                _, dep_part, arg_part = pre_dep.dep_type.split('_')

                for _, w, wk, wdata in out_edges_by_arg[arg_part]:
                    # create pivot
                    pivot_id = next_node_id
                    next_node_id += 1
                    pivot_data = dependencies_graph.nodes(data=True)[v]
                    dependencies_graph.add_node(pivot_id, **pivot_data)

                    # points on pivot
                    dependencies_graph.add_edge(n_id, pivot_id, **wdata)
                    dependencies_graph.add_edge(pivot_id, w, data=SpanDependencyData(dep_type=dep_part))
                    dependencies_graph.remove_edge(n_id, w)

                # remove pre edge
                dependencies_graph.remove_edge(n_id, v, vk)
                # check if it is the last pre edge
                if dependencies_graph.in_degree(v) == 0:
                    dependencies_graph.remove_node(v)
