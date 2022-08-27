from collections import defaultdict

from overrides import overrides

import qdecomp_with_dependency_graphs.utils.data_structures as util

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies, SpansData, Span, SpanDependencyData
from qdecomp_with_dependency_graphs.dependencies_graph.data_types.steps_spans import TokenData
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_extractors.collapsers.base_collapser import BaseCollapser


class DupSingleToMultipleStepsCollapser(BaseCollapser):
    def __init__(self, count=5):
        super().__init__()
        self.separator = '[DUP]'
        self.dependency = 'duplicate'
        self.additional_tokens = [self.separator for _ in range(count)]

    @overrides
    def collapse(self, spans_dependencies: SpansDependencies, decomposition: str= None) -> None:
        offset = len(spans_dependencies._tokens)
        spans_dependencies._tokens = list(spans_dependencies._tokens) + [
            TokenData(text=x, bio_tag=None)
            for x in self.additional_tokens
        ]

        # find conflicts
        index_to_step = defaultdict(set)
        for step_id, data in spans_dependencies.steps():
            for i in [x for s in data.spans_list for x in range(s.start, s.end+1)]:
                index_to_step[i].add(step_id)
        conflicts = {k: tuple(sorted(v)) for k, v in index_to_step.items() if len(v) > 1}

        # clean if not identical (by decomposition)?
        # conflicts_steps = set([x for v in conflicts.values() for x in v])
        # steps_pattern_to_index = util.list_to_index_dict([re.sub(r'#([0-9]+)', '#ref', x) for x in decomposition.split(';')])
        # conflicts_groups = []
        # for i, v in steps_pattern_to_index.items():
        #     v = [x+1 for x in v if (x+1) in conflicts_steps]
        #     if len(v)>1: conflicts_groups.append(v)

        # remove duplicates
        # conflicts_groups.sort()
        # conflicts_groups = list(k for k, _ in itertools.groupby(conflicts_groups))
        # estimated_duplicates = len(set(x for l in conflicts_groups for x in l))
        # assert estimated_duplicates <= len(self.additional_tokens), \
        #     f'not enough additional tokens {self.separator}. has {len(self.additional_tokens)}, need {estimated_duplicates}'


        conflicts_groups_to_indices = util.swap_keys_and_values_dict(conflicts)

        dependencies_graph = spans_dependencies._dependencies_graph
        next_node_id = max(dependencies_graph.nodes())+1
        marked = set()
        for steps_ids, span_indeces in conflicts_groups_to_indices.items():
            # steps_ids = [x for x in steps_ids if x not in marked]
            # if not steps_ids: continue

            # add pivot
            pivot_id = next_node_id
            next_node_id += 1
            dependencies_graph.add_node(pivot_id, data=SpansData(spans_list=[Span(i,i) for i in span_indeces]))

            # point to pivot
            for step_id in steps_ids:
                assert offset<len(spans_dependencies._tokens), f'offset exceeded: offset {offset}, tokens: {len(spans_dependencies._tokens)}'
                data:SpansData = dependencies_graph.nodes(data=True)[step_id]['data']
                for i in span_indeces: data.remove_index(i)
                data.spans_list.append(Span(offset, offset))
                offset+=1
                dependencies_graph.add_edge(step_id, pivot_id, data=SpanDependencyData(dep_type=self.dependency))
                marked.add(step_id)

        # remove remain conflicts
        for i, steps in index_to_step.items():
            remain_steps = [x for x in steps if x not in marked]
            if len(remain_steps)>1:
                for s_id in remain_steps:
                    data: SpansData = dependencies_graph.nodes(data=True)[s_id]['data']
                    data.remove_index(i)


    @overrides
    def unwind(self, spans_dependencies: SpansDependencies) -> None:
        offset = min(i for i, data in enumerate(spans_dependencies.tokens()) if data.text == self.separator)
        dependencies_graph = spans_dependencies._dependencies_graph
        for u, v, k, data in list(dependencies_graph.edges(data=True, keys=True)):
            data: SpanDependencyData = data['data']
            if data.dep_type != self.dependency: continue
            udata: SpansData = dependencies_graph.nodes(data=True)[u]['data']
            vdata: SpansData = dependencies_graph.nodes(data=True)[v]['data']
            for i in range(offset, offset+len(self.additional_tokens)): udata.remove_index(i)
            udata.spans_list.extend(vdata.spans_list)
            dependencies_graph.remove_edge(u,v,k)

            # remove pivot
            if dependencies_graph.in_degree(v) == 0:
                dependencies_graph.remove_node(v)
