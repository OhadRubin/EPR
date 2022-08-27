"""
1. normalize each step (remove words, properties words, filters-chain)
2. reorder graph
    key = operator, props, [id:dep_type], [tokens]
    update = replace ids
3. key(g1) == key(g2)
"""
import re
import string
from typing import Dict, List, Union, Iterable, Tuple, Set, Any
from dataclasses import dataclass
from abc import ABC
from queue import Queue, deque

import spacy
from spacy.tokens import Token
import networkx as nx

import qdecomp_with_dependency_graphs.utils.data_structures as util
from qdecomp_with_dependency_graphs.scripts.data_processing.break_app_store_generation import get_lexicon_tokens, get_additional, \
    get_token_variations_groups
from qdecomp_with_dependency_graphs.utils.graph import reorder_by_level
from qdecomp_with_dependency_graphs.dependencies_graph.data_types import DependencyType, SpanDependencyData
from qdecomp_with_dependency_graphs.scripts.qdmr_to_logical_form.operator_identifier import QDMROperator, get_identifier, ArgumentType
from qdecomp_with_dependency_graphs.evaluation.normal_form.normalization_rules import ReferenceToken


@dataclass
class QDMRStepTokens:
    operator: QDMROperator
    properties: List[str]
    arguments_tokens: Dict[ArgumentType, List[str]]


class QDMRStepTokensDependencies:
    def __init__(self, dependencies_graph: nx.MultiDiGraph):
        """
        :param dependencies_graph: MultiDiGraph
        where node.data['data'] is of type QDMRStepTokens and edge.data['data'] is of type SpanDependencyData
        """
        self._dependencies_graph: nx.MultiDiGraph = dependencies_graph

    def to_string(self, reorder: bool = False):
        # return LogicalFromStructuralMatcher.graph_key(self)
        steps = []
        for step_id, data in self._dependencies_graph.nodes(data=True):
            data: QDMRStepTokens = data['data']
            args_str = []
            refs_types: Dict[DependencyType, Any] = util.list_to_multivalue_dict(
                self._dependencies_graph.out_edges(step_id, data=True),
                key=lambda x: x[2]['data'].dep_type)
            for arg, tokens in data.arguments_tokens.items():
                refs = [f'#{x[1]}' for x in refs_types.get(arg, [])]
                arg_val = ','.join(x for x in refs+[" ".join(tokens or [])] if x)
                args_str.append(f'{arg.get_arg_name()}={arg_val}')
            # make sure referenced steps with no tokens appears
            for arg, refs_list in refs_types.items():
                if arg not in data.arguments_tokens:
                    refs = ','.join([f'#{x[1]}' for x in refs_list])
                    args_str.append(f'{arg.get_arg_name()}={refs}')

            prop_part = f'[{",".join(data.properties or [])}]' if data.properties else ''
            steps.append((step_id, f'{data.operator}{prop_part}({"; ".join(sorted(args_str))})'))

        if reorder:
            # ordered_steps = sorted(steps, key=lambda x: sorted(re.findall(r'#(\d+)', x), reverse=True))
            # old_to_new_ids = {re.findall(r'^(\d+)\.', x)[0]: str(i+1) for i, x in enumerate(ordered_steps)}
            # ordered_steps = [re.sub(r'^(\d+)\.', lambda x: f'{old_to_new_ids[x.group(1)]}.', step) for step in ordered_steps]
            # steps = [re.sub(r'#(\d+)', lambda x: f'#{old_to_new_ids[x.group(1)]}', step) for step in ordered_steps]

            levels = []
            previous_levels = set([])
            steps_with_refs = {step_id: (step, re.findall(r'#(\d+)', step)) for step_id, step in steps}
            while steps_with_refs:
                level_steps = []
                for step_id, (step, refs) in list(steps_with_refs.items()):
                    if all(((x in previous_levels) or x == str(step_id)) for x in refs):
                        del steps_with_refs[step_id]
                        level_steps.append(str(step_id))
                if not level_steps:
                    # a cycle
                    levels.append([str(step_id) for step_id in steps_with_refs.keys()])
                    break
                levels.append(level_steps)
                previous_levels.update(level_steps)

            ordered_ids = [step_id for l in levels for step_id in l]
            old_to_new = {str(step_id): str(i+1) for i, step_id in enumerate(ordered_ids)}
            steps = [(old_to_new[str(step_id)], re.sub(r'#(\d+)', lambda x: f'#{old_to_new[x.group(1)]}', step))
                     for step_id, step in steps]
        steps_str = [f'{i}. {s}' for i, s in sorted(steps, key=lambda x: int(x[0]))]
        return '\n'.join(steps_str)


class LogicalFromStructuralMatcher:
    """
    Graph:
        * nodes are {'data': QDMRStepTokens}
        * edges are {'data': SpanDependencyData}
    """
    def __init__(self):
        self.rules: Iterable[NormalizationRule] = [
            ToLowerNormalizationRule(),

            # clean properties indicators
            CleanPropertiesIndicatorNormalizationRule(is_soft=True),

            # clean unnecessary tokens
            CleanNormalizationRule(),
            CleanCharactersNormalizationRule(),  # assumption: after CleanPropertiesIndicatorNormalizationRule() due to #REF (removes the '#')

            # replace token with representative
            ReplaceWithRepresentativeNormalizationRule(),

        ]
        self.merge_patterns: Iterable[MergePattern] = [
            MergePattern(DependencyType.FILTER_SUB, QDMROperator.FILTER, merge_operator=QDMROperator.FILTER,
                         dep_to_merge_map={}),
            MergePattern(DependencyType.FILTER_SUB, QDMROperator.SELECT, merge_operator=QDMROperator.FILTER,
                         dep_to_merge_map= MergePattern.get_constant_dict(DependencyType.FILTER_SUB)),
            MergePattern(DependencyType.PROJECT_SUB, QDMROperator.SELECT, merge_operator=QDMROperator.PROJECT,
                         dep_to_merge_map= MergePattern.get_constant_dict(DependencyType.PROJECT_SUB)),
        ]

    def is_match(self, question_id: str, question_text: str, graph1: QDMRStepTokensDependencies, graph2: QDMRStepTokensDependencies) -> bool:
        try:
            self.normalize_logical_graph(question_id, question_text, graph1)
            self.normalize_logical_graph(question_id, question_text, graph2)
            return self.graph_key(graph1) == self.graph_key(graph2)
        except Exception as ex:
            return False

    def normalize_logical_graph(self, question_id: str, question_text: str, dep_graph: QDMRStepTokensDependencies):
        graph = dep_graph._dependencies_graph
        # normalize steps
        for _, data in graph.nodes(data=True):
            for rule in self.rules:
                rule.normalize(step=self.node_data(data), question_id=question_id, question_text=question_text)

        # merge rules
        self.merge(graph)

        # remove empty args

        # reorder
        old_to_new_node_ids_map = {}
        reorder_by_level(
            graph,
            key=lambda n_id, _: self.node_key(n_id, dep_graph, old_to_new_node_ids_map),
            update_node=lambda _, old_to_new_map: old_to_new_node_ids_map.update(old_to_new_map)
        )

    def merge(self, graph: nx.MultiDiGraph):
        nodes_operators = {n_id: self.node_data(data).operator for n_id, data in graph.nodes(data=True)}
        next_node_id = max(nodes_operators.keys()) + 1
        updated_nodes = Queue()
        updated_nodes.queue = deque(nodes_operators.keys())
        while not updated_nodes.empty():
            #  pop a source node (step)
            cur_node = updated_nodes.get()

            # node is not valid anymore
            if cur_node not in nodes_operators:
                continue

            for pattern in self.merge_patterns:
                # source node operator does not fit the pattern
                if pattern.dependency_type.get_operator() != nodes_operators.get(cur_node, None):
                    continue
                for u, v, data in graph.out_edges(cur_node, data=True):
                    # dependency does not fit the pattern
                    if not (self.edge_data(data).dep_type == pattern.dependency_type and
                            nodes_operators.get(v, None) == pattern.dest_step_operator):
                        continue

                    u_step = self.node_data(graph.nodes[u])
                    v_step = self.node_data(graph.nodes[v])

                    # merge operator and properties
                    merged_operator = pattern.merge_operator
                    if merged_operator == u_step.operator:
                        merged_properties = u_step.properties
                    elif merged_operator == v_step.operator:
                        merged_properties = v_step.properties
                    else:
                        merged_properties = None

                    # merge arguments by pattern mapping
                    merged_args = {}
                    for step in [u_step, v_step]:
                        for x, y in step.arguments_tokens.items():
                            merged_arg_name = pattern.get_merged_dependency(DependencyType(x))
                            merged_args[merged_arg_name] = merged_args.get(merged_arg_name, []) + y

                    # merged step
                    merged_step = QDMRStepTokens(operator=merged_operator,
                                                 properties=merged_properties,
                                                 arguments_tokens=merged_args)

                    graph.add_node(next_node_id, data=merged_step)
                    nodes_operators[next_node_id] = merged_step.operator
                    updated_nodes.put(next_node_id)

                    # update u incoming dependencies to points on the new node (which is logically identical to u)
                    for x, y, z in graph.in_edges(u, data=True):
                        graph.add_edge(x, next_node_id, **z)
                        updated_nodes.put(x)

                    # map the outgoing dependencies (references) of u/v to the new dode dependencies
                    for x, y, z in list(graph.out_edges(u, data=True))+list(graph.out_edges(v, data=True)):
                        if (x, y) == (u, v):
                            continue
                        z_dep = self.edge_data(z)
                        new_dep = SpanDependencyData(dep_type=pattern.get_merged_dependency(z_dep.dep_type))
                        graph.add_edge(next_node_id, y, data=new_dep)

                    # we have no references anymore - change to select operator
                    if not any(graph.successors(next_node_id)):
                        merged_step.operator = QDMROperator.SELECT
                        merged_step.arguments_tokens = {ArgumentType.SELECT_SUB: [x for s in merged_step.arguments_tokens.values() for x in s]}
                        nodes_operators[next_node_id] = QDMROperator.SELECT

                    # remove u (it is represented by the new node)
                    graph.remove_node(u)
                    del nodes_operators[u]

                    # remove v if it has no incoming dependencies (but those of u)
                    if all(x == u for x in graph.predecessors(v)):
                        graph.remove_node(v)
                        del nodes_operators[v]

                    next_node_id += 1
                    break

    @staticmethod
    def node_data(attr) -> QDMRStepTokens:
        return attr['data']

    @staticmethod
    def edge_data(attr) -> SpanDependencyData:
        return attr['data']

    @staticmethod
    def node_key(n_id: int, dep_graph: QDMRStepTokensDependencies, node_ids_map=None):
        graph = dep_graph._dependencies_graph
        nid = lambda x: x if node_ids_map is None else node_ids_map.get(x, x)
        step = LogicalFromStructuralMatcher.node_data(graph.nodes[n_id])
        operator_part = f"{step.operator}"
        props_part = ','.join(sorted(step.properties or []))
        args_part = ','.join(sorted(set(x for arg in step.arguments_tokens.values() for x in arg)))
        dep_part = ",".join(sorted(f"({nid(v)}, {LogicalFromStructuralMatcher.edge_data(d).dep_type})"
                                   for u, v, d in graph.out_edges(n_id, data=True)))
        return f"{operator_part.upper()}[{props_part.lower()}]{dep_part.lower()}{{{args_part.lower()}}}"

    @staticmethod
    def graph_key(dep_graph: QDMRStepTokensDependencies) -> str:
        graph = dep_graph._dependencies_graph
        nodes_ids = sorted(list(graph))
        nodes_keys = [LogicalFromStructuralMatcher.node_key(n_id, dep_graph) for n_id in nodes_ids]
        return '; '.join(nodes_keys)


@dataclass
class MergePattern:
    dependency_type: DependencyType
    dest_step_operator: QDMROperator
    merge_operator: QDMROperator
    dep_to_merge_map: Dict[DependencyType, DependencyType]

    @staticmethod
    def get_constant_dict(val):
        # todo: move to util? wit _Dummy()?
        class _ConstantDict(dict):
            def __getitem__(self, _):
                return val
        return _ConstantDict()

    def get_merged_dependency(self, dependency: DependencyType):
        if dependency.get_operator() == self.merge_operator:
            return dependency
        return self.dep_to_merge_map[dependency]


###############################
#   Normalization rules
###############################

class NormalizationRule(ABC):
    def normalize(self, step: QDMRStepTokens, question_id: str=None, question_text: str=None):
        raise NotImplementedError()


class ToLowerNormalizationRule(NormalizationRule):
    def __init__(self):
        super().__init__()

    def normalize(self, step: QDMRStepTokens, question_id: str=None, question_text: str=None):
        if step.arguments_tokens:
            step.arguments_tokens = {
                k: [x.lower() for x in v]
                for k, v in step.arguments_tokens.items()
            }


# class CleanDETNormalizationRule(NormalizationRule):
#     def normalize(self, step: QDMRStepTokens, question_id: str=None, question_text: str=None):
#         if step.arguments_tokens:
#             step.arguments_tokens = {
#                 k: [x for x in v if x.pos_ not in ["DET"]]
#                 for k,v in step.arguments_tokens.items()
#             }

class CleanCharactersNormalizationRule(NormalizationRule):
    def __init__(self):
        super().__init__()

    def normalize(self, step: QDMRStepTokens, question_id: str=None, question_text: str=None):
        if step.arguments_tokens:
            step.arguments_tokens = {
                k: [x for s in v for x in re.split('\W+', s) if x]
                for k, v in step.arguments_tokens.items()
            }


class CleanNormalizationRule(NormalizationRule):
    def __init__(self):
        super().__init__()
        self._tokens = get_additional() + ['be', 'as']

    def normalize(self, step: QDMRStepTokens, question_id: str=None, question_text: str=None):
        if step.arguments_tokens:
            step.arguments_tokens = {
                k: [x for x in v if x not in self._tokens]
                for k, v in step.arguments_tokens.items()
            }


class CleanPropertiesIndicatorNormalizationRule(NormalizationRule):
    def __init__(self, is_soft: bool):
        super().__init__()
        self.is_soft = is_soft

    def normalize(self, step: QDMRStepTokens, question_id: str=None, question_text: str=None):
        if not step.arguments_tokens:
            return
        identifier = get_identifier(step.operator)
        indicators = ((step.properties and identifier.properties_indicators(step.properties)) or []) + identifier.operator_indicators()
        indicators = [x.lower() for x in indicators]
        if indicators:
            if self.is_soft:
                # soft cleaning
                indicators = [x for i in indicators for x in i.split(' ') if x]
                step.arguments_tokens = {
                                k: [x for x in v if x not in indicators]
                                for k,v in step.arguments_tokens.items()
                }
            else:
                # restrict cleaning - check for sequential inclusion
                indicators = sorted([x.split(' ') for x in indicators],
                                    key=lambda x: len(x), reverse=True)
                new_args = {}
                for k, v in step.arguments_tokens.items():
                    i = 0
                    new_v = []
                    while i < len(v):
                        skipped = False
                        for ind in indicators:
                            if len(ind)+i > len(v): continue
                            if all(x==y for x,y in zip(v[i:], ind)):
                                i += len(ind)
                                skipped = True
                                break
                        if not skipped:
                            new_v.append(v[i])
                            i+=1
                    new_args[k] = new_v
                step.arguments_tokens = new_args


class ReplaceWithRepresentativeNormalizationRule(NormalizationRule):
    def __init__(self):
        super().__init__()
        self.cache = {}
        self._parser = spacy.load('en_core_web_sm', disable=['ner'])

    def parse(self, question_text: str, question_id: str=None):
        return self._parser(re.sub(r'\s+', ' ', question_text))

    def get_representative_map(self, question_id: str=None, question_text: str=None) -> Dict[str, str]:
        if question_id in self.cache:
            return self.cache[question_id]
        question_tokens = self.parse(question_text=question_text, question_id=question_id)
        question_tokens_variations = [get_lexicon_tokens(x) for x in question_tokens]
        variations_groups = list(get_token_variations_groups(is_operational=True)) + question_tokens_variations
        tok_to_variations = util.merge_equivalent_classes(variations_groups)
        # todo: support multi-tokens representative (like at least)
        representative_map = {k: sorted([x.lower() for x in v if ' ' not in x])[0] for k,v in tok_to_variations.items()
                              if ' ' not in k}
        self.cache[question_id] = representative_map
        return representative_map

    def normalize(self, step: QDMRStepTokens, question_id: str=None, question_text: str=None):
        if step.arguments_tokens:
            rep_map = self.get_representative_map(question_id=question_id, question_text=question_text)
            step.arguments_tokens = {
                k: [rep_map.get(x, x) for x in v]
                for k,v in step.arguments_tokens.items()
            }