import traceback
from typing import List

import networkx as nx
import pandas as pd
import spacy
from spacy.tokens import Doc

import qdecomp_with_dependency_graphs.utils.data_structures as util

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import TokensDependencies, SpansDependencies, SpansData, DependencyType, Span
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_to_qdmr_extractors.converters import BaseCollapser, JoinCollapser
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_to_qdmr_extractors.converters import _get_refs_if_match
from qdecomp_with_dependency_graphs.evaluation.normal_form.normalized_graph_matcher import NormalizedGraphMatchScorer
from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition
from qdecomp_with_dependency_graphs.scripts.eval.evaluate_predictions import format_qdmr

from qdecomp_with_dependency_graphs.scripts.qdmr_to_program import QDMROperation
import qdecomp_with_dependency_graphs.scripts.qdmr_to_logical_form.operator_identifier as op


def get_normalized(decomposition: str):
    try:
        decomposition = format_qdmr(decomposition)
        norm = normalizer.normalized_decomposition(decomposition)
        return norm.to_string()
    except Exception as ex:
        print(str(ex))
        return "ERROR"


def add_normalized(df: pd.DataFrame) -> pd.DataFrame:
    df['gold_normalized'] = [get_normalized(x) for x in df['gold']]
    df['prediction_normalized'] = [get_normalized(x) for x in df['prediction']]
    return df


parser = spacy.load('en_core_web_sm', disable=['ner'])
def get_semi_normalized(tokens_dependencies: TokensDependencies, collapser: BaseCollapser = None):
    try:
        # spans dependencies graph
        spans_dependencies: SpansDependencies = tokens_dependencies.to_spans_dependencies()

        # unwind collapsed steps
        collapser and collapser.unwind(spans_dependencies)

        # convert to qdmr
        qdmr_graph: nx.DiGraph = _convert_dependencies_to_qdmr(spans_dependencies)

        graph = normalizer.normalize_graph(qdmr_graph, skip_init=True)

        return Decomposition.from_graph(graph).to_string()
    except Exception as ex:
        print(str(ex))
        traceback.print_exc()
        return "ERROR"


def _convert_dependencies_to_qdmr(spans_dependencies: SpansDependencies) -> nx.DiGraph:
    question_tokens = [x.text.lower() for x in spans_dependencies.tokens()]
    doc = parser(' '.join(question_tokens))

    dependencies_graph = spans_dependencies._dependencies_graph
    decomposition_graph = nx.DiGraph()
    next_node_id = max(list(dependencies_graph.nodes())+[0]) + 1

    for n, data in list(dependencies_graph.nodes(data=True)):
        spans: SpansData = data.get('data', None)
        spans_list = spans.spans_list if spans else []
        spans_str = [' '.join(str(x) for x in question_tokens[s.start:s.end + 1]) for s in spans_list]
        outgoing_edges = dependencies_graph.out_edges(n, data=True)

        label, args = _get_label_and_refs(outgoing_edges, doc, spans_list)
        # if label is None:
        #     # todo: might be select or none
        #     raise ValueError(f'could not build qdmr for step {spans_str}, {outgoing_edges}')
        refs = []
        if args:
            for x in args:
                if isinstance(x,list):
                    new_doc = [t for s in x for t in doc[s.start: s.end+1]]
                    new_id = next_node_id
                    next_node_id += 1
                    decomposition_graph.add_node(new_id, doc=new_doc)
                    refs.append(new_id)
                else:
                    refs.append(x)
        if refs:
            decomposition_graph.add_edges_from((n, x) for x in refs)
        if n not in decomposition_graph:
            decomposition_graph.add_node(n)
        if label:
            refs = [f'@@{x}@@' for x in refs]
            params = f"""({",".join(refs)})""" if refs else ""
            decomposition_graph.nodes[n]['label'] = label+params
        else:
            decomposition_graph.nodes[n]['doc'] = [t for s in spans_list for t in doc[s.start: s.end+1]]
    return decomposition_graph


def _get_label_and_refs(outgoing_edges, doc:Doc, ranges_str:List[Span]):
    def extract_aggregate(text:str):
        op_identifier = op.IdentifyOperator()
        op_identifier.step = text
        return op_identifier.extract_properties(text)

    tokens_str = ' '.join(str(doc[x.start: x.end+1]) for x in ranges_str)
    refs_types = util.list_to_multivalue_dict(outgoing_edges, key=lambda x: x[2]['data'].dep_type)

    def meta_part(meta: List[str]):
        meta = meta and [x for x in meta if x]
        return f'[{",".join(meta)}]' if meta else ''

    # select
    if not refs_types:
        return None, None

    # aggregate
    res = _get_refs_if_match(refs_types, [
        DependencyType.AGGREGATE_ARG])
    if res:
        refs, _ = res
        agg = tokens_str if tokens_str else 'number of '
        agg = op.IdentifyOperatorAggregate().extract_properties(agg)
        return f'{QDMROperation.AGGREGATE.name}{meta_part(agg)}', refs

    # arithmetic
    res = _get_refs_if_match(refs_types, [
        DependencyType.ARITHMETIC_LEFT, DependencyType.ARITHMETIC_RIGHT])
    if res:
        refs, _ = res
        meta = [x for x in ['sum', 'difference', 'multiplication', 'division'] if x in tokens_str]
        # todo: normalize meta
        return f'{QDMROperation.ARITHMETIC.name}{meta_part(meta)}', refs

    # boolean
    operation = op.IdentifyOperatorBoolean()
    res = _get_refs_if_match(refs_types, [
        DependencyType.BOOLEAN_SUB, DependencyType.BOOLEAN_CONDITION])
    if res:
        refs, _ = res
        logical_op = None
        if "both" in tokens_str and "and" in tokens_str:
            logical_op = "logical_and"
        elif "either" in tokens_str and "or" in tokens_str:
            logical_op = "logical_or"
        if logical_op is not None:
            bool_expr = "false" if "false" in tokens_str else "true"
            return f'{QDMROperation.BOOLEAN.name}{meta_part([logical_op, bool_expr])}', refs
        # todo: handle "if_exists"
        # todo: handle empty range_str (=>"equals")
        return f'{QDMROperation.BOOLEAN.name}', refs+[ranges_str]

    res = _get_refs_if_match(refs_types, [
        DependencyType.BOOLEAN_SUB])
    if res:
        refs, refs_tokens = res
        return f'{QDMROperation.BOOLEAN.name}', refs+[ranges_str]

    # comparative
    res = _get_refs_if_match(refs_types, [
        DependencyType.COMPARATIVE_SUB, DependencyType.COMPARATIVE_ATTRIBUTE])
    if res:
        refs, _ = res
        return f'{QDMROperation.COMPARATIVE.name}', refs+[ranges_str]

    res = _get_refs_if_match(refs_types, [
        DependencyType.COMPARATIVE_SUB, DependencyType.COMPARATIVE_ATTRIBUTE, DependencyType.COMPARATIVE_CONDITION])
    if res:
        refs, _ = res
        return f'{QDMROperation.COMPARATIVE.name}', refs + [ranges_str]

    # comparison
    res = _get_refs_if_match(refs_types, [
        DependencyType.COMPARISON_ARG])
    if res:
        refs, _ = res
        meta = op.IdentifyOperatorComparison().extract_properties(tokens_str)
        return f'{QDMROperation.COMPARISON.name}{meta_part(meta)}', refs + [ranges_str]

    # discard
    res = _get_refs_if_match(refs_types, [
        DependencyType.DISCARD_SUB, DependencyType.DISCARD_EXCLUDE])
    if res:
        refs, _ = res
        return f'{QDMROperation.DISCARD.name}', refs

    # filter
    res = _get_refs_if_match(refs_types, [
        DependencyType.FILTER_SUB])
    if res:
        refs, _ = res
        return f'{QDMROperation.FILTER.name}', refs + [ranges_str]

    res = _get_refs_if_match(refs_types, [
        DependencyType.FILTER_SUB, DependencyType.FILTER_CONDITION])
    if res:
        refs, _ = res
        return f'{QDMROperation.FILTER.name}', refs + [ranges_str]  # todo: filter_sub condition+filter_arg

    # group
    res = _get_refs_if_match(refs_types, [
        DependencyType.GROUP_VALUE, DependencyType.GROUP_KEY])
    if res:
        refs, _ = res
        agg = op.IdentifyOperatorGroup().extract_properties(tokens_str) or ["count"]
        return f'{QDMROperation.GROUP.name}{meta_part(agg)}', refs

    # intersection
    res = _get_refs_if_match(refs_types, [
        DependencyType.INTERSECTION_PROJECTION, DependencyType.INTERSECTION_INTERSECTION, DependencyType.INTERSECTION_INTERSECTION])
    if res:
        refs, _ = res
        return f'{QDMROperation.INTERSECTION.name}', refs

    # project
    res = _get_refs_if_match(refs_types, [
        DependencyType.PROJECT_SUB])
    if res:
        return None, None
        # refs, _ = res
        # return f'{QDMROperation.PROJECT.name}', [ranges_str]+refs

    res = _get_refs_if_match(refs_types, [
        DependencyType.PROJECT_PROJECTION, DependencyType.PROJECT_SUB])
    if res:
        refs, _ = res
        return f'{QDMROperation.PROJECT.name}', refs

    # sort
    res = _get_refs_if_match(refs_types, [
        DependencyType.SORT_SUB, DependencyType.SORT_ORDER])
    if res:
        refs, _ = res
        return f'{QDMROperation.SORT.name}', refs

    # superlative
    res = _get_refs_if_match(refs_types, [
        DependencyType.SUPERLATIVE_SUB, DependencyType.SUPERLATIVE_ATTRIBUTE])
    if res:
        refs, _ = res
        agg = op.IdentifyOperatorSuperlative().extract_properties(tokens_str) or ['max']
        return f'{QDMROperation.SUPERLATIVE.name}{meta_part(agg)}', refs

    # union
    res = _get_refs_if_match(refs_types, [
        DependencyType.UNION_SUB], strict_count=False)
    if res:
        refs, refs_tokens = res
        return f'{QDMROperation.UNION.name}', refs

    return None, None


from qdecomp_with_dependency_graphs.dependencies_graph.debug_dependencies_creation import to_dependencies_and_back_from_dataset
import os
def eval_by_normalize(*args, **kwargs):
    collapser = JoinCollapser()
    normalized_pred = []
    normalized_gold = []
    def func(gold_decomposition: str, tokens_dependencies: TokensDependencies = None, **kwargs):
        normalized_pred.append(get_semi_normalized(tokens_dependencies, collapser))
        normalized_gold.append(get_normalized(gold_decomposition))

    df = to_dependencies_and_back_from_dataset(*args, **kwargs, func=func, is_eval=False)
    df['gold_norm'] = normalized_gold
    df['decomposition_norm'] = normalized_pred

    df['exact_match'] = df['gold_decomposition'].str.lower() == df[
        'decomposition'].str.lower()
    df['normalized_exact_match'] = df['gold_norm'].str.lower() == df[
        'decomposition_norm'].str.lower()

    if "dest_dir" in kwargs:
        df.to_csv(os.path.join(kwargs['dest_dir'], 'dependencies_graph_redecomp.csv'), index=False)

    # print eval:
    print('FULL')
    print(df.mean().round(3))

    print('NO-ERROR')
    redecomp_df_no_error = df[df['decomposition'] != "ERROR"]
    print(redecomp_df_no_error.mean().round(3))


if __name__ == '__main__':
    import qdecomp_with_dependency_graphs.evaluation.normal_form.normalization_rules as norm_rules
    import qdecomp_with_dependency_graphs.evaluation.normal_form.operations_normalization_rules as op_norm_rules
    from qdecomp_with_dependency_graphs.evaluation.normal_form.operations_normalization_rules__new import NewWrapperDecomposeRule

    norm_rules.load_cache('_debug/normal_form/cache')
    # normalizer = NormalizedGraphMatchScorer()
    extract_params = True
    normalizer = NormalizedGraphMatchScorer(
        rules=[
            norm_rules.RemoveDETDecomposeRule(),
            op_norm_rules.FilterAdjectiveLikeNounDecomposeRule(is_extract_params=extract_params),
            norm_rules.NounsExtractionDecomposeRule(),
            norm_rules.ADPDecomposeRule(),
            norm_rules.CompoundNounExtractionDecomposeRule(),
            norm_rules.AdjectiveDecomposeRule(),
            norm_rules.AdjectiveLikeNounDecomposeRule(),
            op_norm_rules.FilterAdjectiveDecomposeRule(is_extract_params=extract_params),
            op_norm_rules.FilterADPDecomposeRule(is_extract_params=extract_params),
            op_norm_rules.FilterCompoundNounDecomposeRule(is_extract_params=extract_params),
            op_norm_rules.FilterConditionDecomposeRule(is_extract_params=extract_params),
            op_norm_rules.WrapperFixesAggregateDecomposeRule(is_extract_params=extract_params),
            op_norm_rules.WrapperFixesBooleanDecomposeRule(is_extract_params=extract_params),
            NewWrapperDecomposeRule(is_extract_params=extract_params)
            # op_norm_rules.WrapperDecomposeRule(is_extract_params=extract_params),
        ]
    )

    # df = pd.read_csv('_debug/dep-qdmr-parser/dependencies_graph_redecomp_eval_full.csv')
    # add_normalized(df).to_csv('_debug/dep-qdmr-parser/dependencies_graph_redecomp_eval_full__with_normalized.csv', index=False)

    eval_by_normalize('datasets/Break/QDMR/dev.csv',
                      # random_n=5,
                      dest_dir='_debug/normal_form/_debug',
                      spans_file='datasets/Break/QDMR/dev_spans.json')

    norm_rules.save_cache('_debug/normal_form/cache')


    # from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition
    # dec = Decomposition.from_str(
    #     """available @@SEP@@ denver @@SEP@@ flight @@SEP@@ philadelphia @@SEP@@ from @@2@@ @@SEP@@ to @@4@@ @@SEP@@ FILTER(@@3@@,@@5@@) @@SEP@@ FILTER(@@7@@,@@6@@) @@SEP@@ FILTER(@@8@@,@@1@@)"""
    # )
    # with open('_debug/normal_form/tst.svg', 'wt') as w:
    #     cnt = dec.draw_decomposition()
    #     w.write(cnt)

