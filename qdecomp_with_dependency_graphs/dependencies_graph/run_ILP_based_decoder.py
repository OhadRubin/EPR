import argparse
import json
import logging
import os
from typing import List, Tuple, Dict, Any
from dataclasses import asdict
import multiprocessing

import numpy as np
import pandas as pd
from allennlp.data import Vocabulary
from tqdm import tqdm

from qdecomp_with_dependency_graphs.dependencies_graph.check_frequency__structural_constraints import run_check_freq_on_preds_file
from qdecomp_with_dependency_graphs.dependencies_graph.decoders.ILP_based_decoder import ILPDecoder
from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.evaluate_dep_graph import evaluate_learned_dep_graph
from qdecomp_with_dependency_graphs.dependencies_graph.examine_predictions import json_file_to_dependencies_graphs, gold_to_dependencies
from qdecomp_with_dependency_graphs.utils.graph import render_dependencies_graph_svg
from qdecomp_with_dependency_graphs.utils.helpers import Timer
from qdecomp_with_dependency_graphs.utils.html import HTMLTemplate

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def _is_same_solutions(arcs1: List[Tuple[int,int]], arc_tags1: List[str], arcs2: List[Tuple[int,int]], arc_tags2: List[str]):
    deps1 = [(i,j,d) for (i,j), d in zip(arcs1, arc_tags1)]
    deps2 = [(i, j, d) for (i, j), d in zip(arcs2, arc_tags2)]
    if len(deps1) != len(deps2):
        return False
    return set(deps1) == set(deps2)


def _decode(preds_file: str, timeout: int = None, question_ids: List[str] = None, subdir: str = 'ILP_decode',
            enumerator=None, precision: int = 3,
            ignore_concatenated_tags: bool = True,
            limit_arcs_number: bool = False, drop_arcs_only: bool = False, keep_arcs: bool = False,
            sanity_check: bool = False, skip_violation_check: bool = False):
    if enumerator is None:
        enumerator = get_enumerate_preds(preds_file=preds_file, question_ids=question_ids)
    model_dir = os.path.dirname(os.path.dirname(preds_file))
    vocab = Vocabulary.from_files(directory=os.path.join(model_dir, 'model.tar.gz'))
    decoder = ILPDecoder(tags_by_index=vocab.get_index_to_token_vocabulary('labels'), max_time_in_seconds=timeout,
                         ignore_concatenated_tags=ignore_concatenated_tags,
                         limit_arcs_number=limit_arcs_number, drop_arcs_only=drop_arcs_only, keep_arcs=keep_arcs,
                         sanity_check=sanity_check, skip_violation_check=skip_violation_check)
    constraints_names = list(sorted([type(x).__name__ for x in decoder.constraints]))
    _logger.info(f'constraints: {constraints_names}')

    base_output_dir = os.path.join(os.path.dirname(preds_file), subdir)
    os.makedirs(base_output_dir)
    dest_preds_file = os.path.join(base_output_dir, os.path.basename(preds_file))
    timer = Timer()

    # fix preds
    def params_enumerator():
        for pred in enumerator():
            yield pred, decoder, precision

    with multiprocessing.Pool() as p:
        fixed_preds = p.starmap(decode_pred, params_enumerator())

    # write new preds
    with open(dest_preds_file, 'wt') as wfp:
        for l in fixed_preds:
            wfp.write(json.dumps(l)+'\n')

    # summarize
    summarize_statistics(dest_preds_file,
                         extra={
                             'running_time': timer.get_time_diff_str(),
                             'constraints': constraints_names
                         })

    # evaluate
    evaluate_learned_dep_graph(
        dataset=pd.read_csv('datasets/Break/QDMR/dev.csv'),
        model_dir=model_dir,
        skip_exists=True,
        all_dev=False,
        preds_file=dest_preds_file,
    )

    # verify constraints
    run_check_freq_on_preds_file(preds_file=dest_preds_file)


def decode_pred(pred_dict: dict, decoder: ILPDecoder, precision: int):
    qid = pred_dict['metadata']['question_id']
    tokens = pred_dict['metadata']['tokens']
    n = len(tokens)
    arc_prob = np.nan_to_num(pred_dict.pop('arc_probs'))[:n, :n] if (pred_dict.get('arc_probs') is not None) else None
    arc_tag_prob = np.nan_to_num(pred_dict.pop('arc_tag_probs'))[:n, :n]
    try:
        res = decoder.decode(
            question_id=qid,
            tokens=tokens,
            arc_prob=arc_prob,
            arc_tag_prob=arc_tag_prob,
            curr_arcs=pred_dict['arcs'],
            curr_arc_tags=pred_dict['arc_tags'],
            precision=precision
        )
        if res is not None:
            pred_dict['is_same'] = _is_same_solutions(pred_dict['arcs'], pred_dict['arc_tags'], res.arcs, res.arc_tags)
            pred_dict['is_ILP'] = True
            pred_dict['arcs'] = res.arcs
            pred_dict['arc_tags'] = res.arc_tags
            pred_dict.update(asdict(res))
        else:
            pred_dict['is_ILP'] = False
    except Exception as ex:
        _logger.exception(f'ERROR in {qid}: {str(ex)}')
        pred_dict['ILP_error'] = str(ex)
        pred_dict['is_ILP_error'] = True
    return pred_dict


def summarize_statistics(ILP_preds_file: str, extra: Dict[str, Any] = None):
    df = pd.read_json(ILP_preds_file, lines=True)
    base_name = os.path.splitext(ILP_preds_file)[0]
    summarize_statistics_(df, base_name, extra=extra)


def summarize_statistics_(ILP_preds: pd.DataFrame, base_name: str = None, extra: Dict[str, Any] = None):
    means = ILP_preds.mean().to_dict()
    sum = ILP_preds.sum().to_dict()
    counts = ILP_preds.count().to_dict()
    total = len(ILP_preds)
    summary = {**(extra or {}), 'total': total}
    summary.update({
        k: f'{means.get(k, 0)*100.0:.2f}% ({sum.get(k)}/{counts.get(k)})' for k in means.keys()
    })
    if base_name:
        with open(f'{base_name}__summary.json', 'wt') as fp:
            json.dump(summary, fp, indent=2)
    return summary


def unified_preds_and_eval_files(ILP_preds_file: str, original_preds_file: str, plot_all: bool = False):
    ILP_preds = pd.read_json(ILP_preds_file, lines=True)
    ILP_preds['question_id'] = ILP_preds['metadata'].apply(lambda x: x['question_id'])
    ILP_preds['arcs_and_tags'] = ILP_preds.apply(lambda row: list(zip(row['arcs'], row['arc_tags'])), axis=1)
    ILP_eval = pd.read_csv(os.path.splitext(ILP_preds_file)[0]+'_eval.csv')

    preds = pd.read_json(original_preds_file, lines=True)
    preds['question_id'] = preds['metadata'].apply(lambda x: x['question_id'])
    preds['arcs_and_tags'] = preds.apply(lambda row: list(zip(row['arcs'], row['arc_tags'])), axis=1)
    eval = pd.read_csv(os.path.splitext(original_preds_file)[0]+'_eval.csv')

    df = ILP_eval
    df = df.merge(ILP_preds[['question_id', 'arcs_and_tags', 'is_ILP', 'is_same', 'ILP_error', 'satisfy_all', 'objective']],
                  on='question_id', how='left', suffixes=('', ''))
    df = df.merge(eval[['question_id', 'pred_logical_form_tokens', 'pred_logical_form_tokens_original',	'logical_form_tokens_exact_match']],
                  on='question_id', how='left', suffixes=('', '__NO_ILP'))
    df = df.merge(preds[['question_id', 'arcs_and_tags']],
                  on='question_id', how='left', suffixes=('', '__NO_ILP'))

    df = df[(df['is_ILP'] != False) & ((df['is_same'] != True) | plot_all)].copy()
    base_name = os.path.splitext(ILP_preds_file)[0]+'__unified'
    df.to_csv(f'{base_name}.csv', index=False)

    summary = summarize_statistics_(df)
    summary['no_ILP_logical_form_EM=True'] = summarize_statistics_(df[df['logical_form_tokens_exact_match__NO_ILP'] == True])
    summary['no_ILP_logical_form_EM=False'] = summarize_statistics_(df[df['logical_form_tokens_exact_match__NO_ILP'] == False])

    with open(f'{base_name}__summary.json', 'wt') as fp:
        json.dump(summary, fp, indent=2)

    plots_dir = base_name + '_plots'
    df_qids = df['question_id'].unique()
    os.makedirs(plots_dir)
    template = HTMLTemplate()
    for question_id in df_qids:
        row = preds[preds['question_id']==question_id].iloc[0]
        ILP_row = ILP_preds[ILP_preds['question_id']==question_id].iloc[0]
        tokens = row['metadata']['tokens']

        svg = render_dependencies_graph_svg(
            words=zip(tokens, [str(i) for i in range(len(tokens))]),
            arcs=[(i, j, t) for (i, j), t in zip(row['arcs'], row['arc_tags'])]
        )
        ilp_svg = render_dependencies_graph_svg(
            words= zip(tokens, [str(i) for i in range(len(tokens))]),
            arcs = [(i,j,t) for (i,j), t in zip(ILP_row['arcs'], ILP_row['arc_tags'])]
        )

        content = template.get_body(question_id=question_id, body=
        f"pred:<figure>{svg}</figure>ILP:<figure>{ilp_svg}</figure>")

        with open(os.path.join(plots_dir, f'{question_id}.html'), 'wt') as fp:
            fp.write(content)

    os.system(f"""cd "{plots_dir}"; rm -f ../"{os.path.basename(plots_dir)}".zip""")
    os.system(f"""cd "{plots_dir}"; zip -r ../"{os.path.basename(plots_dir)}".zip *""")


def compact_pred_file(preds_file: str, base_probs_file: str = None):
    dest_preds_file = os.path.splitext(preds_file)[0]+'_no-probs.json'
    final_preds_file = os.path.splitext(preds_file)[0]+'_with-probs.json'
    dest_preds_probs_file = os.path.splitext(preds_file)[0]+'_probs'
    probs = {}
    if base_probs_file:
        probs = np.load(base_probs_file, allow_pickle=True).item()
    with open(preds_file, 'rt') as rfp, open(dest_preds_file, 'wt') as wfp, tqdm(desc='compact_pred_file', total=os.path.getsize(preds_file)) as pbar:
        for line in rfp:
            pbar.update(len(line))
            l = json.loads(line)
            qid = l['metadata']['question_id']
            assert isinstance(qid, str), qid
            tokens = l['metadata']['tokens']
            n = len(tokens)
            line_probs = {}
            if 'arc_probs' in l:
                line_probs['arc_probs'] = np.array(l.pop('arc_probs'))[:n, :n]
            if 'arc_tag_probs' in l:
                line_probs['arc_tag_probs'] = np.array(l.pop('arc_tag_probs'))[:n, :n]
            if line_probs:
                probs[qid] = line_probs
            wfp.write(json.dumps(l)+'\n')
    np.save(dest_preds_probs_file, probs, allow_pickle=True)
    os.rename(preds_file, final_preds_file)
    os.rename(dest_preds_file, preds_file)


def get_enumerate_preds(preds_file: str, question_ids: List[str] = None):
    def enumerator():
        probs = np.load(os.path.splitext(preds_file)[0]+'_probs.npy', allow_pickle=True)
        # preds_file_ = os.path.splitext(preds_file)[0]+'_no-probs.json'
        with open(preds_file, 'rt') as rfp, tqdm(desc='read-preds', total=os.path.getsize(preds_file)) as pbar:
            for line in rfp:
                pbar.update(len(line))
                l = json.loads(line)
                qid = l['metadata']['question_id']
                if not (question_ids is None or qid in question_ids):
                    continue
                l.update(probs.item().get(qid))
                yield l
    return enumerator


def get_enumerate_preds_original(preds_file:str, question_ids: List[str] = None):
    def enumerator():
        proccessed_qids = []

        with open(preds_file, 'rt') as rfp, tqdm(desc='enumerate-preds', total=os.path.getsize(preds_file)) as pbar:
            for line in rfp:
                pbar.update(len(line))
                l = json.loads(line)
                qid = l['metadata']['question_id']
                proccessed_qids.append(qid)
                if question_ids is None or qid in question_ids:
                    l['arc_probs'] = np.array(l['arc_probs'])
                    l['arc_tag_probs'] = np.array(l['arc_tag_probs'])
                    yield l

                if set(proccessed_qids) == set(question_ids or []):
                    break
    return enumerator


def get_enumerate_preds_probs_with_gold(gold_file: str, preds_enumerator):
    gold = {}
    for question_id, (tokens, pos, arcs_tags) in json_file_to_dependencies_graphs(gold_file, gold_to_dependencies):
        gold[question_id] = arcs_tags

    def enumerator():
        for l in preds_enumerator():
            qid = l['metadata']['question_id']
            if qid not in gold:
                continue
            gold_arcs, gold_arc_tags = zip(*[((i,j), t) for i,j,t in gold[qid]])
            l['arcs'] = list(gold_arcs)
            l['arc_tags'] = list(gold_arc_tags)
            yield l
    return enumerator


def compare_objectives(gold_file: str, pred_file: str):
    gold_df = pd.read_json(gold_file, orient='records', lines=True)
    gold_df['question_id'] = gold_df['metadata'].apply(lambda x: x['question_id'])
    pred_df = pd.read_json(pred_file, orient='records', lines=True)
    pred_df['question_id'] = pred_df['metadata'].apply(lambda x: x['question_id'])

    df = gold_df.merge(pred_df, on='question_id', suffixes=('_gold', '_pred'))
    df['is_less_than_gold'] = df['objective_gold'] > df['objective_pred']
    df['is_equal_gold'] = df['objective_gold'] == df['objective_pred']
    print(df.mean())
    df.to_csv(os.path.splitext(pred_file)[0]+'__compared_objective.csv', index=False)


qids = """
ATIS_dev_191
CLEVR_dev_2690
DROP_dev_history_2196_4c3a97cc-6c26-480c-b555-de4c53fda310
CWQ_dev_WebQTrn-1026_b0615bb0ab6c234eca9558c39de1ba69
NLVR2_dev_dev-169-0-0
NLVR2_dev_dev-787-1-0
SPIDER_dev_24
NLVR2_dev_dev-517-2-0
NLVR2_dev_dev-544-3-0
DROP_dev_nfl_1516_772dceb5-8863-4120-b526-aa1b37f22ad3
CWQ_dev_WebQTrn-3064_0281ead9c42fde29e92269a2ccf10495
DROP_dev_nfl_2077_abde882c-56cb-4894-9767-616f2684cc95
SPIDER_dev_464
NLVR2_dev_dev-395-0-0
NLVR2_dev_dev-543-0-1
NLVR2_dev_dev-163-2-0
NLVR2_dev_dev-708-3-1
NLVR2_dev_dev-225-0-0
CLEVR_dev_2293
NLVR2_dev_dev-449-3-1
SPIDER_dev_42
NLVR2_dev_dev-436-3-0
DROP_dev_history_1352_d1397441-2f73-4c7b-8ad5-0cbef7a35ea9
CLEVR_dev_4741
NLVR2_dev_dev-51-3-0
NLVR2_dev_dev-299-3-0
ATIS_dev_329
CLEVR_dev_5113
DROP_dev_history_1450_8be3e663-e0a6-41c6-9f2b-55f9a60bc54d
COMQA_dev_cluster-4163-1
NLVR2_dev_dev-133-1-0
NLVR2_dev_dev-578-0-1
CLEVR_dev_6198
DROP_dev_history_1731_d1d9687f-0cf4-4c5a-81b0-65cb09dec3e7
DROP_dev_history_2162_3c1da219-7fc4-43a3-9543-f7310b8c4281
NLVR2_dev_dev-605-0-0
CLEVR_dev_5333
COMQA_dev_cluster-1674-1
NLVR2_dev_dev-588-3-0
CLEVR_dev_3021
CLEVR_dev_6855
CLEVR_dev_6884
NLVR2_dev_dev-232-2-1
NLVR2_dev_dev-329-0-1
DROP_dev_history_2086_925de638-cd7e-4b29-ac5c-b72836d098e4
NLVR2_dev_dev-818-0-1
NLVR2_dev_dev-30-2-0
SPIDER_dev_393
NLVR2_dev_dev-433-2-1
CLEVR_dev_5770
CWQ_dev_WebQTrn-1632_55881f911c82274c9bea7fe00b3b6793
NLVR2_dev_dev-164-1-1
DROP_dev_history_1665_0e654ddb-d39f-45fc-982e-1b0bba2e2862
SPIDER_dev_504
COMQA_dev_cluster-671-1
ATIS_dev_327
DROP_dev_history_1998_1fe9ed85-656f-49a4-ae00-987ab427e97a
NLVR2_dev_dev-615-1-0
DROP_dev_nfl_1454_4a964c39-47ad-4bba-8942-c98cb289afb3
NLVR2_dev_dev-224-1-0
COMQA_dev_cluster-4162-1
CLEVR_dev_6524
NLVR2_dev_dev-309-0-1
DROP_dev_history_1853_3bdedbf5-1856-438f-9f79-26f3ce4501a4
CWQ_dev_WebQTrn-786_c79e703ecbfb4226ba2041f923f19585
SPIDER_dev_435
DROP_dev_history_13_8e8e69da-bc00-452f-9f62-a5412015ee28
CWQ_dev_WebQTrn-1625_35808d5c3b6a527fc5d78d0b0fc6d27a
CLEVR_dev_6823
NLVR2_dev_dev-367-2-1
NLVR2_dev_dev-274-3-0
CLEVR_dev_4116
NLVR2_dev_dev-775-3-1
DROP_dev_history_1065_ee23be57-a1c1-41c1-8a7a-524171c142f8
DROP_dev_history_1275_7728f949-f615-4bca-8173-4a93673bddd2
DROP_dev_history_1731_a28d736c-4b32-44f3-b24a-987775399f4f
COMQA_dev_cluster-4006-1
NLVR2_dev_dev-740-2-1
DROP_dev_history_2474_ac633f01-14e7-4c02-97a9-857082c694a6
DROP_dev_history_1884_61b15cc4-a284-4504-9858-45f1cf40c1fa
DROP_dev_history_2199_f7819d7b-bf66-4963-beaa-6c39dec5e9a9
NLVR2_dev_dev-408-2-1
NLVR2_dev_dev-675-2-0
CLEVR_dev_6192
NLVR2_dev_dev-776-1-0
CLEVR_dev_4421
NLVR2_dev_dev-765-1-0
NLVR2_dev_dev-738-2-1
COMQA_dev_cluster-3008-1
CLEVR_dev_5095
COMQA_dev_cluster-3275-2
SPIDER_dev_167
CLEVR_dev_7190
DROP_dev_history_2380_c3d501f6-868c-4516-acfb-58a1096e5e4d
NLVR2_dev_dev-871-2-1
COMQA_dev_cluster-4050-1
NLVR2_dev_dev-707-2-0
ATIS_dev_58
NLVR2_dev_dev-11-1-1
NLVR2_dev_dev-361-0-0
"""

if __name__ == '__main__':
    def run_compact(args):
        assert args.preds_file
        compact_pred_file(preds_file=args.preds_file, base_probs_file=args.base_probs)

    def run_decode(args):
        assert args.prediction and args.subdir
        # args.question_ids = args.question_ids or [x for x in qids.split('\n') if x]
        enumerator = None
        if args.gold:
            enumerator = get_enumerate_preds_probs_with_gold(
                args.gold, get_enumerate_preds(preds_file=args.prediction, question_ids=args.question_ids))
        _decode(preds_file=args.prediction, timeout=args.timeout, subdir=args.subdir,
                question_ids=args.question_ids,
                enumerator=enumerator,
                sanity_check=args.sanity_check,
                skip_violation_check=args.all,
                ignore_concatenated_tags=not args.use_concatenated_tags,
                limit_arcs_number=args.limit_arcs, drop_arcs_only=args.drop_only, keep_arcs=args.keep_arcs,
                )

    def run_summarize(args):
        assert args.prediction
        summarize_statistics(args.prediction)

    def run_unified(args):
        assert args.prediction and args.ILP_prediction
        unified_preds_and_eval_files(args.ILP_prediction, args.prediction, args.all)

    parser = argparse.ArgumentParser(description='Decode dependencies graph predictions by ILP')
    subparsers = parser.add_subparsers()

    parser_compact = subparsers.add_parser('compact', help='prepare predictions probabilities')
    parser_compact.set_defaults(func=run_compact)
    parser_compact.add_argument('--preds_file', type=str, help='predictions file with probs')
    parser_compact.add_argument('--base_probs', type=str, required=False, help='base probs file from other compact')

    parser_decode = subparsers.add_parser('decode', help='ILP decode for predictions file')
    parser_decode.set_defaults(func=run_decode)
    parser_decode.add_argument('--prediction', '-p', type=str, help='prediction file with probs')
    parser_decode.add_argument('--timeout', '-t', type=int, required=False, help='timeout in seconds')
    parser_decode.add_argument('--subdir', '-s', type=str, default='ILP_decode', help='subdirectory for outputs')
    parser_decode.add_argument('--question_ids', '-q', nargs='+', required=False, help='specific samples')
    parser_decode.add_argument('--use_concatenated_tags', action="store_true",
                               help='do not ignore tags of the form t1&,,,&tp')

    parser_decode.add_argument('--limit_arcs', action="store_true", help='limit arcs number up to predicted arcs number')
    parser_decode.add_argument('--drop_only', action="store_true", help='drop predicted arcs until satisfy')
    parser_decode.add_argument('--keep_arcs', action="store_true", help='like drop_only, but allowing change tags for arcs')
    parser_decode.add_argument('--sanity_check', action="store_true", help='check current solution')
    parser_decode.add_argument('--all', action="store_true", help='decode all samples (skip violation check)')
    parser_decode.add_argument('--gold', type=str, required=False,
                               help='gold graphs to use instead of independent predictions. Mostly for sanity check.')

    parser_sum = subparsers.add_parser('sum', help='summarize ILP decoded predictions file')
    parser_sum.set_defaults(func=run_summarize)
    parser_sum.add_argument('--prediction', '-p', type=str, help='ILP prediction file')

    parser_uni = subparsers.add_parser('unified', help='summarize ILP decoded predictions file')
    parser_uni.set_defaults(func=run_unified)
    parser_uni.add_argument('--ILP_prediction', '-ilp', type=str, help='ILP prediction file')
    parser_uni.add_argument('--prediction', '-p', type=str, help='original prediction file')
    parser_uni.add_argument('--all', action="store_true", help='')

    args = parser.parse_args()
    args.func(args)

    # compact_pred_file('tmp/_best/biaffine-graph-parser--transformer-encoder/eval_with_probs/dev_preds.json')
    # compare_objectives(
    #     'tmp/_best/biaffine-graph-parser--transformer-encoder/eval_with_probs/_debug_ILP_decoder_gold_sanity_check/dev_preds.json',
    #     'tmp/_best/biaffine-graph-parser--transformer-encoder/eval_with_probs/ILP_decode_100_t=240/dev_preds.json'
    #     )
