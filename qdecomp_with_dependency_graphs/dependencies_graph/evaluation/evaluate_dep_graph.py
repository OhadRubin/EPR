import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Iterable, List, Callable, Tuple, Optional

import pandas as pd
import spacy
import re

from qdecomp_with_dependency_graphs.dependencies_graph.create_dependencies_graphs import get_tokens_dependencies_generator
from qdecomp_with_dependency_graphs.dependencies_graph.config.configuration_loader import config

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import TokensDependencies, SpansDependencies
from qdecomp_with_dependency_graphs.dependencies_graph.examine_predictions import predict_all_set, json_file_to_dependencies_graphs, \
    prediction_to_dependencies_graph
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_extractors.collapsers import BaseCollapser
from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition

import os
from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.spans_dependencies_to_logical_form_tokens import SpansDepToQDMRStepTokensConverter
from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.qdmr_to_logical_form_tokens import QDMRToQDMRStepTokensConverter
from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.logical_form_matcher import LogicalFromStructuralMatcher, QDMRStepTokensDependencies
from qdecomp_with_dependency_graphs.scripts.eval.evaluate_predictions import get_predictions_from_allennlp_preds_file, format_qdmr

logger = logging.getLogger(__name__)

parser = spacy.load('en_core_web_sm', disable=['ner'])
def get_logical_form_tokens_formatters() -> Tuple[Callable[[str, str, str, Optional[dict]], str], Callable[[str, str, TokensDependencies, Optional[dict]], str]]:
    tokens_dependencies_extractor = config.tokens_dependencies_extractor
    converter = config.spans_dependencies_to_logical_form_converter
    qdmr_converter = QDMRToQDMRStepTokensConverter()
    formatter = LogicalFromStructuralMatcher()

    def _add_meta(meta: dict, qdmr_graph: QDMRStepTokensDependencies, reorder: bool):
        if meta is not None:
            try:
                # meta['original'] = formatter.graph_key(qdmr_graph)
                meta['original'] = qdmr_graph.to_string(reorder=reorder)
            except Exception as ex:
                meta['original'] = f'ERROR: {str(ex)}'

    def dep_graph_formatter(question_id: str, question_text: str, tokens_dependencies: TokensDependencies, meta: dict = None) -> str:
        try:
            # spans dependencies graph
            spans_dependencies: SpansDependencies = tokens_dependencies_extractor.to_spans_dependencies(
                tokens_dependencies=tokens_dependencies)

            # convert to qdmr
            qdmr_graph: QDMRStepTokensDependencies = converter.convert(spans_dependencies)
            _add_meta(meta, qdmr_graph, True)

            formatter.normalize_logical_graph(question_id, question_text, qdmr_graph)

            return formatter.graph_key(qdmr_graph)
        except Exception as ex:
            logging.exception(f'{question_id}: {str(ex)}')
            return "ERROR"

    def qdmr_formatter(question_id: str, question_text: str, decomposition: str, meta: dict = None) -> str:
        try:
            qdmr_graph: QDMRStepTokensDependencies = qdmr_converter.convert(question_id=question_id, question_text=question_text, decomposition=decomposition)
            _add_meta(meta, qdmr_graph, False)
            formatter.normalize_logical_graph(question_id, question_text, qdmr_graph)
            return formatter.graph_key(qdmr_graph)
        except Exception as ex:
            logging.exception(f'{question_id}: {str(ex)}')
            return "ERROR"

    # def format_func(question_id: str, question_text: str, gold_decomposition: str, tokens_dependencies: TokensDependencies):
    #     return qdmr_formatter(question_id, question_text, gold_decomposition), \
    #            dep_graph_formatter(question_id, question_text, tokens_dependencies)

    return qdmr_formatter, dep_graph_formatter


def evaluate(samples, pred_decomps, formatters_dict, dest_path: str, comparable_mode: bool = False):
    old_df = pd.read_csv(dest_path) if (comparable_mode and os.path.exists(dest_path)) else None

    records = []
    for (question_id, question_text, gold), pred in zip(samples, pred_decomps):
        try:
            record = {
                'question_id': question_id,
                'question_text': question_text,
                'gold': gold,
            }
            records.append(record)
            for k, (qdmr_formatter, dep_graph_formatter) in formatters_dict.items():
                gold_meta = {}
                gold = qdmr_formatter(question_id, question_text, gold, gold_meta)
                record[f'gold_{k}'] = gold
                record.update({f'gold_{k}_{mk}': v for mk, v in gold_meta.items()})

                pred_meta = {}
                pred = dep_graph_formatter(question_id, question_text, pred, pred_meta)
                record[f'pred_{k}'] = pred
                record.update({f'pred_{k}_{mk}': v for mk, v in pred_meta.items()})

        except Exception as ex:
            print(f"error in {question_id}: {str(ex)}")

    df = pd.DataFrame.from_records(records)
    # exact match
    for k in formatters_dict:
        df[f'{k}_exact_match'] = (df[f'gold_{k}'].str.lower() == df[f'pred_{k}'].str.lower()) & (df[f'gold_{k}']!="ERROR")

    if comparable_mode and (old_df is not None and len(df) == len(old_df)):
        common_formatters = [x for x in formatters_dict if f'pred_{x}' in old_df.columns]
        if common_formatters:
            old_df.index = df.index
            # for k in formatters_dict:
            #     old_df[f'gold_{k}_new'] = df[f'gold_{k}']
            #     old_df[f'pred_{k}_new'] = df[f'pred_{k}']
            #     old_df[f'{k}_exact_match_new'] = df[f'{k}_exact_match']
            for col in old_df.columns:
                if col not in ['question_id', 'question_id', 'gold']:
                    old_df[f'{col}_new'] = df[col]

            query = ' | '.join(f'{x}!={x}_new' for k in common_formatters for x in [f'gold_{k}', f'pred_{k}', f'{k}_exact_match'])
            diff = old_df.query(query)
            if len(diff) > 0:
                path = os.path.splitext(dest_path)[0]+'-diff.csv'
                print(f'{len(diff)} changes where detected. save to {path}')
                diff.to_csv(path, index=False)

            query = ' | '.join(
                f'{x}!={x}_new' for k in common_formatters for x in [f'{k}_exact_match'])
            diff_em = old_df.query(query)
            if len(diff_em) > 0:
                path = os.path.splitext(dest_path)[0] + '-diff-em.csv'
                print(f'{len(diff_em)} EM changes where detected. save to {path}')
                reg_query = ' | '.join(f'{x}_new==False' for k in common_formatters for x in [f'{k}_exact_match'])
                print(f'{len(diff_em.query(reg_query))} regressions')
                diff.to_csv(path, index=False)

    if dest_path:
        if old_df is not None:
            dest_path = os.path.splitext(dest_path)[0]+'-new.csv'
        df.to_csv(dest_path, index=False)

    # print eval:
    summary = {}
    summary['full'] = df.mean().round(3).to_dict()
    for x in formatters_dict:
        query = f'(gold_{x}!="ERROR" & pred_{x}!="ERROR")'
        df_no_error = df.query(query)
        summary[f'no-error {x}'] = {
            'rate': f'{len(df_no_error)}/{len(df)} ({len(df_no_error)/len(df)*100:.2f} %)',
            **df_no_error.mean().round(3).to_dict()}

    df['dataset'] = df['question_id'].apply(lambda x: x.split('_')[0])
    summary['per_ds'] = df.groupby('dataset').agg("mean").round(3).to_dict()

    print(re.sub('[{},"]', '', json.dumps(summary, indent=2)))
    with open(os.path.splitext(dest_path)[0] + '_summary.json', 'wt') as f:
        json.dump(summary, f, indent=2)
    return summary


def get_samples(df: pd.DataFrame):
    for index, row in df.iterrows():
        question_id, question_text, gold_decomposition, *_ = row
        yield question_id, question_text, gold_decomposition


def evaluate_dep_graph(dataset: pd.DataFrame, dest_dir:str, predicted_graphs: Iterable[TokensDependencies]):
    return evaluate(samples=get_samples(dataset),
                    pred_decomps=predicted_graphs,
                    formatters_dict={
                        'logical_form_tokens': get_logical_form_tokens_formatters()
                    },
                    dest_path=os.path.join(dest_dir, 'evaluate_dep_graph.csv'),
                    comparable_mode=True
                    )


def evaluate_gold_dep_graph(dataset: pd.DataFrame, dest_dir:str):
    preds = (x for question_id, x in get_tokens_dependencies_generator(
        dataset=dataset,
        tokens_dependencies_extractor=config.tokens_dependencies_extractor,
        error_handler=lambda question_id, exception: print(f'error in {question_id}: {str(exception)}')))

    return evaluate(samples=get_samples(dataset),
                    pred_decomps=preds,
                    formatters_dict={
                        'logical_form_tokens': get_logical_form_tokens_formatters()
                    },
                    dest_path=os.path.join(dest_dir, 'evaluate_gold_dep_graph.csv')
                    )


def evaluate_learned_dep_graph(dataset: pd.DataFrame, model_dir: str, skip_exists: bool = True, all_dev: bool = True,
                               preds_file: str = None):
    set_ = dataset['question_id'].iloc[0].split('_')[1]
    assert not (all_dev and preds_file is not None)
    if all_dev:
        preds_file = os.path.join(model_dir, 'eval', f'{set_}_preds.json')
    elif preds_file is None:
        pathlist = Path(model_dir).rglob(f"*{set_}_dependencies_graph__preds.json")
        preds_file = list(pathlist)
        assert len(preds_file)==1, f'expected to a single predictions file. found {len(preds_file)} {preds_file}'
        preds_file = str(preds_file[0])

    dest_path = f'{os.path.splitext(preds_file)[0]}_eval.csv'
    if skip_exists and os.path.exists(dest_path):
        print(f'skipping already exists: {dest_path}')
        return
    if all_dev:
        predict_all_set(models_root=model_dir, set_=set_)
    preds_map = {question_id: x for question_id, x in
                 json_file_to_dependencies_graphs(preds_file, prediction_to_dependencies_graph)}
    if not all_dev:
        dataset = dataset[dataset['question_id'].isin(preds_map.keys())]
    assert len(dataset) == len(preds_map)

    def get_predictions():
        for _, row in dataset.iterrows():
            question_id = row['question_id']
            yield preds_map[question_id]

    evaluate(samples=get_samples(dataset),
             pred_decomps=get_predictions(),
             formatters_dict={
                 'logical_form_tokens': get_logical_form_tokens_formatters()
             },
             dest_path=dest_path
             )


def evaluate_qdmr(dataset: pd.DataFrame, dest_path: str, eval_df: pd.DataFrame = None, preds_file:str = None):
    assert eval_df or preds_file
    def get_predictions():
        if preds_file:
            if preds_file.endswith('.csv'):
                # standard eval format
                df = pd.read_csv(preds_file)
                predictions = df['decomposition'].tolist()
                question_ids = dataset['question_id'].tolist()
                assert len(predictions) == len(question_ids), \
                    f"mismatch size of prediction file {preds_file} ({len(predictions)} predictions, {len(question_ids)} samples)"
            else:
                # allennlp
                predictions, question_ids = get_predictions_from_allennlp_preds_file(preds_file)
            predictions: List[Decomposition] = [format_qdmr(x) for x in predictions]
        else:
            question_ids = eval_df['question_id'].to_list()
            predictions = eval_df['prediction'].to_list()
            predictions: List[Decomposition] = [Decomposition.from_str(x) for x in predictions]
        question_ids_to_predictions = {k: v for k, v in zip(question_ids, predictions)}
        for _, row in dataset.iterrows():
            yield question_ids_to_predictions[row['question_id']].to_break_standard_string()

    formatter, _ = get_logical_form_tokens_formatters()
    res = evaluate(samples=get_samples(dataset),
                   pred_decomps=get_predictions(),
                   formatters_dict={
                       # 'decomp': (lambda question_id, question_text, decomposition, *_: re.sub(r'\s+', ' ', decomposition),
                       #            lambda question_id, question_text, decomposition, *_: re.sub(r'\s+', ' ', decomposition)),
                       'logical_form_tokens': (formatter, formatter)
                   },
                   dest_path=dest_path
                   )

    if eval_df:
        new_eval_df = pd.read_csv(dest_path)
        for c in eval_df.columns:
            if c not in ['question_id', 'question', 'gold', 'prediction', *new_eval_df.columns]:
                new_eval_df[c] = eval_df[c]
        new_eval_df.to_csv(os.path.splitext(dest_path)[0]+'_merged.csv', index=False)

    return res


if __name__ == '__main__':
    def evaluate_gold_graphs(args):
        assert args.dest_dir
        evaluate_gold_dep_graph(dataset=df,
                                dest_dir=args.dest_dir)

    def evaluate_graph_parser(args):
        assert args.root or args.preds_file
        assert not args.preds_file or not args.all
        if args.preds_file is not None:
            model_path = os.path.dirname(os.path.dirname(args.preds_file))
            evaluate_learned_dep_graph(dataset=df,
                                       model_dir=model_path,
                                       all_dev=False,
                                       preds_file=args.preds_file)
        else:
            pathlist = Path(args.root).rglob("model.tar.gz")
            for path in pathlist:
                try:
                    archive_path = str(path)
                    model_path = os.path.dirname(archive_path)
                    evaluate_learned_dep_graph(dataset=df,
                                               model_dir=model_path,
                                               all_dev=args.all)
                except Exception as ex:
                    logger.exception(f'Error on {archive_path}: {str(ex)}')


    def evaluate_qdmr_parser(args):
        assert args.input_file or args.preds_file or (args.root and args.pattern)
        if args.root:
            preds_files = [str(x) for x in Path(args.root).rglob(args.pattern)]
            if not preds_files:
                logger.warning(f"No predictions file found in {args.root} (pattern {args.pattern})")
        elif args.preds_file:
            preds_files = [args.preds_file]
        else:
            preds_files = [None]

        for preds_file in preds_files:
            try:
                dest_path = os.path.splitext(args.input_file or preds_file)[0]+'__lf_eval.csv'
                if args.dest_dir:
                    dest_path = os.path.join(args.dest_dir, os.path.basename(dest_path))
                eval_df = args.input_file and pd.read_csv(args.input_file)
                evaluate_qdmr(dataset=df,
                              eval_df=eval_df,
                              dest_path=dest_path,
                              preds_file=preds_file)
            except Exception as ex:
                logger.exception(f'failed to evaluate: preds_file {preds_file}; input_file {args.input_file}')


    parser = argparse.ArgumentParser(description='evaluate qdmr decomposition using logical form')
    parser.add_argument('--dataset', default="datasets/Break/QDMR/dev.csv", help='path to dataset file')
    subparser = parser.add_subparsers()

    gold_parser = subparser.add_parser('gold', help='evaluate gold graph against original qdmr')
    gold_parser.set_defaults(func=evaluate_gold_graphs)
    gold_parser.add_argument('-d', '--dest-dir', type=str, help='destination directory')

    graph_parser = subparser.add_parser('graph', help='use graph parser predictions')
    graph_parser.set_defaults(func=evaluate_graph_parser)
    graph_parser.add_argument('-r', '--root', type=str, required=False, help='root directory of experiments')
    graph_parser.add_argument('--all', default=False, action='store_true', help='try to evaluate on the whole dev set')
    graph_parser.add_argument('-p', '--preds_file', type=str, required=False, help='specific predictions file')

    qdmr_parser = subparser.add_parser('qdmr', help='use qdmr (say seq2seq) eval file')
    qdmr_parser.set_defaults(func=evaluate_qdmr_parser)
    qdmr_parser.add_argument('-i', '--input-file', type=str, required=False, help='evaluation file')
    qdmr_parser.add_argument('-p', '--preds_file', type=str, required=False, help='predictions file')
    qdmr_parser.add_argument('-r', '--root', type=str, required=False, help='predictions root dir')
    qdmr_parser.add_argument('--pattern', default='*dev_seq2seq__preds.json', help='pattern for predictions files (when root is given)')
    qdmr_parser.add_argument('-d', '--dest-dir', required=False, type=str,
                             help='destination directory (default: same as evaluation file)')

    args = parser.parse_args()
    df = pd.read_csv(args.dataset)
    args.func(args)