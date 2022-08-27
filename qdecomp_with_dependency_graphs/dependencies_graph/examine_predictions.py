import argparse
import re
import shutil
from typing import Dict, Any, List, Tuple
import traceback

import os
import io
import statistics
import numpy as np
import pandas as pd
from allennlp.models.archival import load_archive

from qdecomp_with_dependency_graphs.dependencies_graph.config.configuration_loader import config
from qdecomp_with_dependency_graphs.dependencies_graph.data_types import TokensDependencies, SpansDependencies
from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition
from qdecomp_nlp.predictors.dependencies_graph.biaffine_dependency_parser_predictor import BiaffineDependencyParserPredictor
from collections import defaultdict
from pathlib import Path
import json

import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import FBetaMeasure

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import qdecomp_with_dependency_graphs.utils.data_structures as utils
import qdecomp_with_dependency_graphs.utils.html as utils_html

##############################
#    Utilities               #
##############################
# todo: use allennlp predict instead?


def predict_all_set(models_root, set_:('dev', 'train') = 'dev', dest_sub_dir: str = 'eval'):
    # import here to avoid circular import !
    import qdecomp_with_dependency_graphs.scripts.train.run_experiments  # for include_packages (for prediction)

    df = pd.read_csv(f'datasets/Break/QDMR/{set_}.csv')
    pathlist = Path(models_root).rglob("model.tar.gz")
    for path in pathlist:
        archive_path = str(path)
        model_dir = os.path.dirname(archive_path)
        preds_dir = os.path.join(model_dir, dest_sub_dir)
        dest_path = os.path.join(preds_dir, f'{set_}_preds.json')
        if os.path.exists(dest_path):
            print(f'skip {dest_path}: already exists')
            continue
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        archive = load_archive(archive_path, cuda_device=0)
        predictor = BiaffineDependencyParserPredictor.from_archive(archive, 'biaffine-dependency-parser')

        dep_graph_preds_file = list(Path(preds_dir).rglob(f"*{set_}_dependencies_graph__preds.json"))
        if len(dep_graph_preds_file) == 1:
            with open(str(dep_graph_preds_file[0]), 'rt') as f:
                preds = [json.loads(x.strip()) for x in f.readlines()]
            q_ids_to_preds = {x['metadata']['question_id']:x for x in preds}
        else:
            raise AssertionError('there is no dependencies_graph__preds.json. the prediction will be very slow...')
            q_ids_to_preds = {}

        with open(dest_path, 'wt') as f:
            for _, row in df.iterrows():
                try:
                    if row['question_id'] in q_ids_to_preds:
                        pred = q_ids_to_preds[row['question_id']]
                    else:
                        pred = predictor.predict(row["question_text"], metadata={'question_id': row['question_id']})
                    f.write(json.dumps(pred)+'\n')
                except Exception as ex:
                    raise Exception(f"error while predicting {row['question_id']}: {str(ex)}") from ex


#########################################
#   load data as tokens dependencies    #
#########################################

def prediction_to_dependencies_graph(prediction_dict: Dict[str, Any]) -> TokensDependencies:
    tokens, pos, arcs_tags = prediction_to_dependencies(prediction_dict)
    return TokensDependencies.from_tokens(tokens, pos, arcs_tags)


def prediction_to_dependencies(prediction_dict: Dict[str, Any]) -> Tuple[List[str], List[str], List[Tuple[int, int, str]]]:
    tokens = prediction_dict['metadata']['tokens']
    pos = prediction_dict['metadata']['pos']
    arcs = prediction_dict['arcs']
    arcs_tags = prediction_dict['arc_tags']
    arcs_tags = [(i, j, dep) for (i, j), t in zip(arcs, arcs_tags)
                 for dep in t.split('&')
                 if t != "NONE"]
    return tokens, pos, arcs_tags


def gold_to_dependencies_graph(gold_dict: Dict[str, Any]) -> TokensDependencies:
    tokens, pos, arcs_tags = gold_to_dependencies(gold_dict)
    return TokensDependencies.from_tokens(tokens, pos, arcs_tags)


def gold_to_dependencies(gold_dict: Dict[str, Any]) -> Tuple[List[str], List[str], List[Tuple[int, int, str]]]:
    tokens_ = gold_dict['tokens'] + gold_dict.get('extra_tokens',[])
    tokens = [x['text'] for x in tokens_]
    pos = [x['tag'] for x in tokens_]
    arcs_tags = gold_dict['deps']
    return tokens, pos, arcs_tags


def json_file_to_dependencies_graphs(file_path: str, line_converter) -> Tuple[str, TokensDependencies]:
    with open(file_path, 'rt') as f:
        for line in f.readlines():
            content = json.loads(line.strip())
            question_id = content['metadata']['question_id']
            yield question_id, line_converter(content)


###############################
#       render                #
###############################

def wrap(func):
    try:
        return func()
    except Exception as ex:
        return None

def _tokens_dependencies_to_html(tokens_dependencies: TokensDependencies)->str:
    debug = {}
    spans_dependencies: SpansDependencies = wrap(
        lambda: tokens_dependencies_extractor.to_spans_dependencies(tokens_dependencies=tokens_dependencies,
                                                                    debug=debug))
    pre_unwind_spans_dependencies: SpansDependencies = debug.get('pre_unwind_spans_dependencies', None)
    rebuild_decomposition = wrap(lambda: token_dep_to_qdmr_extractor.extract(tokens_dependencies))
    return f"""
    <div class="rebuild_decomposition" >re-decomposition <pre>{rebuild_decomposition and rebuild_decomposition.to_break_standard_string(multiline=True)}</pre></div>
    <div class="spans_dependencies">spans_dependencies{spans_dependencies and spans_dependencies.render_html()}</div>
    <div class="pre_unwind_spans_dependencies">pre_unwind_spans_dependencies{pre_unwind_spans_dependencies and pre_unwind_spans_dependencies.render_html(include_properties=False)}</div>
    <div class="tokens_dependencies">tokens_dependencies<figure>{tokens_dependencies and tokens_dependencies.render_svg()}</figure></div>
    """


def render_predictions(models_root: str, force: bool = False, join: bool = False, break_gold_data:str ='datasets/Break/QDMR/dev.csv'):
    pathlist = Path(models_root).rglob("*_preds.json")

    gold_cache= {}
    def get_gold_tok_dep_render(gold_dep_data: str):
        if gold_dep_data in gold_cache:
            return gold_cache[gold_dep_data]

        question_id_to_gold = {}
        # if break_gold_data is None:
        #     break_gold_data = os.path.join(os.path.dirname(gold_dep_data),
        #                                    f"{os.path.basename(gold_dep_data).split('_')[0]}.csv")
        for question_id, dependencies_graph in json_file_to_dependencies_graphs(gold_dep_data, gold_to_dependencies_graph):
            question_id_to_gold[question_id] = _tokens_dependencies_to_html(dependencies_graph)
        gold_cache[gold_dep_data] = question_id_to_gold
        return question_id_to_gold

    gold_dep_data_set = set([])
    for path in pathlist:
        try:
            gold_dep_data = _get_gold_dataset_path(str(path.parent.parent))
            gold_dep_data_set.add(gold_dep_data)
            render_json_file(str(path), prediction_to_dependencies_graph, force=force,
                             break_gold_data=break_gold_data,
                             gold_metadata=get_gold_tok_dep_render(gold_dep_data))
        except Exception as ex:
            print(f"ERROR on {path}: {str(ex)}")
            traceback.print_exc()
    if join:
        assert len(gold_dep_data_set) == 1, f'expected one gold data, found {len(gold_dep_data_set)} ({gold_dep_data_set})'
        join_predictions(models_root=models_root)


def render_json_file(file_path: str, line_converter, break_gold_data: str, force=False, gold_metadata=None):
    break_df = pd.read_csv(break_gold_data)

    def get_html(question_id: str, tokens_dependencies: TokensDependencies):
        template = utils_html.CollapsibleHTML()
        row = break_df[break_df['question_id'] == question_id].iloc[0]
        question = row['question_text']
        decomposition = Decomposition.from_str(row['decomposition'], sep=';', ref_regex=r'#(\d+)')
        return template.get_body(
            question_id=question_id,
            body=f"""
            <gold>
            <pre>{question}</pre>
            <pre>{decomposition and decomposition.to_break_standard_string(multiline=True)}</pre>
            {gold_metadata and template.wrap_collapsible('details', gold_metadata.get(question_id, ''))}
            </gold>
            <h3>prediction:</h3>
            <pred>
            {_tokens_dependencies_to_html(tokens_dependencies)}
            </pred>
            """
        )

    print(file_path)
    dest_dir = os.path.join(os.path.dirname(file_path), os.path.splitext(os.path.basename(file_path))[0]+'_plots')
    if not force and os.path.exists(dest_dir):
        raise Exception(f"directory {dest_dir} already exists")
    shutil.rmtree(dest_dir, ignore_errors=True)
    os.makedirs(dest_dir)
    for question_id, dependencies_graph in json_file_to_dependencies_graphs(file_path, line_converter):
        content = get_html(question_id=question_id, tokens_dependencies=dependencies_graph)
        dest_file = os.path.join(dest_dir, f'{question_id}.html')
        with open(dest_file, 'wt') as df:
            df.write(content)
    os.system(f"""cd "{dest_dir}"; rm -f ../"{os.path.basename(dest_dir)}".zip""")
    os.system(f"""cd "{dest_dir}"; zip -r ../"{os.path.basename(dest_dir)}".zip *""")


def join_predictions(models_root:str):
    """
    pre request: run render_predictions()
    :param models_root:
    :param gold_dep_data:
    :param break_gold_data:
    :return:
    """
    pathlist = Path(models_root).rglob("*_preds.json")
    pathlist_by_pred = utils.list_to_multivalue_dict(pathlist, key=lambda x: os.path.splitext(x.name)[0])
    question_id_to_gold: Dict[str, str] = {}
    template = utils_html.ToggleBarHTML(class_names=[
        "rebuild_decomposition","spans_dependencies","pre_unwind_spans_dependencies","tokens_dependencies"
    ])  # according to _tokens_dependencies_to_html

    for pred, pathlist in pathlist_by_pred.items():
        question_id_to_html: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for path in pathlist:
            model_name = os.path.basename(os.path.dirname(os.path.dirname(str(path))))
            for rendered in Path(os.path.splitext(str(path))[0]+'_plots').rglob("*.html"):
                question_id = os.path.splitext(rendered.name)[0]
                with rendered.open('rt') as fp:
                    content = fp.read()
                if question_id not in question_id_to_gold:
                    question_id_to_gold[question_id] = re.search('<gold>([\s\S]*)</gold>', content).group(1)
                question_id_to_html[question_id].append((model_name, re.search('<pred>([\s\S]*)</pred>', content).group(1)))

        dest_dir = os.path.join(models_root, f'summarized_plots__{pred}')
        shutil.rmtree(dest_dir, ignore_errors=True)
        os.makedirs(dest_dir)
        for question_id, contents in sorted(question_id_to_html.items(), key=lambda x: x[0]):
            contents = '\n'.join(f"""<h4>{model}</h4>{c}""" for model, c in sorted(contents, key=lambda x: x[0]))
            html_txt = template.get_body(
                question_id=question_id,
                body=f"""
                    {question_id_to_gold.get(question_id, '')}
                    {contents}
                """
            )
            with open(os.path.join(dest_dir, f'{question_id}.html'), 'wt') as f:
                f.write(html_txt)
        os.system(f"""cd "{dest_dir}"; rm -f ../"{os.path.basename(dest_dir)}".zip""")
        os.system(f"""cd "{dest_dir}"; zip -r ../"{os.path.basename(dest_dir)}".zip *""")


##############################
#       To QDMR              #
##############################

def predictions_to_qdmr(models_root: str):
    pathlist = Path(models_root).rglob("*_preds.json")
    for path in pathlist:
        preds_file = str(path)
        try:
            print(preds_file)
            dest_path = os.path.join(path.parent, 'qdmr', os.path.splitext(path.name)[0]+'_qdmr.csv')
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            # if os.path.exists(dest_path):
            #     continue
            qids = []
            decomps = []
            error_cause = {}
            for question_id, dependencies_graph in json_file_to_dependencies_graphs(str(path),
                                                                                    prediction_to_dependencies_graph):
                try:
                    qids.append(question_id)
                    decomposition = token_dep_to_qdmr_extractor.extract(dependencies_graph)
                    decomposition_str = decomposition.to_break_standard_string()
                    decomps.append(decomposition_str)
                except Exception as ex:
                    decomps.append('ERROR')
                    error_cause[question_id] = str(ex)
            df = pd.DataFrame()
            df['question_id'] = qids
            df['dataset'] = df['question_id'].apply(lambda x: x.split('_')[0])
            df['decomposition'] = decomps
            df['error'] = df['question_id'].apply(lambda x: error_cause.get(x, ''))
            df.to_csv(dest_path, index=False)
        except Exception as ex:
            print(f"ERROR on {preds_file}: {str(ex)}")


def error_rate(models_root: str):
    pathlist = Path(models_root).rglob("*_preds_qdmr.csv")
    for path in pathlist:
        preds_file = str(path)
        try:
            print(preds_file)
            df = pd.read_csv(preds_file)
            df['dataset'] = df['question_id'].apply(lambda x: x.split('_')[0])
            df['is_success'] = df['decomposition'] != 'ERROR'

            with open(os.path.splitext(preds_file)[0]+'__error_rate.txt', 'wt') as fp:
                print(f"{df['is_success'].sum()}/{len(df)} success ({df['is_success'].mean()*100:.2f}%)", file=fp)
                print(df['dataset'].value_counts(), file=fp)
                print((df[['dataset', 'is_success']].groupby('dataset').agg("mean")*100).round(2), file=fp)

        except Exception as ex:
            print(f"ERROR on {preds_file}: {str(ex)}")


def add_has_gold(models_root: str, compare_to: str = None):
    gold_question_ids = {}
    def get_question_ids(model_dir):
        dataset = _get_gold_dataset_path(model_dir)
        if dataset not in gold_question_ids:
            question_ids_list = []
            with open(dataset, 'rt') as f:
                for line in f.readlines():
                    content = json.loads(line.strip())
                    question_id = content['metadata']['question_id']
                    question_ids_list.append(question_id)
            gold_question_ids[dataset] = question_ids_list
        return gold_question_ids[dataset]

    pathlist = Path(models_root).rglob("dev_preds_qdmr.csv")
    for path in pathlist:
        preds_file = str(path)
        df = pd.read_csv(preds_file)
        df['has_gold'] = df['question_id'].apply(lambda x: x in get_question_ids(str(path.parent.parent.parent)))
        df.to_csv(preds_file, index=False)
        if compare_to:
            compare_to_model(eval_path=compare_to, dep_pred_file=preds_file)


def compare_to_model(eval_path: str, dep_pred_file: str):
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)

    eval = pd.read_csv(eval_path)
    pred = pd.read_csv(dep_pred_file)
    eval['has_gold'] = pred['has_gold']
    eval['dataset'] = pred['dataset']
    score_keys = [key for key in eval.columns if key not in ["question_id", "question", "gold", "prediction", "dataset", 'has_gold']]

    def print_df(title:str, df:pd.DataFrame, to_csv=True, dest_file=None):
        print('\n'+title, file=fp)
        df = df.round(3)
        if to_csv:
            s = io.StringIO()
            df.to_csv(s, sep='\t')
            print(s.getvalue(), file=dest_file)
        else:
            print(df, file=dest_file)

    def get_eval_per_dataset(eval_df:pd.DataFrame):
        total = pd.DataFrame(eval_df[score_keys].mean()).transpose()
        total['dataset'] = 'total'
        df = eval_df[['dataset'] + score_keys].groupby('dataset').agg("mean")
        df = df.append(total.set_index('dataset'))
        return df

    eval_gold = eval[eval['has_gold'] == True]
    eval_without_gold = eval[eval['has_gold'] == False]

    with open(os.path.splitext(dep_pred_file)[0] + '__compared.txt', 'wt') as fp:
        df_with_gold = get_eval_per_dataset(eval_gold)
        print_df("with gold", df_with_gold, dest_file=fp)

        df_without_gold = get_eval_per_dataset(eval_without_gold)
        print_df('without gold', df_without_gold, dest_file=fp)

        diff = df_with_gold - df_without_gold
        print_df('diff', diff, dest_file=fp)

        diff_perc = (diff / df_with_gold) * 100
        print_df('diff %', diff_perc, dest_file=fp)


#############################
#   metrics                 #
#############################

def f1_per_tag(models_root: str):
    pathlist = Path(models_root).rglob("*dependencies_graph__preds.json")
    for path in pathlist:
        model_dir = str(path.parent.parent)
        gold_path = _get_gold_dataset_path(model_dir)
        f1_per_tag_(gold_path, model_dir, str(path))


def f1_per_tag_(gold_path: str, model_path: str, preds_path: str):
    _, gold_deps =zip(*json_file_to_dependencies_graphs(gold_path, gold_to_dependencies))
    _, preds_deps = zip(*json_file_to_dependencies_graphs(preds_path, prediction_to_dependencies))

    out_dir = os.path.join(model_path, 'eval/f1_and_confusion')
    os.makedirs(out_dir, exist_ok=True)

    vocab = Vocabulary.from_files(os.path.join(model_path, 'vocabulary'))
    vocab.add_token_to_namespace('NONE', 'labels')
    token_to_index = vocab.get_token_to_index_vocabulary(namespace='labels')
    index_to_token = vocab.get_index_to_token_vocabulary(namespace='labels')

    classes_counts = {k:0 for k in token_to_index.keys()}

    def token_dep_to_adjenc(tokens, deps, counters=None):
        n = len(tokens)
        none_index = token_to_index["NONE"]
        a = np.zeros((n, n), dtype=np.int)
        a.fill(none_index)
        none_counter = n*n
        for u,v,l in deps:
            a[u,v] = token_to_index[l]
            if counters:
                counters[l] += 1
                none_counter -= 1
        if counters:
            counters['NONE'] += none_counter
        return a

    labels_and_indices = list(token_to_index.items())  # freeze order, maybe unnecessary
    labels_indices_without_none = [v for k, v in labels_and_indices if k != "NONE"]

    f1 = FBetaMeasure(labels=[i for l,i in labels_and_indices])
    f1_micro = FBetaMeasure(average='micro', labels=labels_indices_without_none)
    f1_macro = FBetaMeasure(average='macro', labels=labels_indices_without_none)

    golds = []
    preds = []
    for gold_, pred_ in zip(gold_deps, preds_deps):
        gold = token_dep_to_adjenc(gold_[0], gold_[2], classes_counts)
        pred = token_dep_to_adjenc(pred_[0], pred_[2])
        pred_logits = (np.arange(max(index_to_token.keys())+1) == pred[...,None]).astype(int)

        golds.append(gold)
        preds.append(pred)

        gold_t = torch.from_numpy(gold).unsqueeze(0)
        pred_t = torch.from_numpy(pred_logits).unsqueeze(0)
        f1(pred_t, gold_t)
        f1_micro(pred_t, gold_t)
        f1_macro(pred_t, gold_t)

    f1_metrics = {
        'f1': f1.get_metric(),
        'f1_micro': f1_micro.get_metric(),
        'f1_macro': f1_macro.get_metric()
    }

    labels_counts_f = sorted([(l, classes_counts[l], f1_metrics['f1']['fscore'][i]) for l,i in labels_and_indices], key=lambda x: x[1], reverse=True)
    f1_metrics['f1_macro_ignore_not_exists'] = statistics.mean(f for l, count, f in labels_counts_f if (count > 0 and l != 'NONE'))
    with open(os.path.join(out_dir,'f-scores.json'), 'wt') as f:
        json.dump({k:v for k,v in f1_metrics.items() if k!='f1'}, f, indent=2, sort_keys=True)

    labels, counts, fscores = zip(*labels_counts_f)
    none_ind = labels.index('NONE')
    filtered_ind = [none_ind] + [i for i, x in enumerate(counts) if x == 0]
    def filter_out_NONE(lst):
        return [x for i, x in enumerate(lst) if i not in filtered_ind]

    counts_ = filter_out_NONE(counts)
    counts_ = [x/max(counts_) for x in counts_]
    fig, ax = plt.subplots()
    ax.bar(filter_out_NONE(labels), counts_, label="count")
    ax.plot(filter_out_NONE(labels), filter_out_NONE(fscores), label='f1', color="orange")
    ax.legend()
    plt.savefig(os.path.join(out_dir,"f1-scores.png"))

    df = pd.DataFrame.from_records(zip(labels, counts, fscores), columns=['tag', 'count', 'f1'])
    df.to_csv(os.path.join(out_dir,'f1-scores.csv'), index=False)

    def conf_matrix(normalize: [None, 'true', 'pred', 'all'] = None):
        cm = confusion_matrix(np.concatenate([np.asarray(x).reshape(-1) for x in golds]),
                              np.concatenate([np.asarray(x).reshape(-1) for x in preds]),
                              labels=[token_to_index[x] for x in labels],
                              normalize=normalize)
        df = pd.DataFrame(cm, columns=labels, index=labels)
        df.to_csv(os.path.join(out_dir,f'confusion_matrix{"__norm_"+normalize if normalize else ""}.csv'))

    conf_matrix()
    conf_matrix("true")
    conf_matrix("pred")
    conf_matrix("all")


#############################
#       help functions      #
#############################

def _get_gold_dataset_path(model_dir:str):
    with open(os.path.join(model_dir, 'config.json'), 'rt') as fp:
        return json.load(fp)['validation_data_path']


if __name__ == '__main__':
    tokens_dependencies_extractor = config.tokens_dependencies_extractor
    token_dep_to_qdmr_extractor = config.tokens_dependencies_to_qdmr_extractor

    def predict_command(args):
        assert args.models_root and args.dest_sub_dir
        predict_all_set(models_root=args.models_root, dest_sub_dir=args.dest_sub_dir)

    def render_command(args):
        assert args.models_root
        render_predictions(models_root=args.models_root, force=args.force, join=args.join)

    def metrics_command(args):
        assert args.models_root
        f1_per_tag(models_root=args.models_root)

    def qdmr_command(args):
        assert args.models_root
        predictions_to_qdmr(models_root=args.models_root)
        add_has_gold(models_root=args.models_root, compare_to=args.compare_eval)
        error_rate(models_root=args.models_root)

    parser = argparse.ArgumentParser(description='Examine dependencies graph parser predictions')
    subparsers = parser.add_subparsers()

    # predict
    parser_predict = subparsers.add_parser('predict', help='re-predict all dev set')
    parser_predict.set_defaults(func=predict_command)
    parser_predict.add_argument('--models_root', '-r', type=str, help='root directory of models')
    parser_predict.add_argument('-d', '--dest_sub_dir', type=str, default='eval', help='destination sub directory for output')

    # render
    parser_render = subparsers.add_parser('render', help='render model predictions')
    parser_render.set_defaults(func=render_command)
    # parser_render.add_argument('--data', type=str, default=r'datasets/Break/QDMR/dev.csv',
    #                            help='csv dataset file (default: datasets/Break/QDMR/dev.csv)')
    parser_render.add_argument('--models_root', '-r', type=str, help='root directory of models')
    # parser_render.add_argument('-n', '--random_n', type=int, help='amount of random samples')
    # parser_render.add_argument('-q', '--question_ids', nargs='+', type=str,
    #                            help='specific question_ids')
    parser_render.add_argument('--force',  action='store_true', help='replace exists rendered')
    parser_render.add_argument('--join',  action='store_true', help='join the models prediction')

    # metrics
    parser_metrics = subparsers.add_parser('metrics', help='run metrics (f1)')
    parser_metrics.set_defaults(func=metrics_command)
    parser_metrics.add_argument('--models_root', '-r', type=str, help='root directory of models')

    # to qdmr
    parser_qdmr = subparsers.add_parser('qdmr', help='convert to qdmr')
    parser_qdmr.set_defaults(func=qdmr_command)
    parser_qdmr.add_argument('--models_root', '-r', type=str, help='root directory of models')
    parser_qdmr.add_argument('--compare_eval', type=str, help='a model (vanilla) eval file')

    args = parser.parse_args()
    args.func(args)
