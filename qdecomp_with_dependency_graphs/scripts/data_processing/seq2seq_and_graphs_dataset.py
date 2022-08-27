import os
import json
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd

from qdecomp_with_dependency_graphs.dependencies_graph.examine_predictions import predict_all_set
from qdecomp_with_dependency_graphs.scripts.train.run_experiments import eval


def _gold_graph_map(file_path: str) -> Dict[str, Tuple[List[str], List[Tuple[int, int, str]]]]:
    question_id_to_graph = {}
    with open(file_path, 'rt') as fp:
        for line_str in fp.readlines():
            line = json.loads(line_str)
            question_id = line['metadata']['question_id']
            tokens = [x['text'] for x in (line['tokens'] + line['extra_tokens'])]
            dep = line['deps']
            question_id_to_graph[question_id] = (tokens, dep)
    return question_id_to_graph

def _pred_graph_map(file_path: str) -> Dict[str, Tuple[List[str], List[Tuple[int, int, str]]]]:
    question_id_to_graph = {}
    with open(file_path, 'rt') as fp:
        for line_str in fp.readlines():
            line = json.loads(line_str)
            question_id = line['metadata']['question_id']
            tokens = line['metadata']['tokens']
            dep = [(x,y,d) for (x,y), d in zip(line['arcs'], line['arc_tags'])]
            question_id_to_graph[question_id] = (tokens, dep)
    return question_id_to_graph


def merge_seq2seq_and_graphs(
        seq2seq_file_path: str,
        qid_to_graph_map: Dict[str, Tuple[List[str], List[Tuple[int, int, str]]]],
        dest_dir: str,
        suffix: str = '_with_graphs',
        create_partitials: bool = True,
    ):
    os.makedirs(dest_dir, exist_ok=True)
    s2s = pd.read_csv(seq2seq_file_path)
    original_len = len(s2s)

    def get_graph(question_id: str, tuple_ind: int):
        if question_id not in qid_to_graph_map:
            return None
        return json.dumps(qid_to_graph_map[question_id][tuple_ind])
    s2s['question_tokens'] = s2s['question_id'].apply(lambda x: get_graph(x, 0))
    s2s['dependencies'] = s2s['question_id'].apply(lambda x: get_graph(x, 1))
    s2s = s2s[s2s['question_tokens'].notnull()]
    assert len(s2s) == len(qid_to_graph_map)
    dest_file_name = f'{os.path.splitext(os.path.basename(seq2seq_file_path))[0]}{suffix}.csv'
    dest_path = os.path.join(dest_dir, dest_file_name)
    s2s.to_csv(dest_path, index=False)
    with open(os.path.splitext(dest_path)[0]+'_summary.json', 'wt') as fp:
        json.dump({
            'rate': f'{len(s2s)*100.0/original_len:.2f}% ({len(s2s)}/{original_len})'
        }, fp=fp, indent=2)

    if create_partitials:
        def filter_csv(path: str):
            df = pd.read_csv(path)
            df = df[df['question_id'].isin(qid_to_graph_map.keys())]
            dest_path = os.path.join(dest_dir, os.path.basename(path))
            df.to_csv(dest_path, index=False)

        set_ = os.path.basename(seq2seq_file_path).split('_')[0]
        # dataset
        filter_csv(f'datasets/Break/QDMR/{set_}.csv')
        # s2s set
        filter_csv(seq2seq_file_path)


def prepare_with_gold_graphs(graph_dir: str):
    for set_ in ['dev', 'train']:
        merge_seq2seq_and_graphs(
            seq2seq_file_path=os.path.join(graph_dir, f'{set_}_seq2seq.csv'),
            qid_to_graph_map=_gold_graph_map(os.path.join(graph_dir, f'{set_}_dependencies_graph.json')),
            dest_dir=graph_dir,
            # suffix='_with_gold_graphs'
        )


def prepare_with_predicted_graphs(graph_model_dir: str, full_sets: bool = False, working_dir: str = 'eval'):
    pred_dir = os.path.join(graph_model_dir, working_dir)
    for set_ in ['dev', 'train']:
        # create predictions file
        dep_graph_preds_files = list(Path(pred_dir).rglob(f"*{set_}_dependencies_graph__preds.json"))
        assert len(dep_graph_preds_files) <=1, f'found: {dep_graph_preds_files}'
        if len(dep_graph_preds_files) == 0:
            # TODO: fails on 'train' set since token_based_metric (LogicalFormEM) expected dataset_path=dev.csv
            eval(
                model_dir=graph_model_dir,
                dataset_set=set_,
                predict_only=True,
                dest_sub_dir=working_dir
            )
        pred_file = list(Path(pred_dir).rglob(f"*{set_}_dependencies_graph__preds.json"))[0]
        if full_sets:
            predict_all_set(
                models_root=graph_model_dir,
                set_=set_,
                dest_sub_dir=working_dir
            )
            pred_file = list(Path(pred_dir).rglob(f"{set_}_preds.json"))[0]

        # merge
        dest_dir = 'with_graphs' + ('_full' if full_sets else '')
        merge_seq2seq_and_graphs(
            seq2seq_file_path=f'datasets/Break/QDMR/{set_}_seq2seq.csv',
            qid_to_graph_map=_pred_graph_map(str(pred_file)),
            dest_dir=os.path.join(graph_model_dir, working_dir, dest_dir),
            create_partitials=True
        )


if __name__ == '__main__':
    # prepare_with_gold_graphs(graph_dir='datasets/Break/QDMR/')
    prepare_with_predicted_graphs(
        graph_model_dir='tmp/_best/biaffine-graph-parser--transformer-encoder/',
        full_sets=True
    )
    prepare_with_predicted_graphs(
        graph_model_dir='tmp/_best/biaffine-graph-parser--transformer-encoder/',
        full_sets=False
    )