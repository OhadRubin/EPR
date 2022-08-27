from typing import List, Iterable

import pandas as pd
import numpy as np
import os
import json


def partial(dataset_path: str, keys: List[str], values: Iterable, dest_dir: str):
    df = pd.read_csv(dataset_path, delimiter=',')
    i1 = df.set_index(keys).index
    new_df = df[i1.isin(values)]
    dest_path = os.path.join(dest_dir, os.path.basename(dataset_path))
    assert len(new_df) == len(values), 'mismatch length'
    new_df.to_csv(dest_path, index=False)


def partial_by_dependencies_graphs(dataset_to_cut_dir: str, dependencies_graph_parser_dir:str):
    for x in ['dev', 'train', 'test']:
        graphs_path = os.path.join(dependencies_graph_parser_dir, f'{x}_dependencies_graph.json')
        if not os.path.exists(graphs_path):
            print(f'skip {x}: could not find {graphs_path}')
            continue
        graphs_df = pd.read_json(graphs_path, orient='records', lines=True)
        qids = [r['question_id'] for r in graphs_df['metadata']]
        partial(os.path.join(dataset_to_cut_dir, f'{x}_seq2seq.csv'), ['question_id'], qids, dependencies_graph_parser_dir)
        partial(os.path.join(dataset_to_cut_dir, f'{x}.csv'), ['question_id'], qids, dependencies_graph_parser_dir)


def partial_by_dependencies_graphs_with_rate(dataset_to_cut_dir: str, dependencies_graph_parser_dir:str, dest_root: str,
                                             rate: float):
    for x in ['train']:
        dest_dir = os.path.join(dest_root, f'rand_{rate}')
        try:
            os.makedirs(dest_dir)
        except Exception as ex:
            print(f'skip {x}: could not create {dest_dir}. {str(ex)}')
            continue
        graphs_path = os.path.join(dependencies_graph_parser_dir, f'{x}_dependencies_graph.json')
        if not os.path.exists(graphs_path):
            print(f'skip {x}: could not find {graphs_path}')
            continue
        graphs_df = pd.read_json(graphs_path, orient='records', lines=True)
        mask = np.random.rand(len(graphs_df)) < rate
        graphs_df = graphs_df[mask]
        qids = [r['question_id'] for r in graphs_df['metadata']]
        graphs_df.to_json(os.path.join(dest_dir, f'{x}_dependencies_graph.json'), orient='records', lines=True)
        partial(os.path.join(dataset_to_cut_dir, f'{x}_seq2seq.csv'), ['question_id'], qids, dest_dir)
        partial(os.path.join(dataset_to_cut_dir, f'{x}.csv'), ['question_id'], qids, dest_dir)


def partial_by_dependencies_graphs__leave_one_out(dataset_to_cut_dir: str, dependencies_graph_parser_dir:str, dest_root: str):
    for x in ['dev', 'train']:
        graphs_path = os.path.join(dependencies_graph_parser_dir, f'{x}_dependencies_graph.json')
        if not os.path.exists(graphs_path):
            print(f'skip {x}: could not find {graphs_path}')
            continue
        total_graphs_df = pd.read_json(graphs_path, orient='records', lines=True)
        total_qids = [r['question_id'] for r in total_graphs_df['metadata']]
        datasets = set(x.split('_')[0] for x in total_qids)
        for ds in datasets:
            dest_dir = os.path.join(dest_root, ds)
            try:
                os.makedirs(dest_dir, exist_ok=True)
            except Exception as ex:
                print(f'skip {x}: could not create {dest_dir}. {str(ex)}')
                continue
            mask = [not x.startswith(ds) for x in total_qids]
            graphs_df = total_graphs_df[mask]
            qids = [r['question_id'] for r in graphs_df['metadata']]
            graphs_df.to_json(os.path.join(dest_dir, f'{x}_dependencies_graph.json'), orient='records', lines=True)
            partial(os.path.join(dataset_to_cut_dir, f'{x}_seq2seq.csv'), ['question_id'], qids, dest_dir)
            partial(os.path.join(dataset_to_cut_dir, f'{x}.csv'), ['question_id'], qids, dest_dir)


def check_partial(src_dir: str):
    for x in ['dev', 'train', 'test']:
        try:
            graph_path = os.path.join(src_dir, f'{x}_dependencies_graph.json')
            if not os.path.exists(graph_path):
                continue
            graphs_df = pd.read_json(graph_path, orient='records', lines=True)
            qids = graphs_df['metadata'].apply(lambda x: x['question_id'])
            for ds in [f'{x}.csv', f'{x}_seq2seq.csv']:
                ds_path = os.path.join(src_dir, ds)
                df = pd.read_csv(ds_path)
                if not (df['question_id'] == qids).all():
                    print(f'found mismatch {src_dir}, {x}')
                print(f'{ds_path} is ok')
        except Exception as ex:
            print(f'error in {src_dir}, {x}. {str(ex)}')


if __name__ == '__main__':
    for rate in [0.01, 0.05, 0.1, 0.2, 0.5]:
        partial_by_dependencies_graphs_with_rate('datasets/Break/QDMR', 'datasets/Break/QDMR',
                                                 'datasets/partial_train_set',
                                                 rate)

    # check_partial('datasets/partial_train_set/rand_0.5')

    partial_by_dependencies_graphs__leave_one_out('datasets/Break/QDMR', 'datasets/Break/QDMR',
                                                 'datasets/partial_train_set/leave_one_out'
                                                  )