import re
from typing import Dict
import argparse
import json
import logging
import os
from datetime import datetime
import traceback

import statistics
import pandas as pd
from pathlib import Path
import numpy as np
from pprint import pprint


def main(model_to_eval_file: Dict[str, str], dest_file_path: str, none: str):
    models_to_df = {k: pd.read_csv(v) for k, v in model_to_eval_file.items()}
    df: pd.DataFrame = list(models_to_df.values())[0][['question_id', 'question_text', 'gold']].copy()
    for k, v in models_to_df.items():
        df[k] = v['logical_form_tokens_exact_match']
    df['text'] = df['question_text']
    sorted_keys = sorted(models_to_df.keys())

    def set_label(row):
        labels = []
        for k in sorted_keys:
            if row[k]:
                labels.append(k)
        if labels:
            return '&'.join(labels)
        return none

    df['correct_models'] = df.apply(set_label, axis=1)
    df['label'] = df['correct_models']
    df.to_json(dest_file_path, orient='records', lines=True)

    for model in models_to_df:
        df['label'] = df['correct_models'].apply(lambda x: model if model in x else none)
        dest_dir = os.path.join(os.path.dirname(dest_file_path), model)
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, os.path.basename(dest_file_path))
        df.to_json(path, orient='records', lines=True)


def combine_preds(gold_labels_file: str, pred_labels_file: str, main_model: str = None, binary_mode: bool = False):
    gold_df = pd.read_json(gold_labels_file, orient='records', lines=True)
    preds_df = pd.read_json(pred_labels_file, orient='records', lines=True)
    correct = []
    for (_, gold_row), (_, pred_row) in zip(gold_df.iterrows(), preds_df.iterrows()):
        gold_labels = [x for x in gold_row['correct_models'].split('&') if x!='NONE']
        pred_labels = [x for x in pred_row['label'].split('&') if x!='NONE']
        if binary_mode:
            assert len(pred_labels) <= 1
            correct.append((any(pred_labels) and (pred_labels[0] in gold_labels)) or
                           (not any(pred_labels) and any(x for x in gold_labels if x != main_model)))
        else:
            # correct.append((any(pred_labels) and all(x in gold_labels for x in pred_labels)) or
            #                ((not any(pred_labels)) and (main_model is not None) and (main_model in gold_labels)))
            correct.append((len(pred_labels) == 1 and (pred_labels[0] in gold_labels)) or
                          (len(pred_labels) != 1 and (main_model is not None) and (main_model in gold_labels)))

    preds_df['new_logical_form_em'] = correct
    preds_df['gold_label'] = gold_df['label']
    preds_df['gold_correct_models'] = gold_df['correct_models']
    preds_df['question_id'] = gold_df['question_id']
    base_file = os.path.splitext(pred_labels_file)[0]+'__ensemble'
    preds_df.to_csv(base_file+'.csv', index=False)

    summary = preds_df.mean().round(3).to_dict()

    # confusion_matrix = pd.crosstab(preds_df['gold_correct_models'], preds_df['label'], rownames=['Gold'], colnames=['Predicted'])
    # confusion_matrix_norm = pd.crosstab(preds_df['gold_correct_models'], preds_df['label'], rownames=['Gold'], colnames=['Predicted'], normalize='all')*100
    for n in [False, 'all', 'pred']:
        cm = pd.crosstab(preds_df['gold_correct_models'],
                         preds_df['label'],
                         rownames=['Gold'], colnames=['Predicted'],
                         margins=True,
                         normalize={'gold': 'index', 'pred': 'columns'}.get(n, n))
        if n:
            cm = (cm * 100).round(2)
        summary[f'confusion_matrix(normalized={n})']= cm.to_string()
    with open(base_file+'_summary.json', 'wt') as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    # print(json.dumps(summary, indent=2))
    for k, v in summary.items():
        if isinstance(v, str):
            v = re.sub(r" +", "\t", v)
        print(k, '\n', v, '\n')


def split_dataset(*ensemble_dataset_path: str, train_ratio: float = 0.5):
    path_and_df = [(x,pd.read_json(x, orient='records', lines=True)) for x in ensemble_dataset_path]
    assert all(len(path_and_df[0][1]) == len(df) for _, df in path_and_df)
    assert all(path_and_df[0][1]['question_id'].equals(df['question_id']) for _, df in path_and_df)

    train_mask = np.random.rand(len(path_and_df[0][1])) < train_ratio

    for path, df in path_and_df:
        train_df = df[train_mask]
        dev_df = df[~train_mask]
        ens_file_name, ens_file_ext = os.path.splitext(os.path.basename(path))
        dest_sub_directory = f'{ens_file_name}__ratio_{train_ratio}__train_{len(train_df)}_dev_{len(dev_df)}'
        dir_path = os.path.join(os.path.dirname(path), dest_sub_directory)
        os.makedirs(dir_path, exist_ok=True)
        train_df.to_json(os.path.join(dir_path, f'train{ens_file_ext}'), orient='records', lines=True)
        dev_df.to_json(os.path.join(dir_path, f'dev{ens_file_ext}'), orient='records', lines=True)
        summary_dataset(dir_path, f'train{ens_file_ext}', f'dev{ens_file_ext}')


def summary_dataset(dir_path: str, train_file: str, dev_file: str):
    train_path = os.path.join(dir_path, train_file)
    dev_path = os.path.join(dir_path, dev_file)
    train_df = pd.read_json(train_path, orient='records', lines=True)
    dev_df = pd.read_json(dev_path, orient='records', lines=True)
    summary = {
        'new_train': train_df.mean().round(4).to_dict(),
        'new_dev': dev_df.mean().round(4).to_dict(),
    }
    if 'correct_models' in dev_df:
        all_models = list(sorted([x.split('&') for x in dev_df['correct_models']],
                                 key=lambda x: len(x), reverse=True))[0]
        all_correct = dev_df['correct_models'].apply(lambda x: all(m in x for m in all_models))
        some_correct = dev_df['correct_models'].apply(lambda x: any(m in x for m in all_models))
        summary.update({
            'lower_bound': f'{all_correct.mean().round(4)} ({all_correct.sum()}/{all_correct.count()})',
            'upper_bound': f'{some_correct.mean().round(4)} ({some_correct.sum()}/{some_correct.count()})',
        })
    with open(os.path.join(dir_path,'eval_summary.json'), 'wt') as fp:
        json.dump(summary, fp, indent=2)
    print(dir_path)
    print(json.dumps(summary, indent=2))


def complete_graphs_only(dev_file, full_dev):
    dev_df = pd.read_json(dev_file, orient='records', lines=True)
    full_dev_df = pd.read_json(full_dev, orient='records', lines=True)
    # dev_df = dev_df[['question_id']]
    # dev_df['question_id'] = dev_df['question_id'].astype(str)
    # joint = dev_df.join(full_dev_df, on=['question_id'])
    dev_df = dev_df[['question_id']]
    additional = {'label':[], 'copynet-bert': []}
    for qid in dev_df['question_id']:
        row = full_dev_df[full_dev_df['question_id']==qid].iloc[0]
        for col in additional:
            additional[col].append(row[col])
    for col in additional:
        dev_df[col] = additional[col]
    dev_df.to_json(os.path.splitext(dev_file)[0]+'_full.json', orient='records', lines=True)


def run_create_datasets():
    parser = argparse.ArgumentParser(
        description="""prepare ensemble datasets
                    example:
                    -o datasets/ensemble/2021-01-01/dev.json --none NONE
                    {\"copynet-bert\":\"tmp/_best/copynet--transformer-encoder/eval/evaluate_qdmr.csv\",
                    \"graph-parser\":\"tmp/_best/biaffine-graph-parser--transformer-encoder/eval/dev_preds_eval.csv\"}
                    """
    )
    parser.add_argument('input_files', type=json.loads, help='json-formatted string with models eval files'
                                                           'for example: {"copynet-bert": "...path...", ...}')
    parser.add_argument('-o', '--output_file_path', type=str, help='path to output file')
    parser.add_argument('--none', default='NONE', help='none lable')

    args = parser.parse_args()
    assert all(os.path.exists(x) for x in args.input_files.values())
    assert args.output_file_path and args.none

    output_dir = os.path.dirname(args.output_file_path)
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(format='%(message)s',
                        level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(filename=os.path.join(output_dir, 'ensemble_datasets.log'),
                             mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info(f'========================= {datetime.now().strftime("%d.%m.%Y %H:%M:%S")} ==========================')

    try:
        main(model_to_eval_file=args.input_files, dest_file_path=args.output_file_path, none=args.none)
    except Exception as ex:
        logger.exception(f'exception: {str(ex)}')
        traceback.print_exc()


def run_combine(none: str, default_model:str = 'copynet-bert'):
    # combine_preds(gold_labels_file='datasets/ensemble/2021-01-01/graphs_only/dev.json',
    #               pred_labels_file='tmp/ensemble/datasets_ensemble_2021-01-01_graphs_only/ensemble/sample-classification--bert/sample-classification--bert__num_epochs=100_patience=50/eval/datasets_ensemble_2021-01-01_graphs_only_dev__preds.json',
    #               )
    #
    # combine_preds(gold_labels_file='datasets/ensemble/2021-01-01/graphs_only/dev.json',
    #               pred_labels_file='tmp/ensemble/datasets_ensemble_2021-01-01/ensemble/sample-classification--bert/sample-classification--bert__num_epochs=100_patience=50/eval/datasets_ensemble_2021-01-01_dev__preds.json',
    #               main_model='graph-parser')
    #
    # combine_preds(gold_labels_file='datasets/ensemble/2021-01-01/dev.json',
    #               pred_labels_file='datasets/ensemble/2021-01-01/dev.json')

    pathlist = Path(working_dir).rglob("*__preds.json")
    for p in sorted(pathlist):
        with open(os.path.join(p.parent.parent, 'config.json'), 'rt') as fp:
            conf = json.load(fp)
        gold_file = conf['validation_data_path']

        gold_df = pd.read_json(gold_file, orient='records', lines=True)
        labels = [x for x in gold_df['label'].unique() if x != none]
        if len(labels) == 1:
            main_model = labels[0]
            binary = True
        else:
            main_model = default_model
            binary = False
        print(p)
        combine_preds(gold_labels_file=gold_file,
                      pred_labels_file=str(p),
                      main_model=main_model, binary_mode=binary)


def run_split_dataset():
    for ratio in [0.2, 0.5, 0.8]:
        split_dataset('datasets/ensemble/2021-01-18/dev.json',
                      'datasets/ensemble/2021-01-18/graph-parser/dev.json',
                      'datasets/ensemble/2021-01-18/copynet-bert/dev.json',
                      train_ratio=ratio)


def run_summary():
    pathlist = Path('datasets/ensemble/2021-01-18').rglob('*/dev.json')
    for p in sorted(pathlist):
        dir_path = str(p.parent)
        summary_dataset(dir_path=dir_path, train_file='train.json', dev_file='dev.json')


def run_merge():
    from shutil import copyfile

    metadata = pd.read_csv(os.path.join(working_dir, 'graph-parser_vs_copynet_predictions.csv'))

    dest = os.path.join(working_dir, '_ensemble_merged')
    os.makedirs(dest, exist_ok=True)

    pathlist = Path(working_dir).rglob("*_ensemble.csv")
    for p in pathlist:
        df = pd.read_csv(str(p))
        metadata_ = metadata[metadata['question_id'].isin(df['question_id'])].copy()
        df = metadata_.merge(df, on='question_id')
        df.to_csv(os.path.join(dest, p.name), index=False)
        try:
            summary_name = os.path.splitext(p.name)[0] + '_summary.json'
            src = os.path.join(p.parent, summary_name)
            dst = os.path.join(dest, summary_name)
            copyfile(src, dst)
        except:
            pass

if __name__ == '__main__':
    working_dir = 'tmp/ensemble_freeze/'

    # run_create_datasets()
    # run_split_dataset()
    # run_summary()

    run_combine('NONE')
    # run_merge()

    # complete_graphs_only('datasets/ensemble/2021-01-01/graphs_only/dev__ratio_0.8__train_6208_dev_1552/dev.json',
    #                      'datasets/ensemble/2021-01-01/dev.json')

