import sys
from pathlib import Path
import os
from shutil import copyfile
import argparse
import re
import pandas as pd
from collections import OrderedDict
import json

import logging
_logger = logging.getLogger(__name__)


def extract_full_scores(file_path):
    scores = {}
    df = pd.read_csv(file_path)
    df = df.replace(True, 1).replace(False, 0)  # deal with NaN values in boolean column like "exact match"
    mean = df.mean(numeric_only=True)
    max = df.max(numeric_only=True)
    min = df.min(numeric_only=True)
    for index, value in mean.items():
        scores[index] = "{:.3f} ({:.3f}-{:.3f})".format(value, min[index], max[index])

    filename, file_extension = os.path.splitext(file_path)
    write_correlations(df, "{}{}".format(filename, "_correlations.txt"))
    return scores


def write_correlations(df:pd.DataFrame, out_path:str):
    corr = df.corr()
    with open(out_path, "wt") as f:
        f.write(str(corr))


def extract_log_scores(file_path):
    score_pattern = r'^(.+)(?: score?):\s*mean\s*(\S+)\s*max\s*(\S+)\s*min\s*(\S+)\s*$'
    scores = {}
    with open(file_path,'rt') as f:
        for l in f.readlines():
            s=re.search(score_pattern, l)
            if s:
                score, mean, max, min = s.groups()
                scores[score]=f"{mean} ({min}-{max})"

    return scores


def extract_summary_scores(file_path: str):
    with open(file_path, 'r') as f:
        d = json.load(f)
        df = pd.json_normalize(d, sep='.')
        return df.to_dict(orient='records')[0]


def extract_metrics(file_path: str):
    metrics = extract_summary_scores(file_path)
    return {k: v for k, v in metrics.items()
            if k.startswith('best_') or k in ['best_epoch', 'training_duration',
                                              'training_start_epoch', 'training_epochs',
                                              'training_loss']}


def summarize(files_generator, get_scores, dest:str, nested_level: int = None):
    trained_dataset_pattern = r'datasets_([^/\\]+)'
    eval_dataset_pattern = r'datasets_([^/\\]+_(?:train|dev|test))'

    data = []
    for file_path in files_generator:
        model_dir = os.path.dirname(file_path)
        if nested_level is not None:
            for _ in range(nested_level):
                model_dir = os.path.dirname(model_dir)
        else:
            while True:
                if os.path.exists(os.path.join(model_dir, 'model.tar.gz')):
                    break
                if model_dir == os.path.dirname(model_dir):
                    model_dir = os.path.dirname(file_path)
                    break
                model_dir = os.path.dirname(model_dir)
        model = os.path.basename(os.path.dirname(model_dir))
        params = os.path.basename(model_dir).replace(model, '')
        s = re.search(trained_dataset_pattern, file_path)
        trained_dataset, = s.groups() if s else ('',)
        s = re.search(eval_dataset_pattern, os.path.basename(file_path))
        eval_dataset, = s.groups() if s else ('',)
        scores = get_scores(file_path)
        data.append(OrderedDict({'trained_dataset': trained_dataset, 'model': model, 'params': params, 'dir': model_dir,
                                 'eval_dataset': eval_dataset, 'summary_path': file_path, 'summary_file': os.path.basename(file_path),
                                 **scores}))  # preserves columns order
    if not data:
        _logger.warning(f'no files found for {dest}')
        return
    df = pd.DataFrame.from_records(data)
    common_pref = os.path.commonpath(df['dir'].to_list())+os.path.sep
    df.insert(df.columns.get_loc('dir'), 'dir_suff', df['dir'].apply(lambda x: x.replace(common_pref, '')))
    df = df.sort_values(['trained_dataset', 'model', 'params'])
    with open(dest, 'wt') as f:
        df.round(3).to_csv(f, index=False)


def files_enumerator(src_path:str, patterns:[str]):
    for pattern in patterns:
        pathlist = Path(src_path).glob(pattern)
        for path in pathlist:
            path_in_str = str(path)
            yield path_in_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="summarizes eval metrics to one csv ")
    parser.add_argument('--exp_dir', default='tmp', help='path to experiments directory (Default: tmp)')
    parser.add_argument('--by_log', action='store_true', help='use eval logs (.txt aggregated) '
                                                              'in case no summary nor full evaluation files exist')
    parser.add_argument('--by_full', action='store_true', help='use .full evaluation files and  and calculate eval '
                                                               'metrics correlations per model')
    args = parser.parse_args()
    assert os.path.exists(args.exp_dir)

    if args.by_log:
        summarize(files_generator=files_enumerator(args.exp_dir, patterns=["**/*_eval_log.txt"]),
                  get_scores=extract_log_scores,
                  dest=os.path.join(args.exp_dir, 'evals_summary.csv'))
    elif args.by_full:
        summarize(files_generator=files_enumerator(args.exp_dir, patterns=["**/*_eval_full.csv"]),
                  get_scores=extract_full_scores,
                  dest=os.path.join(args.exp_dir, 'evals_summary.csv'))
    else:
        summarize(files_generator=files_enumerator(args.exp_dir, patterns=["**/*_eval_summary.json"]),
                  get_scores=extract_summary_scores,
                  dest=os.path.join(args.exp_dir, 'evals_summary.csv'))

    summarize(files_generator=files_enumerator(args.exp_dir, patterns=["**/metrics.json"]),
              get_scores=extract_metrics,
              dest=os.path.join(args.exp_dir, 'metrics_summary.csv'), nested_level=0)
