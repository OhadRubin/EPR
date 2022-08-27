from typing import Dict, List, Tuple

import argparse
import os
import json
import subprocess
import itertools
import shutil
from pathlib import Path
import time
from collections import defaultdict
import re
import shlex

from heapq import heappush, heappushpop

import torch

from allennlp.common.util import import_module_and_submodules
from allennlp.common import Params
from allennlp.commands.train import train_model_from_file

import logging
_logger = logging.getLogger(__name__)

# _include_package = ['qdecomp_nlp.data.dataset_readers', 'qdecomp_nlp.data.samplers', 'qdecomp_nlp.data.token_indexers',
#                     'qdecomp_nlp.modules.token_embedders',
#                     'qdecomp_nlp.models.seq2seq', 'qdecomp_nlp.models.dependencies_graph', 'qdecomp_nlp.models.hybrid',
#                     'qdecomp_nlp.predictors.seq2seq',
#                     'qdecomp_nlp.training.metrics']
_include_package = ['qdecomp_nlp']
_include_package_str = ' '.join([f'--include-package={x}' for x in _include_package])
for package_name in _include_package:
    import_module_and_submodules(package_name)


# ========================== train ===============================

def train_experiment(experiment: str, serialization_root_dir: str,
                     dataset_dir: str = None,
                     overrides: Dict[str, str] = None,
                     #ext_vars: Dict[str, str] = None,
                     no_eval: bool = None,
                     force: bool = False,
                     ):
    """
    Train a single experiment
    :param experiment: jsonnet configuration file path
    :param serialization_root_dir: serialization root directory.
    the serialization directory is <serialization_root_dir>/<experiment-path>
    :param dataset_dir: dataset directory to use
    :param overrides: override settings by name, e.g lr:0.01
    :param no_eval: skip evaluation
    :return: train the model and evaluate it if needed
    :param force: overwrite destination directory
    """
    assert os.path.exists(experiment), f"experiment doesnt exist {experiment}"

    overrides = overrides or {}
    ext_vars = {k: v for k, v in overrides.items() if '.' not in k}
    settings = Params.from_file(experiment, ext_vars=ext_vars).params

    if dataset_dir:
        for x in ['train_data_path', 'validation_data_path', 'test_data_path']:
            if x in settings:
                # support multitask
                if isinstance(settings[x], dict):
                    for k in settings[x]:
                        settings[x][k] = os.path.join(dataset_dir, os.path.basename(settings[x][k]))
                else:
                    settings[x] = os.path.join(dataset_dir, os.path.basename(settings[x]))
    elif isinstance(settings['train_data_path'], dict):
        # support multitask
        dataset_dir = json.dumps(settings['train_data_path'], sort_keys=True)
    else:
        dataset_dir = os.path.dirname(settings['train_data_path'])

    _val_replace(settings, overrides)
    for k,v in ext_vars.items():
        os.environ[k] = str(v)

    serialization_parent, experiment_name = _get_serialization_parent_dir_and_prefix(serialization_root_dir,
                                                                                     dataset_dir,
                                                                                     experiment)

    def format_val(text: str):
        # return text.replace('\\', '.').replace('/', '.')
        return os.path.basename(text) or os.path.basename(os.path.dirname(text))
    params_str = f'__{"_".join([f"{k}={format_val(v)}" for k,v in overrides.items()])}' if overrides else ''
    dirname = f'{experiment_name}{params_str}'
    if len(dirname) >= 255:
        dirname = dirname[:239]+time.strftime("-%Y%m%d-%H%M%S")
    serialization_dir = os.path.join(serialization_parent, dirname)

    overrides = json.dumps(settings)
    train_model_from_file(
        parameter_filename=experiment,
        serialization_dir=serialization_dir,
        overrides=overrides,
        # include_package=_include_package_str,
        force=force,
        file_friendly_logging=True,
    )

    eval(model_dir=serialization_dir, dataset_set='dev', predict_only=no_eval)


# ========================== evel ===============================

def eval(model_dir:str, dataset_set:str=None, dataset_file:str = None, dataset_dir:str = None, predict_only: bool = False, dest_sub_dir: str = 'eval'):
    """
    Evaluate allennlp model on specific dataset (predict+eval)
    :param model_dir: an allennlp model directory to use
    :param dataset_set: dataset type (train, dev, test) of the model config.json
    :param dataset_dir: replace dataset directory with the given one
    :return: create a new directory, 'model_dir/eval', and generates the predictions and the evaluation output
    in it
    """
    with open(os.path.join(model_dir, 'config.json')) as f:
        settings = json.load(f)
    dataset_set_ = 'validation' if dataset_set == 'dev' else dataset_set
    dataset_path = dataset_file or settings[f'{dataset_set_}_data_path']

    def format_path(path: str):
        return os.path.splitext(path)[0].replace('/', '_').replace('\\', '_')
    # support multitask
    if isinstance(dataset_path, dict):
        dataset_files_list = [(json.dumps({k: v}), v, format_path(v)) for k, v in dataset_path.items()]
    else:
        dataset_files_list = [(dataset_path, dataset_path, format_path(dataset_path))]

    for predict_dataset, dataset_file, output_base in dataset_files_list:
        try:
            if dataset_dir:
                dataset_file = os.path.join(dataset_dir, os.path.basename(dataset_file))

            eval_dir = os.path.join(model_dir, dest_sub_dir)
            os.makedirs(eval_dir, exist_ok=True)

            def execute_cmd(cmd: str, output_file: str) -> str:
                torch.cuda.empty_cache()
                cmd_wrapper = 'bash -c "source ~/anaconda3/etc/profile.d/conda.sh; conda activate qdecomp; {0}"'
                with open(output_file, 'wb') as f:
                    subprocess.check_call(cmd_wrapper.format(cmd.replace('"',r'\"')), stdout=f, shell=True)

            preds_output_file = os.path.join(eval_dir, f"{output_base}__preds.json")
            if os.path.exists(preds_output_file):
               _logger.warning(f'skip {preds_output_file} - already exists')
            else:
                cmd_predict = f"""nohup allennlp predict {shlex.quote(os.path.join(model_dir, 'model.tar.gz'))} {shlex.quote(predict_dataset)} --output-file={shlex.quote(preds_output_file)} --batch-size=32 --use-dataset-reader --cuda-device=0 --silent {_include_package_str}"""
                execute_cmd(cmd_predict, f'{os.path.splitext(preds_output_file)[0]}_log.txt')
            if predict_only:
                continue

            eval_output_file_base = os.path.join(eval_dir, f"{output_base}__eval")
            metadata_file = os.path.join(os.path.dirname(dataset_file), f'{dataset_set}.csv')

            cmd_eval = f"""nohup python scripts/eval/evaluate_predictions.py --dataset_file={shlex.quote(metadata_file)} --preds_file={shlex.quote(preds_output_file)} --allennlp --output_file_base={shlex.quote(eval_output_file_base)} """

            execute_cmd(cmd_eval, f"{eval_output_file_base}_log.txt")
        except Exception as ex:
            _logger.exception(f"Error while evaluating {dataset_file}")

# ========================== tune ===============================

def tune(experiment: str, serialization_root_dir: str,
         dataset_dir: str,
         overrides: Dict[str, List[str]] ,
         dest_dir: str):
    """
    Creates tuning commands set for a model by override hyper-params, to be used in exec_multi.py
    Train the model for each combination of the overrides keys values.
    :param experiment: jsonnet configuration file path
    :param serialization_root_dir: serialization root directory.
    the serialization directory is <serialization_root_dir>/<experiment-path>
    :param dataset_dir: dataset directory to use
    :param overrides: override settings by name, e.g lr:"0.01 0.001" drop:"0.0 0.1"whould tune with
    (lr:0.01, drop:0.0), (lr:0.01, drop:0.1), (lr:0.001, drop:0.0), (lr:0.001, drop:0.1)
    :param dest_dir: directory to create in the commands list file
    :return: creates a file with commands, that can be used with exec_multi.py script
    """
    assert os.path.exists(experiment)

    # cleanup command
    serialization_parent, _ = _get_serialization_parent_dir_and_prefix(serialization_root_dir, dataset_dir, experiment)
    cmd_cleanup = f'python scripts/train/run_experiments.py cleanup --dir {serialization_parent} --top 3'

    configs = itertools.product(*overrides.values())
    cmds = []
    dataset_part = f'--dataset {dataset_dir}' if dataset_dir else ''
    for conf in configs:
        over = []
        for i, key in enumerate(overrides.keys()):
            over.append(f'{key}:{conf[i]}')
        cmd = f"nohup python scripts/train/run_experiments.py train --experiment {experiment} {dataset_part} --no_eval -s {serialization_root_dir} -o {' '.join(over)}"
        cmds.append(f"{cmd} ;{cmd_cleanup}")

    cmd_text = '\n'.join(cmds)
    os.makedirs(dest_dir, exist_ok=True)
    dest_file_path = os.path.join(dest_dir, "tune-"+experiment.replace("/","_").replace("\\", "_"))
    with open(dest_file_path, 'wt') as f:
        f.write(cmd_text)

# ========================== best ===============================

def best(configs_dir: str,
         serialization_root_dir: str,
         dest_dir: str,
         dataset_dir: str = None
         ):
    """
    Creates train commands set for experiments/best configurations, to be used in exec_multi.py
    :param configs_dir: directory with json configurations
    :param serialization_root_dir: serialization root directory.
    the serialization directory is <serialization_root_dir>/<experiment-path>
    :param dataset_dir: dataset directory to use
    :param dest_dir: directory to create in the commands list file
    :return: creates a file with commands, that can be used with exec_multi.py script
    """
    assert os.path.exists(configs_dir)

    dataset_part = f'--dataset {dataset_dir}' if dataset_dir else ''
    cmds = []
    continues_cmds = defaultdict(list)
    for path in Path(configs_dir).rglob('*.json'):
        cmd = f"nohup python scripts/train/run_experiments.py train --experiment {str(path)} {dataset_part} -s {serialization_root_dir}"

        path_str = str(path)
        continues_suff = re.findall(r'_(freeze|tune).json$', path_str)
        if continues_suff:
            continues_cmds[path_str.replace(continues_suff[0], '')].append((path_str, cmd))
        else:
            cmds.append(cmd)

    # continues commands
    for x in continues_cmds.values():
        cmds_ordered = sorted(x, key=lambda x: x[0])
        cmds.append(' ;'.join(c for _,c in cmds_ordered))

    cmd_text = '\n'.join(cmds)
    os.makedirs(dest_dir, exist_ok=True)

    def format_file_name(txt: str):
        return txt.replace("/","_").replace("\\", "_")
    dest_file_path = os.path.join(dest_dir, f"{format_file_name(configs_dir)}{('__'+format_file_name(dataset_dir)) if dataset_dir else ''}.txt")
    with open(dest_file_path, 'wt') as f:
        f.write(cmd_text)

# ========================== cleanup ===============================

def cleanup(experiments_dir:str, top_to_keep:int, clean_rest:bool = False,
            metric: str = 'best_validation_loss', maximize: bool = False):
    pathlist = list(Path(experiments_dir).glob('*/metrics.json'))

    # find top_to_keep experiments
    top_experiments = []
    # heappop pops the smallest element. On minimizing we want to pop the biggest element.
    direction = 1 if maximize else -1
    for m_path in pathlist:
        with open(m_path, 'rt') as f:
            metrics = json.loads(f.read())
            func = heappush if len(top_experiments)<top_to_keep else heappushpop
            func(top_experiments, (direction*metrics[metric], m_path.parent))

    if len(top_experiments)>0:
        keep = [p for _,p in top_experiments]
        if clean_rest:
            _clean_directory(experiments_dir, [p.name for p in keep])
        else:
            for x in pathlist:
                if x.parent not in keep:
                    print('removing -> ' + str(x.parent))
                    _remove_by_path(x.parent)
                else:
                    print('keeping -> ' + str(x.parent))


def _clean_directory(dirpath:str, folders_to_exclude:[str]):
    for root, dirs, files in os.walk(dirpath, topdown=True):
        for items_set in [dirs, files]:
            for item in items_set:
                full_path = os.path.join(root, item)
                if len([f for f in folders_to_exclude if f in full_path])==0:
                    print('removing -> ' + full_path)
                    _remove_by_path(full_path)
                else:
                    print('keeping -> ' + full_path)


def _remove_by_path(path):
    """
    Remove the file or directory
    """
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError:
            print("Unable to remove folder: %s" % path)
    else:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            print("Unable to remove file: %s" % path)


# ========================== utils ===============================

def _get_serialization_parent_dir_and_prefix(serialization_root: str, dataset_dir: str, experiment: str):
    path = os.path.splitext(experiment)[0]
    path = os.path.join(*(path.split(os.path.sep)[1:]))  # remove 'experiments' prefix
    dataset_dir = re.sub(r'\W+', '_', dataset_dir)
    return os.path.join(serialization_root, dataset_dir, path), os.path.basename(path)


def _val_replace(dic: Dict, values: Dict[str,str]):
    values_to_paths = {k:[] for k in values}

    def try_get_by_partial_suffix(values: Dict[str, str], path: str):
        parts = path.split('.')
        partials = []
        for i in range(len(parts)):
            partial = '.'.join(parts[i:])
            if partial in values:
                partials.append(partial)
        assert len(partials) <= 1, f'conflicts in values replacement: {partials}'
        if partials:
            return partials[0]
        return None

    def recursive_val_replace(dic: Dict, values: Dict[str, str], prefix: str):
        for k, v in dic.items():
            cur_prefix = f'{prefix}.{k}' if prefix else k
            val_key = try_get_by_partial_suffix(values, cur_prefix)
            if val_key:
                if type(v) == bool and type(values[val_key])==str:
                    dic[k] = (values[val_key].lower().strip() == "true")
                else:
                    dic[k] = type(v)(values[val_key])
                values_to_paths[val_key].append(cur_prefix)
            elif isinstance(v, dict):
                recursive_val_replace(v, values, prefix=cur_prefix)
            elif isinstance(v, list):
                for v2 in v:
                    if isinstance(v2, dict):
                        recursive_val_replace(v2, values, prefix=cur_prefix)

    recursive_val_replace(dic,values,'')
    multi_paths_map = {k:v for k,v in values_to_paths.items() if len(v)>1}
    if multi_paths_map:
        _logger.warning(f'multiple paths where set by a single value: {multi_paths_map}')


if __name__ == '__main__':
    def run_best(args):
        assert args.configs_dir and args.serialization_dir and args.dest_dir
        best(configs_dir=args.configs_dir,
             serialization_root_dir=args.serialization_dir,
             dataset_dir=args.dataset,
             dest_dir=args.dest_dir)

    def run_train(args):
        assert args.experiment and args.serialization_dir
        overrides = {k:v for k,v in [x.split(':', 1) for x in args.overrides]} if args.overrides else None
        train_experiment(experiment=args.experiment,
                         serialization_root_dir=args.serialization_dir,
                         overrides=overrides,
                         dataset_dir=args.dataset,
                         no_eval=args.no_eval,
                         force=args.force)

    def run_tune(args):
        assert args.experiment and args.serialization_dir and args.dest_dir
        overrides = {k: v.split(' ') for k, v in [x.split(':') for x in args.overrides]} if args.overrides else None
        tune(experiment=args.experiment,
             serialization_root_dir=args.serialization_dir,
             overrides=overrides,
             dataset_dir=args.dataset,
             dest_dir=args.dest_dir)

    def run_cleanup(args):
        assert args.dir and args.top and (args.metric[0] in ['+', '-'])
        cleanup(experiments_dir=args.dir, top_to_keep=args.top, clean_rest=args.clean_rest,
                metric=args.metric[1:], maximize=(args.metric[0] == '+'))

    def run_eval(args):
        assert args.model_dir and (args.dataset_set or args.dataset_file)
        eval(model_dir=args.model_dir, dataset_set=args.dataset_set, dataset_dir=args.dataset,
             predict_only=args.predict_only, dest_sub_dir=args.sub_dir, dataset_file=args.dataset_file)

    parser = argparse.ArgumentParser(description='run train experiments, tune and eval')
    subparser = parser.add_subparsers()

    best_parser = subparser.add_parser('best', help='run experiments with best hyper-params')
    best_parser.set_defaults(func=run_best)
    best_parser.add_argument('--configs_dir', default="experiments/_best", help='directory for training json configurations. Default: "experiments/_best"')
    best_parser.add_argument('-s', '--serialization_dir', type=str, default="tmp", help='serialization root directory. Default: "tmp"')
    best_parser.add_argument('--dataset', type=str, help='dataset directory (Optional)')
    best_parser.add_argument('-d', '--dest_dir', type=str, default='exec/best', help='output directory for commands (input for  exec_multi.py). Default: "exec/best"')

    # train --experiment experiments/seq2seq/seq2seq.jsonnet -s dev_tmp -o lr:0.01 num_epochs:1
    train_parser = subparser.add_parser('train', help='train an experiment')
    train_parser.set_defaults(func=run_train)
    train_parser.add_argument('--experiment', type=str, help='jsonnet file to train')
    train_parser.add_argument('-s', '--serialization_dir', type=str, help='serialization root directory')
    train_parser.add_argument('-o', '--overrides', type=str, nargs='+',
                              help='settings params to override. format: <name>:<value>'
                                   'name may be nested with dots, e.g encoder.dropout:0.2')
    train_parser.add_argument('--dataset', type=str, help='dataset directory (Optional)')
    train_parser.add_argument('--no_eval', action='store_true', help='skip evaluation')
    train_parser.add_argument('--force', action='store_true', help='overwrite serialization directory')

    # tune --experiment experiments/seq2seq/seq2seq.jsonnet -s dev_tmp -d dev_tmp/tune.txt -o lr:"0.01 0.001" num_epochs:"1 2"
    tune_parser = subparser.add_parser('tune', help='tune hyper-params of an experiment')
    tune_parser.set_defaults(func=run_tune)
    tune_parser.add_argument('--experiment', type=str, help='jsonnet file to train')
    tune_parser.add_argument('-s', '--serialization_dir', type=str, help='serialization root directory')
    tune_parser.add_argument('-o', '--overrides', type=str, nargs='+',
                              help='settings params to override. format: <name>:"<value> <value>...". '
                                   'name may be nested with dots, e.g encoder.dropout:0.2')
    tune_parser.add_argument('--dataset', type=str, help='dataset directory (Optional)')
    tune_parser.add_argument('-d', '--dest_dir', type=str, default='exec/tune', help='output directory for commands (input for  exec_multi.py)')

    cleanup_parser = subparser.add_parser('cleanup', help='cleanup tuning experiments (keeps top k, in terms of best_validation_loss). '
                                                          'example: python scripts/run_experiments.py cleanup --dir tmp/tune --top 3 ')
    cleanup_parser.set_defaults(func=run_cleanup)
    cleanup_parser.add_argument('--dir', type=str, help='path to experiments (tune) dir')
    cleanup_parser.add_argument('--top', type=int, help='top experiments to keep')
    cleanup_parser.add_argument('--clean_rest', action='store_true', help='whether to clean directories with no metrics (default: false)')
    cleanup_parser.add_argument('--metric', default='-best_validation_loss', help='metric for comparison. start with +/- for max/min')

    eval_parser = subparser.add_parser('eval', help='evaluate model')
    eval_parser.set_defaults(func=run_eval)
    eval_parser.add_argument('--model_dir', type=str, help='model directory')
    eval_parser.add_argument('--dataset_set', type=str, default='dev', choices=['train', 'dev', 'test'],
                             help="dataset set to evaluate ('train', 'dev', 'test') Default: dev")
    eval_parser.add_argument('--dataset', type=str, required=False, help='dataset directory (Optional)')
    eval_parser.add_argument('--dataset_file', type=str, required=False, help='dataset file (Optional)')
    eval_parser.add_argument('--predict_only', default=False, action='store_true', help='skip evaluation, just predict')
    eval_parser.add_argument('--sub_dir', type=str, default='eval', help='destination sub dir (default: "eval")')

    args = parser.parse_args()
    args.func(args)
