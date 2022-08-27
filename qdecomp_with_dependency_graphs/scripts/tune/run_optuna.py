import argparse
import glob
import json
import logging
import socket
from datetime import datetime
import os
import importlib.util
import sys
from enum import Enum
from typing import Tuple, Any

import optuna
from optuna.study import StudyDirection
from optuna.trial import TrialState

from qdecomp_with_dependency_graphs.scripts.train.run_experiments import _remove_by_path
from qdecomp_with_dependency_graphs.scripts.tune.helpers import dump_best_config

_logger = logging.getLogger(__name__)
_MACHINE_NAME = socket.gethostname()
_CUDA = os.environ.get('CUDA_VISIBLE_DEVICES', '')


def optimize_study(study_file_path: str, output_root_dir: str, load_if_exists: bool,
                   timeout_in_hours: int = None, trials: int = None,
                   top_k_trials_to_keep: int = None):
    study_name = os.path.splitext(os.path.basename(study_file_path))[0]
    output_dir = os.path.join(output_root_dir, study_name)
    os.makedirs(output_dir, exist_ok=True)

    storage = _get_storage_path(output_dir)
    objective, direction = _get_objective(study_file_path=study_file_path,
                                          study_dir=output_dir,
                                          top_k_trials_to_keep=top_k_trials_to_keep)

    study = optuna.create_study(
        storage=storage,  # save results in DB
        sampler=optuna.samplers.TPESampler(),
        study_name=study_name,
        direction={StudyDirection.MAXIMIZE: 'maximize', StudyDirection.MINIMIZE: 'minimize'}[direction],
        pruner=optuna.pruners.HyperbandPruner(),
        load_if_exists=load_if_exists
    )

    timeout = timeout_in_hours and (timeout_in_hours * 60 * 60)  # timeout (sec)
    study.optimize(
        objective,
        n_jobs=1,  # number of processes in parallel execution
        n_trials=trials,  # number of trials to train a model
        timeout=timeout,  # threshold for executing time (sec)
        catch=(Exception,)  # skip failures
    )


def clean_running_trials(study_dir: str):
    """
    Sets RUNNING trails state to FAIL, to deal with not-finished trials when aborting optimization
    Make sure to run it when no optimization process is running
    """
    study = _load_study(study_dir)
    storage = study._storage
    for trial in study.get_trials(deepcopy=False):
        if trial.state == TrialState.RUNNING:
            # trial.set_user_attr(key='fail_reason', value='cleaned not-finished running trial')
            storage.set_trial_user_attr(trial_id=trial._trial_id, key='fail_reason', value='cleaned not-finished running trial')
            storage.set_trial_state(trial_id=trial._trial_id, state=TrialState.FAIL)


def clean_serialization_directory(study_dir: str, top_k: int = None, keep_metadata: bool = True):
    """
    Removes unnecessary trails serialization directories
    :param top_k - if not None, the top k trails only will be keeped
    :param keep_metadata - keep metadata files in all completed trials (with value)
    """
    study = _load_study(study_dir)
    study_serialization_dir = os.path.join(study_dir, 'serialization')
    assert os.path.exists(study_serialization_dir), f'could not find serialization directory {study_serialization_dir}'
    completed = []
    to_delete = []
    for trial in study.get_trials(deepcopy=False):
        trial_serialization_dir = os.path.join(study_serialization_dir, str(trial.number))
        if trial.state in [TrialState.FAIL, TrialState.PRUNED]:
            to_delete.append(trial)
        elif trial.state == TrialState.COMPLETE:
            # make sure training has been finished
            if trial.value is None or not os.path.exists(os.path.join(trial_serialization_dir, 'model.tar.gz')):
                # we might need best trial
                if study.best_trial.number != trial.number:
                    to_delete.append(trial)
            else:
                completed.append(trial)

    if top_k is not None:
        sorted_completed = sorted(completed, key=lambda x: x.value, reverse=(study.direction == StudyDirection.MAXIMIZE))
        to_delete.extend(sorted_completed[top_k:])

    # delete files / directories
    for trial in to_delete:
        trial_serialization_dir = os.path.join(study_serialization_dir, str(trial.number))
        if keep_metadata and trial.value is not None:
            files_list = [x for pattern in ['*.th', 'model.tar.gz'] for x in glob.glob(os.path.join(trial_serialization_dir,pattern))]
            for file_path in files_list:
                try:
                    _remove_by_path(file_path)
                except:
                    _logger.exception("Error while deleting file : ", file_path)
        else:
            _remove_by_path(trial_serialization_dir)



def study_status(study_dir: str):
    study = _load_study(study_dir)
    status_dir = os.path.join(study_dir, 'statuses', datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
    os.makedirs(status_dir, exist_ok=True)

    study.trials_dataframe().to_csv(os.path.join(status_dir, 'trials.csv'))

    try:
        opt_history = optuna.visualization.plot_optimization_history(study)
        opt_history.write_html(os.path.join(status_dir, 'optimization_history.html'))
        # opt_history.write_image(os.path.join(status_dir, 'optimization_history.png'))
    except Exception:
        _logger.exception('failed to plot optimization_history')

    try:
        params_imp = optuna.visualization.plot_param_importances(study)
        params_imp.write_html(os.path.join(status_dir, 'param_importances.html'))
        # params_imp.write_image(os.path.join(status_dir, 'param_importances.png'))
    except Exception:
        _logger.exception('failed to plot param_importances')

    try:
        parallel_coordinate = optuna.visualization.plot_parallel_coordinate(study)
        parallel_coordinate.write_html(os.path.join(status_dir, 'parallel_coordinate.html'))
        # parallel_coordinate.write_image(os.path.join(status_dir, 'parallel_coordinate.png'))
    except Exception:
        _logger.exception('failed to plot parallel_coordinate')

    try:
        slice = optuna.visualization.plot_slice(study)
        slice.write_html(os.path.join(status_dir, 'slice.html'))
        # slice.write_image(os.path.join(status_dir, 'slice.png'))
    except Exception:
        _logger.exception('failed to plot slice')

    try:
        with open(os.path.join(status_dir, 'best_trial.text'), 'wt') as fp:
            best = {
                'number': study.best_trial.number,
                'value': study.best_value,
                'params': study.best_params,
            }
            json.dump(best, fp, indent=2)
        _logger.info(f'best trial ({study.best_trial.number}): value {study.best_value}, params: {study.best_params}')
    except Exception:
        _logger.exception('failed to dump best_trial')

    try:
        config_file, *_, constants = _get_study_config(study_file_path=f'scripts/tune/studies/{study.study_name}.py')
        _set_constants(constants=constants)
        dump_best_config(config_file, os.path.join(status_dir, f'{study.study_name}.json'), study)
    except Exception:
        _logger.exception('failed to dump best configuration')


#########################
#      Utils            #
#########################

class AvoidDuplicatesStrategy(str, Enum):
    PRUNE = 'PRUNE'
    SKIP_OR_FAIL = 'SKIP_OR_FAIL'


def _set_constants(constants: Tuple[str, Any]):
    for arg, val in constants:
        os.environ[arg] = str(val)


def _get_study_config(study_file_path:str):
    spec = importlib.util.spec_from_file_location('module_name', study_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['module_name'] = module
    spec.loader.exec_module(module)
    config_file, metrics, direction = module.get_experiment()
    set_parameters = module.set_parameters
    constants = module.get_constants()
    return config_file, metrics, direction, set_parameters, constants


def _get_storage_path(study_dir: str) -> str:
    storage_path = os.path.join(study_dir, 'storage.db')
    return f"sqlite:///{storage_path}"


def _load_study(study_dir: str) -> optuna.Study:
    storage = _get_storage_path(study_dir=study_dir)
    study_name = os.path.basename(os.path.normpath(study_dir))
    study = optuna.load_study(
        storage=storage,
        study_name=study_name,
    )
    return study


def _get_objective(study_file_path:str, study_dir: str, top_k_trials_to_keep: int = None,
                   avoid_duplicates_strategy: AvoidDuplicatesStrategy = AvoidDuplicatesStrategy.PRUNE):
    """
    Wrap a tuning objective (experiment)
    :param study_file_path:
        file for the study (one of scrips/tune/studies)
    :param study_dir:
        study directory for outputs
    :param top_k_trials_to_keep:
        keep top k experiments according to study metric (delete all the rest directories)
        use this parameter to avoid disk space issues
    :param avoid_duplicates_strategy:
        if given, avoids training with same parameters as previous COMPLETED/RUNNING trail by:
            PRUNE - prune the current trial
            SKIP_OR_FAIL - if previous trail completed return its value, else raise a ValueError
    :return: objective function, direction
    """
    config_file, metrics, direction, set_parameters, constants = _get_study_config(study_file_path=study_file_path)
    serialization_dir = os.path.join(study_dir, 'serialization')
    os.makedirs(serialization_dir, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        # document host name and cuda
        trial.set_user_attr(key='machine', value=_MACHINE_NAME)
        trial.set_user_attr(key='cuda', value=_CUDA)
        trial.set_user_attr(key='pid', value=os.getpid())

        # cleanup less successful trials
        # clean_serialization_directory(study_dir=study_dir, top_k=top_k_trials_to_keep, keep_metadata=True)

        # constants
        _set_constants(constants=constants)

        # hyper parameters parameters
        set_parameters(trial)

        # Check duplication and skip if it's detected
        if avoid_duplicates_strategy:
            candidates = [x for x in trial.study.get_trials(deepcopy=False) if x.number != trial.number and
                          x.state in [TrialState.COMPLETE, TrialState.RUNNING]]
            for t in sorted(candidates, key=lambda x: str(x.state)):  # complete first
                if t.params != trial.params:
                    continue
                if avoid_duplicates_strategy == AvoidDuplicatesStrategy.PRUNE:
                    trial.set_user_attr('prune_reason', 'Duplicate parameter set')
                    raise optuna.exceptions.TrialPruned('Duplicate parameter set')
                if avoid_duplicates_strategy == AvoidDuplicatesStrategy.SKIP_OR_FAIL:
                    if t.state == TrialState.COMPLETE: return t.value
                    raise ValueError('Duplicate parameter set')

        executor = optuna.integration.allennlp.AllenNLPExecutor(
            trial=trial,  # trial object
            config_file=config_file,  # path to jsonnet
            serialization_dir=os.path.join(serialization_dir, str(trial.number)),
            metrics=metrics,
            include_package="qdecomp_nlp"
        )
        return executor.run()

    return objective, direction


if __name__ == '__main__':
    """
    python scripts/tune/run_optuna.py optimize --study scripts/tune/studies/biaffine-graph-parser--transformer-encoder.py --load_if_exists --top 3
    Note: parallel trails might share parameters
    """
    def run_optimize_study(args):
        assert args.study and args.output_dir
        optimize_study(study_file_path=args.study, output_root_dir=args.output_dir,
                       timeout_in_hours=args.timeout, trials=args.trials,
                       load_if_exists=args.load_if_exists,
                       top_k_trials_to_keep=args.top)

    def run_clean_running_trails(args):
        assert args.study_dir
        clean_running_trials(study_dir=args.study_dir)

    def run_clean_trails(args):
        assert args.study_dir and args.top
        clean_serialization_directory(study_dir=args.study_dir, top_k=args.top, keep_metadata=args.keep_metadata)

    """
    python scripts/tune/run_optuna.py status --study_dir tmp/tune_optuna/biaffine-graph-parser--transformer-encoder/
    """
    def run_study_status(args):
        assert args.study_dir
        study_status(study_dir=args.study_dir)

    parser = argparse.ArgumentParser(description='run optuna study')
    subparser = parser.add_subparsers()

    optimize_parser = subparser.add_parser('optimize', help='optimize a study. Note: parallel trails might share parameters. We avoid duplicates by pruning')
    optimize_parser.set_defaults(func=run_optimize_study)
    optimize_parser.add_argument('--study', type=str, help='a study to run (file of scripts/tune/studies)')
    optimize_parser.add_argument('-o', '--output_dir', default='tmp/tune_optuna', help='output root directory')
    optimize_parser.add_argument('--timeout', type=int, required=False, help='timeout in hours')
    optimize_parser.add_argument('--trials', type=int, required=False, help='limit number of trials')
    optimize_parser.add_argument('--load_if_exists', action="store_true", help='continue previous study')
    optimize_parser.add_argument('--top', type=int, required=False, help='top k trials to keep')

    clean_running_parser = subparser.add_parser('set-running', help='clean not finished running trails (set to FAIL)')
    clean_running_parser.set_defaults(func=run_clean_running_trails)
    clean_running_parser.add_argument('--study_dir', type=str, help='the study dir')

    clean_parser = subparser.add_parser('clean', help='clean serialization directory from unnecessary trials')
    clean_parser.set_defaults(func=run_clean_trails)
    clean_parser.add_argument('--study_dir', type=str, help='the study dir')
    clean_parser.add_argument('--top', default=3, type=int, help='top k trials to keep')
    clean_parser.add_argument('--keep_metadata', action="store_true", help='keep metadata files in all completed trials')

    status_parser = subparser.add_parser('status', help='optimize a study')
    status_parser.set_defaults(func=run_study_status)
    status_parser.add_argument('--study_dir', type=str, help='the study dir')

    args = parser.parse_args()
    args.func(args)