import json
import os

import _jsonnet
import optuna
from optuna import _imports


"""
Based on from optuna.integration.allennlp.dump_best_config()
"""
def dump_best_config(input_config_file: str, output_config_file: str, study: optuna.Study) -> None:
    """Save JSON config file after updating with parameters from the best trial in the study.

    Args:
        input_config_file:
            Input Jsonnet config file used with
            :class:`~optuna.integration.AllenNLPExecutor`.
        output_config_file:
            Output JSON config file.
        study:
            Instance of :class:`~optuna.study.Study`.
            Note that :func:`~optuna.study.Study.optimize` must have been called.

    """
    # _imports.check()

    best_params = {**os.environ, **study.best_params}
    for key, value in best_params.items():
        best_params[key] = str(value)
    best_config = json.loads(_jsonnet.evaluate_file(input_config_file, ext_vars=best_params))

    # `optuna_pruner` only works with Optuna.
    # It removes when dumping configuration since
    # the result of `dump_best_config` can be passed to
    # `allennlp train`.
    if "epoch_callbacks" in best_config["trainer"]:
        new_epoch_callbacks = []
        epoch_callbacks = best_config["trainer"]["epoch_callbacks"]
        for callback in epoch_callbacks:
            if callback["type"] == "optuna_pruner":
                continue
            new_epoch_callbacks.append(callback)

        if len(new_epoch_callbacks) == 0:
            best_config["trainer"].pop("epoch_callbacks")
        else:
            best_config["trainer"]["epoch_callbacks"] = new_epoch_callbacks

    with open(output_config_file, "w") as f:
        json.dump(best_config, f, indent=4)