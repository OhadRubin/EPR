import optuna
import os

from optuna.study import StudyDirection


def get_experiment():
    config_file = "scripts/tune/experiments/biaffine-graph-parser--transformer-encoder.jsonnet"
    metrics = "best_validation_logical_form_em"
    direction = StudyDirection.MAXIMIZE
    return config_file, metrics, direction


def get_constants():
    return [
        ("transformer_model", "bert-base-uncased"),
        ("max_length", 128),
        ("transformer_dim", 768),

        ("arc_tags_only", "false"),
        ("multi_label", "false"),

        ("pos_embedding_dim", 100),
        ("tag_representation_dim", 100)
    ]


def set_parameters(trial: optuna.Trial):
    # hyper parameters
    trial.suggest_float("input_dropout", 0.0, 0.8, step=0.1)
    trial.suggest_float("dropout", 0.0, 0.8, step=0.1)
    trial.suggest_int("arc_representation_dim", 300, 700, step=100)
    trial.suggest_int("arc_num_layers", 1, 3)
    trial.suggest_int("tag_num_layers", 1, 3)

    trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2, 1e-1])
    # trial.suggest_categorical("transformer_lr", [2e-5, 3e-5, 5e-5])
    trial.suggest_categorical("transformer_lr", [3e-5, 5e-5, 7e-5])
    trial.suggest_categorical("seed", [24, 42, 64])
