import optuna
import os

from optuna.study import StudyDirection


def get_experiment():
    config_file = "scripts/tune/experiments/operators-aware-biaffine-graph-parser--transformer-encoder.jsonnet"
    metrics = "best_validation_logical_form_em"
    direction = StudyDirection.MAXIMIZE
    return config_file, metrics, direction


def get_constants():
    return [
        ("transformer_model", "bert-base-uncased"),
        ("max_length", 128),
        ("transformer_dim", 768),

        ("decode_strategy", "operators_mask"),

        ("pos_embedding_dim", 100),
    ]


def set_parameters(trial: optuna.Trial):
    # hyper parameters
    trial.suggest_float("input_dropout", 0.0, 0.8, step=0.1)
    trial.suggest_float("dropout", 0.0, 0.8, step=0.1)
    trial.suggest_int("operator_representation_dim", 100, 700, step=100)
    trial.suggest_int("tag_representation_dim", 100, 700, step=100)
    trial.suggest_int("operator_ff_num_layers", 1, 3)
    trial.suggest_int("tag_ff_num_layers", 1, 3)
    trial.suggest_int("operator_embeddings_dim", 0, 300, step=100)

    trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2, 1e-1])
    # trial.suggest_categorical("transformer_lr", [2e-5, 3e-5, 5e-5])
    trial.suggest_categorical("transformer_lr", [2e-5, 3e-5, 5e-5, 7e-5])
    trial.suggest_categorical("seed", [24, 42, 64])
