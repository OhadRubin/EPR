from typing import Iterable, List
import os
from dataclasses import dataclass
from ast import literal_eval

from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.spans_dependencies_to_logical_form_tokens import SpansDepToQDMRStepTokensConverter
from qdecomp_with_dependency_graphs.dependencies_graph.extractors import BaseTokensDependenciesExtractor, BaseSpansExtractor, \
    BaseTokensDependenciesToQDMRExtractor,  BaseCollapser


@dataclass
class Configuration:
    spans_extractor: BaseSpansExtractor = None
    tokens_dependencies_extractor: BaseTokensDependenciesExtractor = None
    tokens_dependencies_to_qdmr_extractor: BaseTokensDependenciesToQDMRExtractor = None
    spans_dependencies_to_logical_form_converter: SpansDepToQDMRStepTokensConverter = None


config: Configuration = Configuration()
_config_str = None
def load(config_file: str):
    global config, _config_str
    with open(config_file, 'rt') as fp:
        _config_str = fp.read()
    _locals = {}
    exec(_config_str, globals(), _locals)
    for attr, value in config.__dict__.items():
        config.__setattr__(attr, _locals[attr])

conf = os.environ.get('DEP_CONF', 'default')
load(f'dependencies_graph/config/config_{conf}.py')


def save(dir_path: str):
    path = os.path.join(dir_path, 'dependencies_graph_config.py')
    with open(path, 'wt') as fp:
        fp.write(_config_str)
