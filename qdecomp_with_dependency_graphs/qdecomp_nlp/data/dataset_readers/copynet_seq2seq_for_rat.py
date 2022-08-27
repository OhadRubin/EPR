"""
based on allennlp_models/generation/dataset_readers/copynet_seq2seq.py
tag: v1.1.0
"""
import json
import logging
import re
from typing import List, Dict, Tuple
import warnings

import numpy as np

from ast import literal_eval
from overrides import overrides

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField, AdjacencyField, \
    SequenceField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import (
    Token,
    Tokenizer,
    SpacyTokenizer,
    PretrainedTransformerTokenizer,
    WhitespaceTokenizer,
)
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from qdecomp_nlp.data.dataset_readers.util import read_break_data
from qdecomp_nlp.data.dataset_readers.copynet_seq2seq_dynamic import CopyNetDynamicDatasetReader
from qdecomp_with_dependency_graphs.utils.data_structures import list_to_multivalue_dict

logger = logging.getLogger(__name__)


@DatasetReader.register("break_copynet_seq2seq_rat")
class CopyNetRATDatasetReader(CopyNetDynamicDatasetReader):
    """
    Add dependencies representation to the instances.
    In case of decompose_dependencies=True, separate the dependencies to different representation
    for operator, arg, properties as ListField
    """
    def __init__(self,
                 dependencies_namespace: str = 'relations_tags',
                 decompose_dependencies: bool = False,
                 **kwargs,
                 ) -> None:
        super().__init__(**kwargs)
        self._dependencies_namespace = dependencies_namespace
        self._decompose_dependencies = decompose_dependencies
        self._source_tokenizer = WhitespaceTokenizer()

    @overrides
    def _read(self, file_path):
        logger.info("Reading instances from lines in file at: %s", file_path)
        args = ['question_tokens', 'dependencies', 'decomposition']
        if self._dynamic_vocab:
            args.append('lexicon_tokens')
        for instance in read_break_data(file_path, self._delimiter, self.text_to_instance, args):
            yield instance

    @overrides
    def text_to_instance(self, source_tokens_str: str, dependencies_str: str, target_string: str = None, allowed_string: str = None) -> Instance:
        source_tokens: List[str] = json.loads(source_tokens_str)
        dependencies: List[Tuple[int, int, str]] = json.loads(dependencies_str)
        instance = super().text_to_instance(
            source_string=' '.join(source_tokens),
            target_string=target_string,
            allowed_string=allowed_string
        )

        sequence_field = instance['source_tokens']

        if self._decompose_dependencies:
            dependencies_ = [(u, v, get_dependency_parts(d)) for u, v, d in dependencies]
            adj_fields = []
            for k in ['operator', 'arg', 'properties']:
                adj_fields.append(
                    get_dependencies_adjacency_field(
                        dependencies=[(u, v, d[k]) for u, v, d in dependencies_ if d.get(k) is not None],
                        sequence_field=sequence_field,
                        namespace=f'{k}_{self._dependencies_namespace}'
                    )
                )
            instance.add_field('relations', ListField(adj_fields))
            return instance

        relations = get_dependencies_adjacency_field(
            dependencies=dependencies,
            sequence_field=instance['source_tokens'],
            namespace=self._dependencies_namespace
        )
        instance.add_field('relations', relations)
        return instance


def get_dependencies_adjacency_field(
        dependencies: List[Tuple[int, int, str]],
        sequence_field: SequenceField,
        namespace: str,
) -> AdjacencyField:
    # merge duplicates
    dep_map = list_to_multivalue_dict(dependencies, key=lambda x: (x[0], x[1]))
    dependencies = [(u, v, '&'.join(sorted(x[2] for x in deps))) for (u, v), deps in dep_map.items()]

    indices, labels = zip(*[((u, v), dep) for u, v, dep in dependencies]) if dependencies else ([], [])
    adj_field = AdjacencyField(
        sequence_field=sequence_field,
        indices=indices, labels=labels, label_namespace=namespace)
    return adj_field


def get_dependency_parts(dep: str) -> Dict[str, str]:
    regex = r'(\w+)(?:-(\w+))?(?:\[(.*)\])?'
    groups = list(re.match(regex, dep).groups())
    return {'operator': groups[0], 'arg': groups[1], 'properties': groups[2]}
