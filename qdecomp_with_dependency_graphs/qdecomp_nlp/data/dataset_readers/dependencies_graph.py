"""
Based on allennlp-models/allennlp_models/syntax/biaffine_dependency_parser/universal_dependencies_reader.py
tag: v1.0.0rc3
"""
from typing import Any, Dict, Tuple, List, Iterable
import logging

from overrides import overrides
import json

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, AdjacencyField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

from qdecomp_nlp.data.fields.adjacency_field import CustomAdjacencyField
from qdecomp_nlp.data.fields.multilabel_adjacency_field import MultiLabelAdjacencyField
from qdecomp_with_dependency_graphs.utils.data_structures import list_to_multivalue_dict

logger = logging.getLogger(__name__)


@DatasetReader.register("dependencies_graph")
class DependenciesGraphDatasetReader(DatasetReader):
    """
    Reads a dependencies graph file in the conllu Universal Dependencies like format.
    Ignores the 'head' and 'deprel' fields, and uses the 'deps' field to represent the graph edges.
    Uses comments for key-value metadata (e.g # sample_id = 1234)

    # Parameters
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : `bool`, optional (default = False)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    tokenizer : `Tokenizer`, optional, default = None
        A tokenizer to use to split the text. This is useful when the tokens that you pass
        into the model need to have some particular attribute. Typically it is not necessary.
    deps_tags_namespace : str , optional , (default = "arcs_tags")
        Vocabulary namespace for dependencies tags
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        use_language_specific_pos: bool = False,
        tokenizer: Tokenizer = None,
        word_field: str = "text",
        pos_field: str = None,
        bio_field: str = None,
        pos_tags_namespace: str = "pos_tags",
        bio_tags_namespace: str = "bio_tags",
        deps_tags_namespace: str = "deps_tags",
        fill_none_tags: bool = False,
        multi_label: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.use_language_specific_pos = use_language_specific_pos
        self.tokenizer = tokenizer
        self.pos_tags_namespace = pos_tags_namespace
        self.bio_tags_namespace = bio_tags_namespace
        self.deps_tags_namespace = deps_tags_namespace
        self.pos_field = pos_field
        self.bio_field = bio_field
        self.word_field = word_field
        self.fill_none_tags = fill_none_tags
        self.multi_label = multi_label

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, "r") as f:
            logger.info("Reading instances from dataset at: %s", file_path)

            for line in f.readlines():
                line = json.loads(line)
                tokens = line['tokens']
                if "extra_tokens" in line:
                    tokens.extend(line['extra_tokens'])
                words = [x[self.word_field] for x in tokens]
                pos_tags = self.pos_field and [x[self.pos_field] for x in tokens]
                bio_tags = self.bio_field and [x[self.bio_field] for x in tokens]
                deps = line.get('deps')

                yield self.text_to_instance(words, pos_tags, bio_tags, deps, line["metadata"])

    @overrides
    def text_to_instance(
        self,  # type: ignore
        tokens: List[str],
        upos_tags: List[str] = None,
        bio_tags: List[str] = None,
        dependencies: List[Tuple[int, int, str]] = None,
        metadata: Dict[str, Any] = None
    ) -> Instance:

        """
        # Parameters
        tokens : `List[str]`, required.
            The tokens in the sentence to be encoded.
        upos_tags : `List[str]`, required.
            The universal dependencies POS tags for each word.
        dependencies : `List[Tuple[str, int]]`, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.
        # Returns
        An instance containing words, upos tags, dependency head tags and head
        indices as fields.
        """
        fields: Dict[str, Field] = {}

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(" ".join(tokens))
        else:
            tokens = [Token(t) for t in tokens]

        text_field = TextField(tokens, self._token_indexers)
        fields["tokens"] = text_field
        if upos_tags:
            fields["pos_tags"] = SequenceLabelField(upos_tags, text_field, label_namespace=self.pos_tags_namespace)
        if self.bio_field:
            fields["bio_tags"] = SequenceLabelField(bio_tags, text_field, label_namespace=self.bio_tags_namespace)
        if dependencies is not None:
            # merge dependencies if needed
            dep_map = {}
            for u, v, dep in dependencies:
                dep_map[(u, v)] = dep_map.get((u, v), []) + [dep]
            if self.multi_label:
                # merge duplicates only
                dependencies = [(u, v, d) for (u, v), deps in dep_map.items()
                                for d in sorted('&'.join(x)
                                                for x in list_to_multivalue_dict(deps, key=lambda x:x).values())]
            else:
                dependencies = [(u, v, '&'.join(sorted(deps))) for (u, v), deps in dep_map.items()]

            indices, labels = zip(*[((u, v), dep) for u, v, dep in dependencies]) if dependencies else ([],[])

            # if self.fill_none_tags:
            #     indices, labels = self._fill_empty_tags(indices, labels, len(tokens))
            fields["arc_indices"] = CustomAdjacencyField(sequence_field=text_field, indices=list(set(indices)),
                                                         fill_with_none=self.fill_none_tags)
            if self.multi_label:
                fields["arc_tags"] = MultiLabelAdjacencyField(sequence_field=text_field, indices=indices,
                                                              labels=labels, label_namespace=self.deps_tags_namespace)
            else:
                fields["arc_tags"] = CustomAdjacencyField(sequence_field=text_field, indices=indices, labels=labels,
                                                          label_namespace=self.deps_tags_namespace,
                                                          fill_with_none=self.fill_none_tags)

        fields["metadata"] = MetadataField({"tokens": tokens, "pos": upos_tags, **(metadata or {})})
        return Instance(fields)

    # @staticmethod
    # def _fill_empty_tags(indicies: Iterable[Tuple[int, int]], labels: Iterable[str], tokens_count:int,
    #                      fill_tag: str = "NONE"):
    #     indicies = list(indicies)
    #     labels = list(labels)
    #     indicies_ = set(indicies)
    #     for i in range(tokens_count):
    #         for j in range(tokens_count):
    #             if (i, j) not in indicies_:
    #                 indicies.append((i, j))
    #                 labels.append(fill_tag)
    #     return indicies, labels
