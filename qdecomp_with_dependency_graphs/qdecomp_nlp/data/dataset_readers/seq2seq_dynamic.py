"""
Based on Seq2SeqDatasetReader, allennlp_models/generation/dataset_readers/seq2seq.py
tag: v1.1.0
"""
import csv
from typing import Dict, Optional
import logging
import copy

from ast import literal_eval
from overrides import overrides
from typing import List

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, NamespaceSwappingField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from qdecomp_nlp.data.dataset_readers.util import read_break_data

logger = logging.getLogger(__name__)


@DatasetReader.register("break_seq2seq")
class Seq2SeqDynamicDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    `ComposedSeq2Seq` model, or any model with a matching API.
    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>
    The output of `read` is a list of `Instance` s with the fields:
        source_tokens : `TextField` and
        target_tokens : `TextField`
    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.
    # Parameters
    source_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to `SpacyTokenizer()`.
    target_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to `source_tokenizer`.
    source_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define input (source side) token representations. Defaults to
        `{"tokens": SingleIdTokenIndexer()}`.
    target_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define output (target side) token representations. Defaults to
        `source_token_indexers`.
    source_add_start_token : `bool`, (optional, default=`True`)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    source_add_end_token : `bool`, (optional, default=`True`)
        Whether or not to add `END_SYMBOL` to the end of the source sequence.
    target_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the target sequence.
    target_add_end_token : bool, (optional, default=True)
        Whether or not to add `END_SYMBOL` to the end of the target sequence.
    delimiter : str, (optional, default="\t")
        Set delimiter for tsv/csv file.
    """

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_add_start_token: bool = True,
        source_add_end_token: bool = True,
        target_add_start_token: bool = True,
        target_add_end_token: bool = True,
        start_symbol: str = START_SYMBOL,
        end_symbol: str = END_SYMBOL,
        delimiter: str = ",",
        source_max_tokens: Optional[int] = None,
        target_max_tokens: Optional[int] = None,
        quoting: int = csv.QUOTE_MINIMAL,
        separator_symbol: str = "@@SEP@@",
        dynamic_vocab: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._source_add_end_token = source_add_end_token
        self._target_add_start_token = target_add_start_token
        self._target_add_end_token = target_add_end_token
        # doesnt work for some tokenizers, like bert
        # self._start_token, self._end_token = self._source_tokenizer.tokenize(
        #     start_symbol + " " + end_symbol
        # )
        self._start_token, self._end_token = Token(start_symbol), Token(end_symbol)
        self._delimiter = delimiter
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self._quoting = quoting

        self._dynamic_vocab = dynamic_vocab
        self._allowed_tokenizer = self._source_tokenizer
        self._separator_symbol = separator_symbol
        # todo: should explicitly check that the source and target namespaces are the same,
        #  or support different namespaces.

    @overrides
    def _read(self, file_path: str):
        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0

        logger.info("Reading instances from lines in file at: %s", file_path)
        args = ['question_text', 'decomposition', 'lexicon_tokens'] if self._dynamic_vocab else ['question_text', 'decomposition']
        for instance in read_break_data(file_path, self._delimiter, self.text_to_instance, args, quoting=self._quoting):
            yield instance

        if self._source_max_tokens and self._source_max_exceeded:
            logger.info(
                "In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
                self._source_max_exceeded,
                self._source_max_tokens,
            )
        if self._target_max_tokens and self._target_max_exceeded:
            logger.info(
                "In %d instances, the target token length exceeded the max limit (%d) and were truncated.",
                self._target_max_exceeded,
                self._target_max_tokens,
            )

    @overrides
    def text_to_instance(
        self, source_string: str, target_string: str = None, allowed_string: str = None) -> Instance:
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
            self._source_max_exceeded += 1
            tokenized_source = tokenized_source[: self._source_max_tokens]
        if self._source_add_start_token:
            tokenized_source.insert(0, copy.deepcopy(self._start_token))
        if self._source_add_end_token:
            tokenized_source.append(copy.deepcopy(self._end_token))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        meta_fields = {"source_tokens": [x.text for x in tokenized_source]}
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            meta_fields["target_tokens"] = [y.text for y in tokenized_target]
            if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                self._target_max_exceeded += 1
                tokenized_target = tokenized_target[: self._target_max_tokens]
            if self._target_add_start_token:
                tokenized_target.insert(0, copy.deepcopy(self._start_token))
            if self._target_add_end_token:
                tokenized_target.append(copy.deepcopy(self._end_token))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            fields_dict = {"source_tokens": source_field, "target_tokens": target_field}
        else:
            tokenized_target = None
            fields_dict = {"source_tokens": source_field}

        # allowed tokens
        if allowed_string:
            source_start = 1 if self._source_add_start_token else 0
            source_end = -1 if self._source_add_end_token else None
            source_tokens_text = [x.text for x in tokenized_source[source_start:source_end]]
            target_tokens_text = [y.text for y in tokenized_target[1:-1]] if tokenized_target else []
            parsed_allowed_string = self._parse_allowed_string(allowed_string, source_tokens_text, target_tokens_text)
            tokenized_allowed = self._allowed_tokenizer.tokenize(parsed_allowed_string)
            tokenized_allowed.insert(0, Token(self._separator_symbol))
            tokenized_allowed.insert(0, copy.deepcopy(self._end_token))
            tokenized_allowed.insert(0, copy.deepcopy(self._start_token))
            allowed_field = TextField(tokenized_allowed, self._target_token_indexers)
            fields_dict["allowed_tokens"] = allowed_field
            fields_dict["allowed_token_ids"] = NamespaceSwappingField(tokenized_allowed, "tokens")
            allowed_tokens_text = [x.text for x in tokenized_allowed]

            # Sanity check
            for token in target_tokens_text:
                if token not in allowed_tokens_text:
                    logger.warning(f"Missing token in allowed tokens: {token}")

        fields_dict["metadata"] = MetadataField(meta_fields)
        return Instance(fields_dict)

    @staticmethod
    def _parse_allowed_string(allowed_string: str,
                              source_tokens_text: List[str],
                              target_tokens_text: List[str]) -> str:
        allowed_tokens = [t.strip() for t in literal_eval(allowed_string) if type(t) == str]
        allowed_tokens = list(set(
            ' '.join(allowed_tokens).split(' ') + source_tokens_text + target_tokens_text
        ))
        return ' '.join(allowed_tokens)