from typing import Any, Callable, List
import csv

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.fields import TextField, MetadataField, NamespaceSwappingField
from allennlp.data.instance import Instance


def read_break_data(file_path: str, delimiter: str,
                    text_to_instance: Callable[..., Instance],
                    args_columns: List[str],
                    metadata_columns: List[str] = ['question_id'],
                    quoting: int = csv.QUOTE_MINIMAL):
    with open(cached_path(file_path), "r") as data_file:
        lines = csv.reader(data_file, delimiter=delimiter, quoting=quoting)
        header = next(lines, None)
        header_to_index = {x: i for i, x in enumerate(header)}
        for line_num, row in enumerate(lines):
            if len(row) != len(header):
                raise ConfigurationError(
                    "Invalid line format: %s (line number %d)" % (row, line_num + 1)
                )

            instance = text_to_instance(*[row[header_to_index[x]] for x in args_columns if x in header_to_index])
            metadata = {x: row[header_to_index[x]] for x in metadata_columns}
            if 'metadata' in instance:
                metadata.update(instance['metadata'])
            instance.add_field('metadata', MetadataField(metadata))
            yield instance
