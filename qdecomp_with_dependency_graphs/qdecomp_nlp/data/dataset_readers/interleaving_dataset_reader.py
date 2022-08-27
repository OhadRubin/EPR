"""
Based on allennlp/data/dataset_readers/interleaving_dataset_reader.py
tag: v1.1.0
"""

from typing import Dict, Mapping, Iterable
import json
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers import InterleavingDatasetReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)

@DatasetReader.register("custom_interleaving")
class CustomInterleavingDatasetReader(InterleavingDatasetReader):
    """
    As InterleavingDatasetReader, but allowing "missing" keys in file_path (useful for separate prediction per model)
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def _read(self, file_path: str) -> Iterable[Instance]:
        if not isinstance(file_path, dict):
            raise ConfigurationError(
                "the file_path for the InterleavingDatasetReader "
                "needs to be a dictionary {reader_name: file_path}"
            )

        if file_path.keys() != self._readers.keys():
            # raise ConfigurationError("mismatched keys")
            logger.warning(f"mismatched keys, taking the common. files:{list(file_path.keys())}, readers: {list(self._readers.keys())}")

        # Load datasets
        datasets = {key: reader.read(file_path[key]) for key, reader in self._readers.items() if key in file_path}

        if self._scheme == "round_robin":
            yield from self._read_round_robin(datasets)
        elif self._scheme == "all_at_once":
            yield from self._read_all_at_once(datasets)
        else:
            raise RuntimeError("impossible to get here")
