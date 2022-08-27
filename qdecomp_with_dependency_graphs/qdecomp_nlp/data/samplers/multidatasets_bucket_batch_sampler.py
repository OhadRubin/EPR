"""
Based on allennlp/data/samplers/bucket_batch_sampler.py
tag: v1.1.0
"""

import logging
from typing import List, Iterable, Tuple, Sequence, Dict
import random
import math

from overrides import overrides

from torch.utils import data

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.samplers import BatchSampler, BucketBatchSampler

from qdecomp_with_dependency_graphs.utils.data_structures import list_to_multivalue_dict

logger = logging.getLogger(__name__)


@BatchSampler.register("multidatasets_bucket")
class MultidatasetsBucketBatchSampler(BucketBatchSampler):
    """
    n sampler which acts like `BucketBatchSampler`, but deal with heterogeneous instances
    that where created by multiple DatasetReaders (e.g by InterleavingDatasetReader)

    # Parameters
    dataset_field_name: str (optional, default="dataset")
        A MetadataField name that stores the dataset name. Each batch will have instances of a single dataset.
    sorting_keys_map: Dict[str, List[str]] (optional, default=None)
        sorting_keys list for each dataset
    """

    def __init__(
            self,
            sorting_keys_map: Dict[str, List[str]] = None,
            dataset_field_name: str = "dataset",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.sorting_keys_map = sorting_keys_map
        self.data_source_map: Dict[str, Iterable[Instance]] = None
        self.dataset_field_name = dataset_field_name

    def _construct_data_sources_map(self, instances: Sequence[Instance]):
        if self.data_source_map is not None:
            return
        self.data_source_map = list_to_multivalue_dict(
            enumerate(instances), key=lambda x: x[1][self.dataset_field_name].metadata
        )

    @overrides
    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        self._construct_data_sources_map(instances)

        batches = []
        for dataset, data_source_items in self.data_source_map.items():
            original_indices, data_source = list(zip(*data_source_items))
            self.sorting_keys = self.sorting_keys_map and self.sorting_keys_map.get(dataset, None)
            indices, _ = self._argsort_by_padding(data_source)
            for group in lazy_groups_of(indices, self.batch_size):
                batch_indices = list(group)
                if self.drop_last and len(batch_indices) < self.batch_size:
                    continue
                batches.append([original_indices[i] for i in batch_indices])

        random.shuffle(batches)
        for batch in batches:
            yield batch

    @overrides
    def get_num_batches(self, instances: Sequence[Instance]) -> int:
        self._construct_data_sources_map(instances)

        def data_source_batches(data_source_len: int):
            batch_count_float = data_source_len / self.batch_size
            if self.drop_last:
                return math.floor(batch_count_float)
            else:
                return math.ceil(batch_count_float)

        return sum(data_source_batches(len(x)) for x in self.data_source_map.values())