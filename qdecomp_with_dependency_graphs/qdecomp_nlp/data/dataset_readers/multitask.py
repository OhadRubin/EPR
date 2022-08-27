import json
import logging
from os import PathLike
from typing import Dict, Iterator, Union

from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)


@DatasetReader.register("custom_multitask")
class CustomMultiTaskDatasetReader(DatasetReader):
    """
    This `DatasetReader` simply collects a dictionary of other `DatasetReaders`.  It is designed for
    a different class (the `MultiTaskDataLoader`) to actually read from each of the underlying
    dataset readers.
    Use `read()` only for predictions.
    Registered as a `DatasetReader` with name "custom_multitask".
    # Parameters
    readers : `Dict[str, DatasetReader]`
        A mapping from dataset name to `DatasetReader` objects for reading that dataset.  You can
        use whatever names you want for the datasets, but they have to match the keys you use for
        data files and in other places in the `MultiTaskDataLoader` and `MultiTaskScheduler`.
    """

    def __init__(self, readers: Dict[str, DatasetReader]) -> None:
        self.readers = readers

    def read(self, file_paths: Dict[str, Union[PathLike, str]]) -> Dict[str, Iterator[Instance]]:  # type: ignore
        logger.warning("This class is not designed to be called like this. Call only on predict")
        if not isinstance(file_paths, dict):
            file_paths = json.loads(file_paths)
        for task, path in file_paths.items():
            for x in self.readers[task].read(path):
                x.add_field('task', MetadataField(task))
                yield x