import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedModel
from transformers import BertTokenizer, BertTokenizerFast
from transformers import BatchEncoding, PreTrainedTokenizerBase
# from src.dataset_readers.qdmr_indexer import QDMRIndexerDatasetReader
from transformers.data.data_collator import DataCollatorWithPadding

class ListWrapper:
    def __init__(self, data: List[Any]):
        self.data = data
    
    def to(self, device):
        return self.data


@dataclass
class DataCollatorWithPaddingAndCuda:

    tokenizer: PreTrainedTokenizerBase
    device: object = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 3000
    pad_to_multiple_of: Optional[int] = None
    

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        metadata = [x.pop("metadata") for x in features]
        has_labels = "labels" in features[0]
        if has_labels:
            labels = [{"input_ids":x.pop("labels")} for x in features]
            labels = self.tokenizer.pad(
                labels,
                padding=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
            )
            
        
        # print(features)
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            
            return_tensors="pt",
        )
        
        batch['metadata'] = ListWrapper(metadata)
        if has_labels:
            batch['labels'] = labels.input_ids
        
        if self.device:
            batch = batch.to(self.device)
        
        
        return batch
