from datasets import load_dataset
from typing import Any, Dict, Iterable
from transformers import AutoTokenizer
import torch
import pandas as pd
import numpy as np
import json
import random
import pandas as pd
from datasets import Dataset
import json
from collections import defaultdict
import re
from src.dataset_readers.scorer_tasks import ScorerTask


class ScorerDatasetReader(torch.utils.data.Dataset):

    def __init__(self, example_file, model_name,task_name,setup_type,ds_size=None,**kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = "<|endoftext|>"
        self.tokenizer.padding_side = "left"
        self.task = ScorerTask.from_name(task_name)(example_file, ds_size)
        self.kwargs = kwargs
        self.setup_type = setup_type
        assert self.setup_type in ["qa","q"]

        def get_instance(entry):
            
            # examples = entry.pop("examples") if "examples" in entry else entry.pop('near_examples')
            examples = entry.pop("ctxs")
            for exp in examples:
                # print(exp)
                # print(self.task.training_dataset[exp['id']])
                exp.update(self.task.training_dataset[exp['id']][1])
                for key,val in entry.items():
                    exp[f"test_{key}"] = val
            yield from examples
                    
        def get_dataset(data):
            for entry in data:
                yield from get_instance(entry)
                
        df = pd.DataFrame(list(get_dataset(self.task.data)))
        self.dataset = Dataset.from_pandas(df)

    def shard(self,accelerator):
        self.dataset = self.dataset.shard(num_shards=accelerator.num_processes,index=accelerator.process_index)
        
        
    def __getitem__(self, index):
        return self.text_to_instance(self.dataset[index],index=index)

    def __len__(self):
        return len(self.dataset)



    def text_to_instance(self, entry: Dict[str, Any],index=-1):
        question,answer,test_question,test_answer = self.task.get_fields(entry)
        if self.setup_type=="qa":
            enc_text = f"{question}\t{answer}\n{test_question}\t{test_answer}"
            tokenized_example = self.tokenizer.encode_plus(enc_text,truncation=False,add_special_tokens=False,return_tensors='pt')
            tokenized_labels = self.tokenizer.encode_plus(test_answer,truncation=False,add_special_tokens=False,return_tensors='pt')
        elif self.setup_type=="q":
            enc_text = f"{question}\t{test_question}"
            tokenized_example = self.tokenizer.encode_plus(enc_text,truncation=False,add_special_tokens=False,return_tensors='pt')
            tokenized_labels = self.tokenizer.encode_plus(test_question,truncation=False,add_special_tokens=False,return_tensors='pt')
        else:
            raise NotImplementedError
        return {
                'input_ids': tokenized_example.input_ids.squeeze(),
                'labels': tokenized_labels.attention_mask.squeeze(),
                "metadata":entry
                        
                    } 	 	 	
                    