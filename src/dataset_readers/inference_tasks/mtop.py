import re
from datasets import load_dataset
from src.utils.dataset_utils import load_train_dataset

import json
from src.utils.tokenizer_utils import get_length




def set_length(example, idx,**kwargs):
    tokenizer = kwargs['tokenizer']
    q_field = example['question']
    a_field = example['logical_form']
    prompt_qa = f"{q_field}\t{a_field}"
    example['prompt_qa'] = prompt_qa
    example['prompt_len'] = get_length(tokenizer,prompt_qa)
    return example

class MtopInferenceTask:
    name = "mtop"
    def __init__(self, prompt_file, tokenizer,ds_size=None):
        self.prompt_file = prompt_file
        with open(self.prompt_file) as f:
            self.prompts = json.load(f)
        dataset = load_dataset("iohadrubin/mtop",name="mtop")
        self.hf_dataset = load_train_dataset(dataset,size=ds_size,listify=False)
        self.hf_dataset = self.hf_dataset.map(set_length,with_indices=True,fn_kwargs={'tokenizer':tokenizer})
        self.training_dataset = list(self.hf_dataset)
        self.postfix = ""
        
    @classmethod
    def postproccess(cls, string):
        return string

    def get_fields(self, entry):
        answer = entry['logical_form'] if 'logical_form' in entry else entry['answers'][0]
        idx_list =[p['id'] for p in entry['ctxs']]
        prompts = self.hf_dataset.select([p['id'] for p in entry['ctxs']])
        return entry['question'],answer,prompts['prompt_qa'],prompts['prompt_len'],idx_list

        