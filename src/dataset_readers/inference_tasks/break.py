import re
from datasets import load_dataset
from src.utils.dataset_utils import load_train_dataset

import json
from src.utils.tokenizer_utils import get_length

def remove_double_space(string):
    return re.sub("[ ]{2,}", " ", string)

def reformat(text):
    return " ".join([f"{i+1}#) {x.strip()}" for i,x in enumerate(text.split(";"))])


def set_length(example, idx,**kwargs):
    tokenizer = kwargs['tokenizer']
    q_field = example['question_text']
    a_field = reformat(example['decomposition'])
    prompt_qa = remove_double_space(f"{q_field}\t{a_field}")
    example['prompt_qa'] = prompt_qa
    example['prompt_len'] = get_length(tokenizer,prompt_qa)
    return example

class BreakInferenceTask:
    name = "break"
    def __init__(self, prompt_file, tokenizer, reverse=False,ds_size=None):
        self.prompt_file = prompt_file
        # self.length_file = length_file
        # with open(self.length_file) as f:
        #     self.lengths_by_qid = json.load(f)
        #     self.lengths_by_nid = {i:v for i,(k,v) in enumerate(self.lengths_by_qid.items())}
        with open(self.prompt_file) as f:
            self.prompts = json.load(f)
        dataset = load_dataset("break_data", "QDMR")
        self.hf_dataset = load_train_dataset(dataset,size=ds_size,listify=False)
        
        
        self.hf_dataset = self.hf_dataset.map(set_length,with_indices=True,fn_kwargs={'tokenizer':tokenizer})
        self.training_dataset = list(self.hf_dataset)
        self.reverse = reverse
        self.postfix = "1#)"

    @classmethod
    def reformat(self,text):
        return " ".join([f"{i+1}#) {x.strip()}" for i,x in enumerate(text.split(";"))])

    @classmethod
    def renorm(self,text):
        text = text.split("\n")[0]
        text = re.sub("[\d]+\#\) ",";", text)
        return text

    @classmethod
    def remove_double_space(cls,string):
        return re.sub("[ ]{2,}", " ", string)

    @classmethod
    def postproccess(cls, string):
        return cls.remove_double_space(string)
        

    def get_fields2(self, entry):
        question = entry['question'] if 'question' in entry else entry['question_text']
        
        if "decomposition" in entry:
            decomposition = entry['decomposition']
            near_examples = entry['near_examples']
            prompts_list = [f"{near_example['question_text']}\t{self.reformat(near_example['decomposition'])}" for near_example in near_examples]
            lengths_list = [self.lengths_by_qid[near_example['question_id']]+10 for near_example in near_examples]
        else:
            decomposition = entry['answers'][0]
            near_examples = entry['ctxs']
            
            prompts_list = [self.training_dataset[int(near_example['id'])] for near_example in near_examples]
            prompts_list = [f"{near_example['question_text']}\t{self.reformat(near_example['decomposition'])}" for near_example in prompts_list]

            lengths_list = [self.lengths_by_nid[int(near_example['id'])]+10 for near_example in near_examples]
        return question, decomposition, prompts_list, lengths_list

    def get_fields(self, entry):
        question = entry['question'] if 'question' in entry else entry['question_text']
        answer = entry['decomposition'] if 'decomposition' in entry else entry['answers'][0]
        idx_list =[p['id'] for p in entry['ctxs']]
        prompts = self.hf_dataset.select([p['id'] for p in entry['ctxs']])
        return question,answer,prompts['prompt_qa'],prompts['prompt_len'],idx_list