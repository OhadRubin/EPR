from datasets import load_dataset
from src.utils.dataset_utils import load_train_dataset

import re
import json


class MtopScorerTask:
    name = "mtop"
    prompt_field = "ctxs"
    question_field = "question"
    def __init__(self,example_file,ds_size=None) -> None:
        dataset = load_dataset("iohadrubin/mtop",name="mtop")
        self.hf_dataset = load_train_dataset(dataset,size=ds_size)
        self.training_dataset = list(enumerate(self.hf_dataset))
        
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)
        

    def get_fields(self, entry,index=-1):
        test_question = entry['test_question']
        question = entry['question']
        logical_form = entry['logical_form']
        test_logical_form = entry['test_logical_form']
        return question,logical_form,test_question,test_logical_form


    @classmethod
    def remove_double_space(cls,string):
        return re.sub("[ ]{2,}", " ", string)
    @classmethod
    def reformat(cls,text):
        return " ".join([f"{i+1}#) {x.strip()}" for i,x in enumerate(text.split(";"))])
