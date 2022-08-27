import re
from datasets import load_dataset
import json
from src.utils.app import App
from src.utils.dataset_utils import load_train_dataset




field_getter = App()

def norm(text):
    return (" ".join(text.split(";"))).split(" ")

@field_getter.add("q")
def get_question(entry):
    if "question" in entry:
        question = entry['question']
    else:
        question = entry['question_text']
    return BreakBM25Task.norm(question)

@field_getter.add("qa")
def get_qa(entry):
    if "question" in entry:
        question = entry['question']
    else:
        question = entry['question_text']
    return BreakBM25Task.norm(f"{question} {entry['decomposition']}")

@field_getter.add("a")
def get_decomp(entry):
    return BreakBM25Task.norm(entry['decomposition'])


class BreakBM25Task:
    name = "break"
    def __init__(self, dataset_split, setup_type, ds_size=None):
        self.setup_type = setup_type
        self.get_field = field_getter.functions[self.setup_type]
        self.dataset_split = dataset_split
        dataset = load_dataset("break_data","QDMR")
        self.train_dataset = load_train_dataset(dataset,size=ds_size)
        if self.dataset_split=="train":
            self.dataset = self.train_dataset 
        else:
            self.dataset = list(dataset[self.dataset_split])
        self.corpus = None

    def get_corpus(self):
        if self.corpus is None:
            self.corpus = [ self.get_field(entry) for entry in self.train_dataset]
        return self.corpus

    @classmethod
    def norm(cls,text):
        return (" ".join(text.split(";"))).split(" ")
    

        