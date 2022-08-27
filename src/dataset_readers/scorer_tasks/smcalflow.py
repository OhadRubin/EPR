from datasets import load_dataset
import re
import json
import random
from src.utils.dataset_utils import load_train_dataset



class SmcalflowScorerTask:
    name = "smcalflow"
    prompt_field = "ctxs"
    question_field = "user_utterance"
    def __init__(self,example_file,ds_size=None) -> None:
        dataset = load_dataset("iohadrubin/smcalflow",name="smcalflow")
        self.hf_dataset = load_train_dataset(dataset,size=ds_size)
        # self.hf_dataset = ['train']
        self.example_file = example_file
        with open(self.example_file) as f:
            self.data = json.load(f)
        idx_list = list(range(len(self.data)))
        random.Random(42).shuffle(idx_list)
        self.data = [self.data[x] for x in idx_list[:44000]]
        print(f"{len(self.data)} examples")
        
        self.training_dataset = list(enumerate(self.hf_dataset))

    def get_fields(self, entry,index=-1):
        test_question = entry['test_user_utterance']
        question = entry['user_utterance']
        lispress = entry['lispress']
        test_lispress = entry['test_lispress']
        return question,lispress,test_question,test_lispress


    @classmethod
    def remove_double_space(cls,string):
        return re.sub("[ ]{2,}", " ", string)
    @classmethod
    def reformat(cls,text):
        return " ".join([f"{i+1}#) {x.strip()}" for i,x in enumerate(text.split(";"))])
