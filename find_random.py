import hydra
import hydra.utils as hu 

import tqdm
import numpy as np
import json
# from src.utils.app import App
from src.dataset_readers.bm25_tasks import BM25Task
from dataclasses import dataclass
import random



class RandomFinder:
    def __init__(self,cfg) -> None:
        self.output_path = cfg.output_path
        self.task_name = cfg.task_name
        assert cfg.dataset_split in ["train","validation","test"]
        self.is_train = cfg.dataset_split=="train"
        self.setup_type = "a"
        
        self.task = BM25Task.from_name(cfg.task_name)(cfg.dataset_split,
                                                        self.setup_type)
        print("started creating the corpus")
        self.corpus = self.task.get_corpus()
        print("finished creating the corpus")






def find(cfg):
    random_finder = RandomFinder(cfg)
    data_list = list(random_finder.task.dataset)
    idx_list = list(range(len(random_finder.task.get_corpus())))
    
    for element in tqdm.tqdm(data_list):
        element['ctxs'] = [{"id":int(a)} for a in random.sample(idx_list,k=200)]
    return data_list


@hydra.main(config_path="configs",config_name="random_finder")
def main(cfg):
    print(cfg)
    
    data_list = find(cfg)
    # print(data_list)
    with open(cfg.output_path,"w") as f:
        json.dump(data_list,f)


if __name__ == "__main__":
    main()