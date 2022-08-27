import hydra
import hydra.utils as hu 

from hydra.core.hydra_config import HydraConfig
import tqdm
import numpy as np
import json
from rank_bm25 import BM25Okapi
# from src.utils.app import App
from src.dataset_readers.bm25_tasks import BM25Task
from dataclasses import dataclass
import multiprocessing

# global_context = {}




    # def __post_init__(self):
    #     self.converter = QDMRToQDMRStepTokensConverter()
    #     self.matcher = LogicalFromStructuralMatcher()
    #     self.scorer = NormalizedGraphMatchScorer()

class BM25Finder:
    def __init__(self,cfg) -> None:
        self.output_path = cfg.output_path
        self.task_name = cfg.task_name
        assert cfg.dataset_split in ["train","validation","test"]
        self.is_train = cfg.dataset_split=="train"
        self.L = cfg.L
        self.setup_type = cfg.setup_type
        assert self.setup_type in ["q","qa","a"]
        self.task = BM25Task.from_name(cfg.task_name)(cfg.dataset_split,
                                                        cfg.setup_type,
                                                        ds_size =  None if "ds_size" not in cfg else cfg.ds_size)
        print("started creating the corpus")
        self.corpus = self.task.get_corpus()
        self.bm25 = BM25Okapi(self.corpus)
        print("finished creating the corpus")

    
def search(tokenized_query,is_train,idx,L):
    # bm25 = global_context['bm25']
    
    bm25 = bm25_global
    scores = bm25.get_scores(tokenized_query)
    near_ids = list(np.argsort(scores)[::-1][:L])
    near_ids = near_ids[1:] if is_train else near_ids
    return [{"id":int(a)} for a in near_ids],idx
    
def _search(args):
    tokenized_query,is_train,idx,L = args
    return search(tokenized_query,is_train,idx,L)


class GlobalState:
    def __init__(self,bm25) -> None:
        self.bm25 = bm25


def find(cfg):
    knn_finder = BM25Finder(cfg)
    
    tokenized_queries = [knn_finder.task.get_field(entry) 
                for entry in knn_finder.task.dataset]
    # global_context['bm25'] = knn_finder.bm25
    
    def set_global_object(bm25):
        global bm25_global
        bm25_global = bm25

    pool = multiprocessing.Pool(processes=None,initializer=set_global_object,initargs=(knn_finder.bm25,))

    cntx_pre = [[tokenized_query,knn_finder.is_train,idx,knn_finder.L] for idx,tokenized_query in enumerate(tokenized_queries)]
    # cntx_post = pool.starmap_async(search, cntx_pre)
    
    data_list = list(knn_finder.task.dataset)
    # cntx_post = cntx_post.get(None)
    cntx_post = []
    with tqdm.tqdm(total = len(cntx_pre)) as pbar:
        for i, res in enumerate(pool.imap_unordered(_search, cntx_pre)):
            pbar.update()
            cntx_post.append(res)
    for ctx,idx in cntx_post:
        data_list[idx]['ctxs'] = ctx
    return data_list


#python find_bm25.py output_path=$PWD/data/test_bm25_1.json dataset_split=validation setup_type=qa task_name=break
@hydra.main(config_path="configs",config_name="bm25_finder")
def main(cfg):
    print(cfg)
    
    data_list = find(cfg)
    # print(data_list)
    with open(cfg.output_path,"w") as f:
        json.dump(data_list,f)


if __name__ == "__main__":
    main()