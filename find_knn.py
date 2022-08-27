import hydra.utils as hu 
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import tqdm
from torch.utils.data import DataLoader
from src.data.collators import DataCollatorWithPaddingAndCuda
import faiss
import numpy as np
import json
class KNNFinder:
    def __init__(self,cfg) -> None:
        self.cuda_device = cfg.cuda_device
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer,device = self.cuda_device)
        self.dataloader = DataLoader(self.dataset_reader,batch_size=50,collate_fn=co)
        self.model = hu.instantiate(cfg.model).to(self.cuda_device)
        self.index = faiss.read_index(cfg.index_path)
        self.output_path = cfg.output_path
        self.is_train = cfg.dataset_split=="train"
        


    def forward(self):
        res_list = []
        for i,entry in enumerate(tqdm.tqdm(self.dataloader)):
            with torch.no_grad():
                res = self.model(**entry)
            res = res.cpu().detach().numpy()
            res_list.extend([{"res":r,"metadata":m} for r,m in  zip(res,entry['metadata'])])

        return res_list

    def search(self,entry,k=50):
        res = np.expand_dims(entry['res'],axis=0)
        near_ids = self.index.search(res, k+1)[1][0]
        near_ids = near_ids[1:] if self.is_train else near_ids
        return [{"id":int(a)} for a in near_ids[:k]]
    
    
    def find(self):
        res_list = self.forward()
        data_list = []
        for entry in tqdm.tqdm(res_list):
            data = self.dataset_reader.task.dataset[entry['metadata']['id']]

            data['ctxs'] = self.search(entry)
            data_list.append(data)
        with open(self.output_path,"w") as f:
            json.dump(data_list,f)




#python find_knn.py  index_path=$PWD/data/break_mpnet_q.bin output_path=$PWD/data/break_mpnet_q_prompts.json cuda_device=1 setup_type=q dataset_split=validation task_name=break 
@hydra.main(config_path="configs",config_name="knn_finder")
def main(cfg):
    print(cfg)
    knn_finder = KNNFinder(cfg)
    knn_finder.find()


if __name__ == "__main__":
    main()