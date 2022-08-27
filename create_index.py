import hydra.utils as hu 
from hydra.core.hydra_config import HydraConfig
import hydra
import torch
import tqdm
from torch.utils.data import DataLoader
from src.data.collators import DataCollatorWithPaddingAndCuda
import faiss
import numpy as np

class Indexer:
    def __init__(self, cfg) -> None:
        self.cuda_device = cfg.cuda_device
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer,device = self.cuda_device)
        self.dataloader = DataLoader(self.dataset_reader,batch_size=cfg.batch_size,collate_fn=co)
        self.model = hu.instantiate(cfg.model).to(self.cuda_device)
        self.output_file = cfg.output_file
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(768))


    def create(self):
        res_list = []
        for entry in tqdm.tqdm(self.dataloader): 
            with torch.no_grad():
                # print(entry)
                metadata = entry.pop("metadata")
                res = self.model(**entry)
            id_list = np.array([m['id'] for m in metadata])
            self.index.add_with_ids(res.cpu().detach().numpy(), id_list)
        faiss.write_index(self.index, self.output_file)

        return res_list
            
        

#python create_index.py setup_name=qa output_file=$PWD/data/break_mpnet_qa.bin  task_name=break dataset_split=train cuda_device=6
@hydra.main(config_path="configs",config_name="create_index")
def main(cfg):
    print(cfg)
    indexer = Indexer(cfg)
    indexer.create()

if __name__ == "__main__":
    main()