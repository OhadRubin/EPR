import asyncio
import string
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.lm_openai_gpt3 import GPT3Client,IncrementalOpenAIGPT3
import json
from omegaconf import OmegaConf
import more_itertools
import tqdm
import hydra.utils as hu 
import hydra
import random
from torch.utils.data import DataLoader
import os




@hydra.main(config_path="configs",config_name="api_scorer")
def main(cfg):
    print(cfg)
    client = GPT3Client(api_key=os.environ["OPENAI_TOKEN"])
    lm = IncrementalOpenAIGPT3(client=client,engine=cfg.engine)
    # assert "shard_id" in cfg
    # assert "n_shards" in cfg
    async def get_pred(prompt,dataset_reader):
        entry_list = [dataset_reader.dataset[x] for x in prompt]
        assert len(entry_list)==1
        entry = entry_list[0]
        question,answer,test_question,test_answer = dataset_reader.task.get_fields(entry)
        enc_text = f"{question}\t{answer}\n{test_question}\t"
        prefix_tokens = lm.tokenizer.encode(enc_text)
        tokenized_labels = lm.tokenizer.encode(test_answer)

        
        res_list = []
        for i,x in enumerate(entry_list):

            x['score'] = await lm.logprob_of_completion(prefix_tokens, tokenized_labels)
            res_list.append(x)

        return  res_list

    async def run(idx_list,dataset_reader):
        task_list = []
        assert cfg.batch_size == 1
        for i,prompt in enumerate(more_itertools.chunked(idx_list,cfg.batch_size)):
            task = asyncio.create_task(get_pred(prompt,dataset_reader))
            task_list.append(task)
        responses = [await f
                    for f in tqdm.tqdm(asyncio.as_completed(task_list), total=len(task_list))]
        return responses


    def run_main(cfg):
        assert "shard_id" in cfg
        assert "n_shards" in cfg
        
        dataset_reader = hu.instantiate(cfg.dataset_reader)
        
        idx_list = list(range(len(dataset_reader)))
        # random.Random(42).shuffle(idx_list)
        # print(len(idx_list))
        # return None
        # dataset = dataset_reader.dataset
        
        if "stop"  in cfg:
            idx_list = idx_list[:200]
        else:
            idx_list = more_itertools.divide(cfg.n_shards,idx_list)[cfg.shard_id]
        
        # dataloader = DataLoader(dataset,batch_size=cfg.batch_size,num_workers=5)
        # data_list = [x for x in dataloader]
        # else:
            
        # data_list = [dataset[x] for x in (idx_list if "stop" not in cfg else idx_list[:200])]
        print("starting")
        res = asyncio.run(run(idx_list,dataset_reader))
        res = list(more_itertools.collapse(res,levels=1))
        with open(cfg.output_file, "w") as f:
            json.dump(res,f)
    
    run_main(cfg)
    
# python gpt3_scorer.py example_file=$PWD/data/bm25_break_a_train.json setup_type=qa output_file=$PWD/data/test_scorer.json batch_size=1 +task_name=break engine=davinci-codex +stop=true 

#python gpt3_scorer.py prompt_file=$PWD/data/random_smcalflow_valid.json      task_name=smcalflow     output_file=$PWD/data/test.json  engine=davinci-codex +stop=true
if __name__ == "__main__":
    main()
# (prompt) >> python gpt3_scorer.py example_file=$PWD/data/bm25_break_a_train.json setup_type=qa output_file=$PWD/data/test_scorer.json batch_size=1 +task_name=break engine=davinci-codex           
    

