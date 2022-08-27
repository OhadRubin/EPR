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
import os





@hydra.main(config_path="configs",config_name="client")
def main(cfg):
    print(cfg)
    client = GPT3Client(api_key=os.environ["OPENAI_TOKEN"])

    async def get_pred(entry_list):
        prompt = [x['enc_text'] for x in entry_list]

        args = {
        "prompt": prompt,
        "max_tokens": 280,
        "stop":["\n"],
        "echo":False,
        "logprobs":1
            
        }
        results = (
                await client.completions_rate_limited(cfg.engine, args)  # type: ignore
            ).json()
        for i,x in enumerate(entry_list):
            x['generated'] = results['choices'][i]['text']
        return  entry_list

    async def run(data_list):
        task_list = []
        
        for i,prompt in enumerate(more_itertools.chunked(data_list,cfg.batch_size)):
            task = asyncio.create_task(get_pred(prompt))
            task_list.append(task)
        responses = [await f
                    for f in tqdm.tqdm(asyncio.as_completed(task_list), total=len(task_list))]
        return responses


    def run_main(cfg):
        dataset_reader = hu.instantiate(cfg.dataset_reader)
        
        idx_list = list(range(len(dataset_reader)))
        random.Random(42).shuffle(idx_list)
        idx_list = idx_list[:1000]
        data_list = [dataset_reader[x]['metadata'] for x in (idx_list if "stop" not in cfg else idx_list[:20])]
        res = asyncio.run(run(data_list))
        res = list(more_itertools.collapse(res,levels=1))
        with open(cfg.output_file, "w") as f:
            json.dump(res,f)
    
    run_main(cfg)
    
        
#python gpt3_client.py prompt_file=$PWD/data/random_smcalflow_valid.json      task_name=smcalflow     output_file=$PWD/data/test.json  engine=davinci-codex +stop=true
if __name__ == "__main__":
    main()
        

    

