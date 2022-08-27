import dataclasses
import os
import subprocess
import hydra
from hydra.utils import get_original_cwd
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from argparse import Namespace
import random
import namegenerator
import dateparser
import datetime as dt
import textwrap


SLURM_ARGS = {'job-name': '{exp_name}',
         'output': 'outputs/{exp_name}/out.txt',
         'error': 'outputs/{exp_name}/out.txt',
         'partition': '{partition}',
         'account': 'gpu-joberant',
         'time': '{time}',
         'nodes': '1',
         'exclude': 'n-201,n-101',
         'ntasks': '1',
         'gpus': '{gpus}'}

def htime_to_mins(htime):
    translation = {"m":1,"h":60,"d":1440}
    mult = translation[htime[-1]]
    return int(htime[:-1])*mult

def get_sbatch_preamble(**kwargs):
    str_list = ['#!/bin/bash'] 
    for key,val in SLURM_ARGS.items():
        str_list.append(f"#SBATCH --{key}={str(val).format(**kwargs)}")
    return str_list

def get_defaults(class_obj):
    return {k:v.default for k,v in class_obj.__dict__['__dataclass_fields__'].items()}

def get_experiment_name(experiment_name):
    i = 0
    while True:
        if not pathlib.Path(f"outputs/{experiment_name}_v{i}").exists():
            return f"{experiment_name}_v{i}"
        i += 1

compute_bm25_cmd = \
    'python find_bm25.py output_path=$PWD/data/{compute_bm25_outfile} \
    dataset_split=train setup_type={bm25_setup_type} task_name={dataset} +ds_size={ds_size} L={finder_L}'

run_scorer_cmd = \
    'accelerate launch --num_processes {gpus} --main_process_port {random_port} \
    scorer.py example_file=$PWD/data/{compute_bm25_outfile} setup_type=qa \
    output_file=$PWD/data/{run_scorer_outfile} batch_size=1 \
    +task_name={dataset} +dataset_reader.ds_size={ds_size} {premble_scr}'


train_retriever_cmd = \
    "python DPR/train_dense_encoder.py train_datasets=[epr_dataset] \
    train=biencoder_local output_dir=$PWD/experiments/{train_retriver_outfile} \
    datasets.epr_dataset.file=$PWD/data/{run_scorer_outfile} \
    datasets.epr_dataset.setup_type=qa  datasets.epr_dataset.hard_neg=true \
    datasets.epr_dataset.task_name={dataset} \
    datasets.epr_dataset.top_k={dpr_top_k} +gradient_accumulation_steps=1 train.batch_size={dpr_bs}\
    train.num_train_epochs={dpr_epochs}"

gen_emb_cmd = \
    "python DPR/generate_dense_embeddings.py model_file=$PWD/experiments/{train_retriver_outfile}/dpr_biencoder.{dpr_epochsm1} \
    ctx_src=dpr_epr shard_id=0 num_shards=1 out_file=$PWD/experiments/{train_retriver_outfile}/dpr_enc_index \
    ctx_sources.dpr_epr.setup_type=qa \
    ctx_sources.dpr_epr.task_name={dataset} +ctx_sources.dpr_epr.ds_size={ds_size}"

retrieve_prompts_cmd = \
    'python DPR/dense_retriever.py model_file=$PWD/experiments/{train_retriver_outfile}/dpr_biencoder.{dpr_epochsm1} \
    qa_dataset=qa_epr ctx_datatsets=[dpr_epr] datasets.qa_epr.dataset_split={split} \
    encoded_ctx_files=["$PWD/experiments/{train_retriver_outfile}/dpr_enc_index_*"]\
    out_file=$PWD/data/{retrieve_prompts_outfile} ctx_sources.dpr_epr.setup_type=qa \
    ctx_sources.dpr_epr.task_name={dataset} datasets.qa_epr.task_name={dataset}'

run_inference_cmd = \
    "accelerate launch --num_processes {gpus} --main_process_port {random_port} \
    inference.py prompt_file=$PWD/data/{retrieve_prompts_outfile} task_name={dataset} \
    output_file=$PWD/data/{run_inference_outfile} batch_size={inf_bs} max_length={inf_maxlen} {premble_inf}"




@dataclass
class EPRConfig:
    dataset: str
    time: str
    partition: str
    # model_name:str = "gptneo"
    ds_size: str  = """null"""
    exp_type: str = "epr"
    gpus: int = 4
    bm25_setup_type: str = "a"
    scr_model:str = "gptneo"
    dpr_epochs: int = 30
    dpr_bs: int = 60
    dpr_top_k: int = 5
    inf_bs: int = 10
    inf_model:str = "gptneo"
    inf_maxlen: int = 1950
    finder_L: int = 50
    split: str = "validation"
    compute_bm25: bool = False
    run_scorer: bool = False
    train_retriever: bool = True
    gen_emb: bool = True
    retrieve_prompts: bool = True
    run_dpr_pipeline: bool = True
    run_inference: bool = True
    
    _kwargs: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.run_dpr_pipeline:
            self.train_retriever = False
            self.gen_emb = False
            self.retrieve_prompts = False
        assert self.exp_type in ["epr","ubm25","cbr"]
        assert self.inf_model in ["gptj","gptneo"]
        assert self.scr_model in ["gptj","gptneo","codex"]
        self.time = htime_to_mins(self.time)
        self._kwargs['finder_diff'] = get_field_diff_from_default("finder",**self.__dict__)
        self._kwargs['scr_diff'] = get_field_diff_from_default("scr",**self.__dict__)
        self._kwargs['dpr_diff'] = get_field_diff_from_default("dpr",**self.__dict__)
        self._kwargs['inf_diff'] = get_field_diff_from_default("inf",**self.__dict__)
        self._kwargs.update({k:self.__dict__[k] for k in get_defaults(EPRConfig).keys() if k != "_kwargs"})
        self._kwargs['dpr_epochsm1'] = str(self.dpr_epochs - 1)
        self._kwargs['random_port'] = random.randint(21966,25000)
        
        
        self._kwargs['compute_bm25_outfile'] = 'bm25_{dataset}-{ds_size}_{finder_diff}{bm25_setup_type}_train.json'.format(**self._kwargs)
        self._kwargs['run_scorer_outfile'] = 'bm25_{dataset}-{ds_size}{finder_diff}{scr_diff}_a_train_scoredqa.json'.format(**self._kwargs)

        self._kwargs['path_dpr'] = '{exp_type}_{dataset}-{ds_size}{finder_diff}{scr_diff}{dpr_diff}'.format(**self._kwargs)
        
        self._kwargs['train_retriver_outfile'] = '{path_dpr}_a_train'.format(**self._kwargs)
        self._kwargs['retrieve_prompts_outfile'] = '{split}_{path_dpr}_a_train_prompts.json'.format(**self._kwargs)
        self._kwargs['run_inference_outfile'] = '{split}_{path_dpr}{inf_diff}_a_train_prede.json'.format(**self._kwargs)
        
        self.add_exp_type()
        gptj_str = 'model_name="EleutherAI/gpt-j-6B" +model.low_cpu_mem_usage=true +model.revision=float16'
        self._kwargs['premble_scr'] = gptj_str if self.scr_model=="gptj" else ''
        self._kwargs['premble_inf'] = gptj_str if self.inf_model=="gptj" else ''
        if self.inf_model=="gptj":
            self.inf_bs=2
        self._kwargs['exp_name'] = get_experiment_name("{path_dpr}{inf_diff}".format(**self._kwargs))


        
    def add_exp_type(self):
        if self.exp_type == "ubm25":
            self._kwargs['run_scorer_outfile'] =\
                'bm25_{dataset}-{ds_size}_q_train.json'.format(**self.__dict__)
            self.compute_bm25 = True
            self.bm25_setup_type = "q"
        elif self.exp_type == "cbr":
            self._kwargs['run_scorer_outfile'] =\
            'cbr_{dataset}_a_train.json'.format(**self.__dict__)
    
    

def get_field_diff_from_default(field,**kwargs):
        default_dict = get_defaults(EPRConfig)
        diff = '+'+"+".join(
            [
                f"{key}-{value}"
                for key, value in kwargs.items()
                if (not key.endswith("_diff") and key in default_dict and value != default_dict[key] and key.startswith(f"{field}_"))
            ]   
        )
        if diff=='+':
            return ''
        return diff
        
cs = ConfigStore.instance()
cs.store(name="config", node=EPRConfig)
def wrap(s):
    bs = ' \\\n\t '
    return bs.join(textwrap.wrap(s,break_long_words=False,break_on_hyphens=False))


def get_cmd_list(**kwargs):
    cmd_list = []
    if kwargs['compute_bm25'] and kwargs['split']!="test":
        cmd_list.append(compute_bm25_cmd)
    if kwargs['run_scorer'] and kwargs['split']!="test":
        cmd_list.append(run_scorer_cmd)
    if kwargs['train_retriever'] and kwargs['split']!="test":
        cmd_list.append(train_retriever_cmd)
    if kwargs['gen_emb'] and kwargs['split']!="test":
        cmd_list.append(gen_emb_cmd)
    if kwargs['retrieve_prompts']:
        cmd_list.append(retrieve_prompts_cmd)
    if kwargs['run_inference']:
        cmd_list.append(run_inference_cmd)
    cmd_list = [f"{cmd.format(**kwargs)} hydra.run.dir=$PWD/outputs/{kwargs['exp_name']}" for cmd in cmd_list]
    return [wrap(cmd) for cmd in cmd_list]

def split_join(key,field):
    return "_".join(key.split("_")[1:])

import pathlib



def get_kwargs(cfg):
    cfg = OmegaConf.to_object(cfg)
    cfg =OmegaConf.structured(cfg)
    return OmegaConf.to_container(cfg,throw_on_missing=True)


@hydra.main(config_path=None, config_name="config")
def main(cfg: EPRConfig) -> None:
    os.chdir(get_original_cwd())
    kwargs = get_kwargs(cfg)
    # subprocess.check_call(
    #     f"git ls-files | tar Tzcf - backup/{experiment_name}.tgz", shell=True
    # )
    kwargs.update(kwargs['_kwargs'])
    experiment_name = kwargs['exp_name']
    
    
    final_cmd_list = []
    if  'no_slurm' not in cfg:
        final_cmd_list.extend(get_sbatch_preamble(**kwargs))
        postfix_commands = [f"srun {x}" for x in get_cmd_list(**kwargs)]
        for x in postfix_commands:
            final_cmd_list.append(x)
            
    else:
        final_cmd_list.extend(get_cmd_list(**kwargs))

    if "debug" in cfg:
        print("\n".join(final_cmd_list))
    else:
        pathlib.Path(f"outputs/{experiment_name}").mkdir()
        with open(f"outputs/{experiment_name}/slurm.sh","w") as f:
            f.write("\n".join(final_cmd_list))
        if "dry_run" in cfg:
            print("\n".join(final_cmd_list))
            print("-"*20)
            print(f"{'sbatch' if 'no_slurm' not in cfg else 'bash'} outputs/{experiment_name}/slurm.sh")
        else:
            cmd_output = subprocess.run(f"sbatch outputs/{experiment_name}/slurm.sh",
                             shell=True,capture_output=True).stdout.decode("utf-8")
            print(cmd_output)
            job_id = cmd_output.strip().split(" ")[-1]

#  python run.py -m dataset=mtop dpr_epochs=30 gpus=4 time="10 minutes" +debug=true
#python run.py -m dataset=break,smcalflow,mtop dpr_epochs=30 gpus=4 time=24h ds_size=0.25,0.5 compute_bm25=true run_scorer=true +debug=true
#python run.py -m dataset=break,smcalflow,mtop dpr_epochs=30 gpus=4 time=24h dpr_top_k=10 compute_bm25=true run_scorer=true +debug=true
#python run.py -m dataset=break,smcalflow,mtop dpr_epochs=30 gpus=4 time=24h finder_L=50 compute_bm25=true run_scorer=true +debug=true

if __name__ == "__main__":
    main()
    