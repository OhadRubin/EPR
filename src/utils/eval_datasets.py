from src.utils.app import App
import json
import pandas as pd
from src.utils.eval_many import eval_many, eval_many_mtop,eval_many_smcalflow
from src.utils.cache_util import BufferedJsonWriter,BufferedJsonReader
import requests
import re
import numpy as np
import httpx
import asyncio

def get_model(file_name):
    if "cbr" in file_name:
        return "cbr"
    elif "epr_" in file_name:
        return "epr"
    elif "random_" in file_name:
        return "random"
    else:
        return "bm25"

def get_lm(file_name):
    if "codex" in file_name:
        return "codex"
    elif "gpt3" in file_name:
        return "gpt3"
    elif "gptj" in file_name:
        return "gptj"
    else:
        return "gptneo"

def get_dataset(file_name):
    if "break" in file_name:
        return "break"
    elif "smcalflow" in file_name:
        return "smcalflow"
    elif "mtop" in file_name:
        return "mtop"
        
def get_example_count(file_name):
    a = re.search("-([0-9]+).*",file_name)
    return int(a.group(1))


def renorm(text):
    text = text.split("\n")[0]
    text = re.sub("[\d]+\#\) ",";", text)
    return text
app = App()

# @app.add("smcalflow")
# def add_smcalflow_acc(file_name,id_list=None):
#     correct_list = []
#     with open(file_name) as f:
#         data = json.load(f)

#         for line in data:
#             if id_list is not None and line['id'] not in id_list:
#                 continue
#             lf = line['lispress'] if 'lispress' in line else line['answers'][0]
#             correct_list.append(line['generated'].strip()==lf)
#     for entry,acc in zip(data,correct_list):
#         entry['acc'] =acc
#     return data

# async def api_eval(line_list):

#     res_list = []
#     async with httpx.AsyncClient() as client:
        
#         for pred,gold in line_list:
#             res = await client.post(url="http://127.0.0.1:8000",params={"pred":pred,"gold":gold})
#             res_list.append(res)
#     return [res.json() for res in res_list]
#     # assert isinstance(res,bool)

    
@app.add("smcalflow")
def add_smcalflow_acc(file_name,id_list=None):
    correct_list = []
    with open(file_name) as f:
        data = json.load(f)
        line_list = []
        for line in data:
            if id_list is not None and line['id'] not in id_list:
                continue
            lf = line['lispress'] if 'lispress' in line else line['answers'][0]
            line_list.append((line['generated'].split("<|endoftext|>")[0].strip(),lf))
    pred,gold = list(zip(*line_list))
    res_list = eval_many_smcalflow(pred,gold)    

    for entry,acc in zip(data,res_list):
        entry['acc'] =acc
    return data


@app.add("break")
def add_break_acc(path,id_list=None):
    with BufferedJsonReader(path) as f:
        df = pd.DataFrame(f.read())
    data = df.to_dict("records")
    question_field = "question" if "question" in  data[0] else 'question_text'
    zipped_data = []
    for entry in data:
        if id_list is not None and entry['id'] in id_list:
            continue
        generated =renorm(entry['generated'].split("\n")[0].split("<|endoftext|>")[0]).strip()
        decomposition = entry['decomposition'] if "decomposition" in entry else entry['answers'][0]
        
        zipped_data.append([entry[question_field],generated,decomposition])
    
    questions,pred,gold = list(zip(*zipped_data))
    acc_results = eval_many(questions,pred,gold)
    for entry,acc in zip(data,acc_results):
        entry['acc'] =acc
    return data

@app.add("mtop")
def add_mtop_acc(file_name,id_list=None):
    correct_list = []
    with open(file_name) as f:
        line_list = []
        data = json.load(f)

        for line in data:
            if id_list is not None and line['id'] not in id_list:
                continue
            lf = line['logical_form'] if 'logical_form' in line else line['answers'][0]
            line_list.append((line['generated'].split("<|endoftext|>")[0].strip(),lf))
    pred,gold = list(zip(*line_list))
    res_list = eval_many_mtop(pred,gold)
    for entry,acc in zip(data,res_list):
        entry['acc'] =acc
    return data


def get_answer(entry):
    if 'logical_form' in entry:
        return entry['logical_form']
    elif 'decomposition' in entry:
        return entry['decomposition']
    elif "lispress" in entry:
        return entry['lispress']
    elif "answers" in entry:
        return entry['answers'][0]
    

