import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

def no_init(loading_code):
    def dummy(self):
        return
    
    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy
    
    result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]
    
    return result


def get_model(**kwargs):
    return no_init(lambda: AutoModelForCausalLM.from_pretrained(**kwargs))