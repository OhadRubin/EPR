import argparse
import os
import csv
import json

from tempfile import TemporaryDirectory
import tarfile
from shutil import copyfile

from _jsonnet import evaluate_file 
import torch
from allennlp.nn import util
from allennlp.common.params import Params

def main(args):
    with TemporaryDirectory() as tmpdirname:
        with tarfile.open(args.src_model, mode='r:gz') as input_tar:
            print('Extracting model...')
            input_tar.extractall(tmpdirname)
        
        Params.from_file(args.config).to_file(os.path.join(tmpdirname, 'config.json'))

        if args.key_mappings is not None:
            print('Updating key mappings...')
            model_state = torch.load(os.path.join(tmpdirname, 'weights.th'), map_location=util.device_mapping(args.cuda_device))
            with open(args.key_mappings) as csvfile:
                key_mappings = csv.reader(csvfile, delimiter=',')
                for mapping in key_mappings:
                    if len(mapping) == 0:
                        continue
                    if len(mapping) == 1:
                        old_key = mapping[0]
                        del model_state[old_key]
                    else:
                        old_key, new_key = mapping[0], mapping[1]
                        temp = model_state[old_key]
                        del model_state[old_key]
                        model_state[new_key] = temp
            torch.save(model_state, os.path.join(tmpdirname, 'weights.th'))

        with tarfile.open(args.dest_model, "w:gz") as output_tar:
            print('Archiving model...')
            output_tar.add(tmpdirname, arcname ="")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("src_model", type=str, help="Source model.tar.gz to modify")
    parse.add_argument("config", type=str, help="Configuration file to use")
    parse.add_argument("dest_model", type=str, help="Output path")
    parse.add_argument("--key-mappings", type=str, default=None, help="CSV file mapping old keys to new keys")
    parse.add_argument("--cuda-device", type=int, default=0, help="GPU to load the model onto")
    args = parse.parse_args()

    main(args)