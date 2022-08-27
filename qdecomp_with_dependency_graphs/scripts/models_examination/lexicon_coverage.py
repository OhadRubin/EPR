from typing import List
import os
import json
import csv
from ast import literal_eval
from itertools import chain
from collections import Counter
import spacy
import pandas as pd
import argparse


def lexicon_coverage(eval_files: List[str], dataset_dir='datasets/Break/QDMR'):
    dataset_file = os.path.join(dataset_dir, 'dev_seq2seq.csv')

    question_ids, source_tokens, target_tokens, allowed_tokens = get_data_samples(dataset_file)

    def is_allowed_coverage_statistics(title:str, eval_file: str):
        print(title)
        eval_df = pd.read_csv(eval_file)
        match_metric = eval_df['normalized_exact_match'].tolist()
        predictions = [tokenize(x) for x in eval_df['prediction']]
        check_is_allowed(question_ids, source_tokens, target_tokens, allowed_tokens, predictions, match_metric)
        print('########################################################################')

    common_pref = os.path.commonpath(eval_files)+os.path.sep
    for x in eval_files:
        is_allowed_coverage_statistics(x.replace(common_pref,''), x)


_nlp = spacy.load('en_core_web_sm', disable=['ner'])
_tokenizer = _nlp.Defaults.create_tokenizer(_nlp)
def tokenize(text: str):
    return [t.text.lower() for t in _tokenizer(text)]
    # return [x.lower() for x in text.split()]


def get_data_samples(dataset_file: str):
    df = pd.read_csv(dataset_file)

    source_tokens = []
    target_tokens = []
    allowed_tokens = []
    for _, row in df.iterrows():
        source_tokens.append(tokenize(row['question_text']))
        target_tokens.append(tokenize(row['decomposition']))
        allowed_tokens.append(list(chain(*[([y.strip().lower() for y in x.split()] if type(x)==str else [x]) for x in literal_eval(row['lexicon_tokens'])]))
                              + source_tokens[-1]
                              + target_tokens[-1]
                              + ['@@sep@@'])

    return df['question_id'].to_list(), source_tokens, target_tokens, allowed_tokens


def check_is_allowed(question_ids, source_tokens, target_tokens, allowed_tokens, predictions, match_metric):
    diffs = []
    for p,s,t,a in zip(predictions, source_tokens, target_tokens, allowed_tokens):
        diff = sum(x not in a for x in p)
        diffs.append(diff)

    mean_not_allowed = sum(diffs)/len(diffs)
    mean_tokens = sum(len(x) for x in predictions)/len(predictions)
    print(f'mean: {mean_not_allowed} / {mean_tokens} ({mean_not_allowed/mean_tokens*100}%)')

    pos_diffs = [(i,x) for i,x in enumerate(diffs) if x>0]
    print(f'samples: {len(pos_diffs)} / {len(diffs)} ({len(pos_diffs)/len(diffs)*100}%)')
    mean_not_allowed = sum([x for _,x in pos_diffs]) / len(pos_diffs)
    mean_tokens = sum(len(predictions[i]) for i,_ in pos_diffs)/len(pos_diffs)
    print(f'samples mean: {mean_not_allowed} / {mean_tokens} ({mean_not_allowed/mean_tokens*100}%)')

    def print_hist(list_):
        items = sorted(Counter(list_).items(), key=lambda x: x[1], reverse=True)
        for k, v in items:
            print(f'\t{k}: {v}')

    print(f'histogram:')
    print_hist(diffs)

    print(f'tokens histogram:')
    print_hist(list(chain(*[[x for x in predictions[i] if x not in allowed_tokens[i]] for i, _ in pos_diffs])))

    # [([x for x in preds[i] if x not in allowed_tokens[i]], preds[i], source_tokens[i], target_tokens[i])
    #  for i, _ in pos_diffs]

    # print eval metric
    total_true = sum(match_metric)
    pos_indexes = [i for i, _ in pos_diffs]
    pos_eval = [match_metric[i] for i in pos_indexes]
    pos_true = sum(pos_eval)
    print(f'total match:{total_true/len(match_metric)} ({total_true}/{len(match_metric)})')
    print(f'not-allowed match:{pos_true/len(pos_eval)} ({pos_true}/{len(pos_eval)})')

    # print samples
    print('samples:')
    for i in pos_indexes:
        print(question_ids[i])
        print('\tques:'+' '.join(source_tokens[i]))
        print('\tgold:'+' '.join(target_tokens[i]))
        print('\tpred:'+' '.join(predictions[i]))
        print('\tmiss:'+','.join([x for x in predictions[i] if x not in allowed_tokens[i]]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='examine models lexicon coverage')
    parser.add_argument('-i', '--input_eval_file', type=str, nargs='+',
                        help='eval_full.csv files of the examined models')
    args = parser.parse_args()
    lexicon_coverage(args.input_eval_file)
