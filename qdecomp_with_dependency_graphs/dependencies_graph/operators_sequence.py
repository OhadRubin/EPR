import json
import os
from ast import literal_eval
import pandas as pd

def create_operators_seq(dataset_file: str):
    df = pd.read_csv(dataset_file)
    df['operators_seq'] = df['operators'].apply(lambda x: ' '.join(literal_eval(x)))
    df[['question_text', 'operators_seq']].to_csv(os.path.splitext(dataset_file)[0]+'_operators_seq2.tsv',
                                                  sep='\t', header=False, index=False)

def eval_operators_seq(dataset_file: str, predictions_file:str):
    df = pd.read_csv(dataset_file)
    preds = []
    with open(predictions_file, 'rt') as f:
        for line in f.readlines():
            content = json.loads(line.strip())
            pred = content['predicted_tokens']
            preds.append(pred)
    df['gold_operators_seq'] = df['operators'].apply(lambda x: literal_eval(x))
    df['predictions'] = preds
    df['exact_match'] = df['gold_operators_seq']==df['predictions']
    base_name = os.path.splitext(predictions_file)[0]+'__eval'
    df.to_csv(base_name+'.csv', index=False)
    summary = df.mean().round(3).to_dict()
    with open(base_name+'_summary.json', 'wt') as f:
        json.dump(summary,f, indent=2, sort_keys=True)
    print(summary)


if __name__ == '__main__':
    # create_operators_seq('datasets/Break/QDMR/dev.csv')
    eval_operators_seq('datasets/Break/QDMR/dev.csv',
                       'tmp/datasets_Break_QDMR/dependencies_graph/operators-seq--seq2seq/operators-seq--seq2seq/eval/datasets_Break_QDMR_dev_operators_seq__preds.json')