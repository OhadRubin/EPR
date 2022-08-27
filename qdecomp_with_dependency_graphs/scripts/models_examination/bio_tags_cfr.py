import json
import os
import random
import pandas as pd


def compact_file(file_path:str):
    new_contents = []
    with open(file_path, 'rt') as f:
        for line in f.readlines():
            content = json.loads(line.strip())
            new_contents.append(json.dumps({'tags':content['tags'], 'words':content['words']})+'\n')
    dest_path = os.path.splitext(file_path)[0]+'_tags_and_words.json'
    with open(dest_path, 'wt') as f:
        f.writelines(new_contents)


def print_words_tags(predictions_file_path:str, dataset_file_path:str, tags_file_path:str, random_n:int = None):
    df = pd.read_csv(dataset_file_path)

    preds_df = pd.read_json(preds_file, orient='records', lines=True)
    df['pred_tags'] = preds_df['tags']
    df['tokens'] = preds_df['words']

    gold_tags = []
    with open(tags_file_path, 'rt') as f:
        for line in f.readlines():
            _, tags = list(zip(*[x.split('###') for x in line.strip().split('\t')]))
            gold_tags.append(list(tags))
    df['gold_tags'] = gold_tags


    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset]
        if random_n:
            ds_df = ds_df.sample(n=random_n)

        for _, x in ds_df.iterrows():
            tokens = x['tokens']
            pred_tags = x['pred_tags']
            gold_tags = x['gold_tags']
            max_length = max([len(y) for y in tokens+pred_tags+gold_tags])

            print(x['question_id'])
            print(x['question_text'])
            print(x['decomposition'])
            print(x['operators'])
            print()
            print('\t'.join([f'{{:>{max_length}}}'.format(y) for y in ['tokens:']+tokens]))
            print('\t'.join([f'{{:>{max_length}}}'.format(y) for y in ['pred:']+pred_tags]))
            print('\t'.join([f'{{:>{max_length}}}'.format(y) for y in ['gold:']+gold_tags]))
            print('--------------')


if __name__ == '__main__':
    dataset_file = 'datasets/Break/QDMR/dev.csv'
    bio_tags_file = 'datasets/Break/QDMR/dev_spans_BIO.txt'
    preds_file = 'tmp/datasets_Break_QDMR/dependencies_graph/spans-extraction--crf/spans-extraction--crf/eval/datasets_Break_QDMR_dev_spans_bio__preds.json'

    # dataset_file = 'datasets/Break_old-version/QDMR/dev.csv'
    # bio_tags_file = 'datasets/dependencies_graph/spans_dev.txt'
    # preds_file = 'tmp/dependencies_graph/spans_extraction__crf/eval/dev_preds.json'
    # compact_file(file_path= preds_file)
    print_words_tags(predictions_file_path=preds_file,
                     dataset_file_path=dataset_file,
                     tags_file_path=bio_tags_file
        ,
                     random_n=10)
